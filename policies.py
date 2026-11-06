"""Routing policies and cache models, shared by the router and the simulator.

The router can't see inside the replicas, so it keeps its own picture of each one:
  pending  - prefill tokens sent there that haven't produced a first token yet
  cache    - an LRU of the prompt blocks it has sent there (an estimate: the real
             cache may have evicted them, or a shared tier may hold more)

Policies:
  least       send to the replica with the least pending prefill work
  prefix      send to the replica with the longest cached prefix (ties: least)
  prefix_cap  like prefix, but only among replicas whose pending work is within
              `cap` tokens of the least loaded one (SGLang's balance threshold)

Caches, for the simulator:
  Replica     LRU. Also the router's picture of a replica.
  ChainCache  LRU among the ends of prefix chains, as vLLM's GPU cache evicts
  Tier        a host tier, trimmed by tier_gc.py's own code on a simulated clock
"""
import heapq
from collections import Counter, OrderedDict

import tier_gc

BLOCK = 512


def block_keys(tokens, block=BLOCK):
    """Chained keys: key i names the whole prefix tokens[:block * (i + 1)]."""
    keys, h = [], 0
    for i in range(0, len(tokens) - block + 1, block):
        h = hash((h, tuple(tokens[i:i + block])))
        keys.append(h)
    return keys


class Replica:
    def __init__(self, capacity_blocks):
        self.pending = 0
        self.capacity = capacity_blocks
        self.cache = OrderedDict()

    def match(self, keys):
        n = 0
        while n < len(keys) and keys[n] in self.cache:
            n += 1
        return n

    def insert(self, keys):
        for k in keys:
            self.cache[k] = True
            self.cache.move_to_end(k)
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)


class ChainCache:
    """LRU over the ends of prefix chains: a block can be evicted only if no cached
    block extends it. A prefix is usable up to its first missing block, so evicting a
    chain's head wastes every block after it; evicting its tail costs one block.
    (vLLM's and SGLang's GPU caches already evict tail-first; tiers usually don't.)"""

    def __init__(self, capacity_blocks):
        self.capacity = capacity_blocks
        self.parent = {}           # cached block -> the block before it in its prompt
        self.children = Counter()  # cached block -> how many cached blocks extend it
        self.last_used = {}
        self.clock = 0
        self.ends = []             # heap of (last_used, block) for blocks with no children

    def match(self, keys):
        n = 0
        while n < len(keys) and keys[n] in self.parent:
            n += 1
        return n

    def insert(self, keys):
        for i, k in enumerate(keys):
            if k not in self.parent:
                self.parent[k] = keys[i - 1] if i else None
                if i:
                    self.children[keys[i - 1]] += 1
            self.clock += 1
            self.last_used[k] = self.clock
        if keys:  # the prompt's last block is now the end of its chain
            heapq.heappush(self.ends, (self.last_used[keys[-1]], keys[-1]))
        while len(self.parent) > self.capacity:
            self.evict_one()

    def evict_one(self):
        while True:
            t, k = heapq.heappop(self.ends)
            if k in self.parent and self.children[k] == 0 and self.last_used[k] == t:
                break  # skip stale entries: used again since, or extended since
        p = self.parent.pop(k)
        del self.last_used[k], self.children[k]
        if p is not None:
            self.children[p] -= 1
            if self.children[p] == 0:  # its parent is now the end of the chain
                heapq.heappush(self.ends, (self.last_used[p], p))


class Tier:
    """A host tier kept under budget the way tier_gc.py keeps it: every `interval`
    seconds, tier_gc.to_delete picks blocks to delete. A block is stored when it is
    computed and stamped when it is loaded; a block a request finds on its GPU isn't
    loaded, so it isn't stamped either (kvtier.py only sees loads)."""

    def __init__(self, capacity_blocks, gc="oldest", interval=1.0):
        self.capacity, self.gc, self.interval = capacity_blocks, gc, interval
        self.blocks = {}  # key -> tier_gc.Block
        self.window = tier_gc.Window()
        self.next_gc = interval

    def collect(self, now):
        """Run the GC passes that would have happened by `now`."""
        while self.next_gc <= now:
            for b in tier_gc.to_delete(list(self.blocks.values()), self.capacity, self.gc,
                                       self.window, self.next_gc):
                del self.blocks[b.path]
            self.next_gc += self.interval

    def match(self, keys, start):
        """How far the tier continues a prefix whose first `start` blocks are on the GPU.
        (vLLM looks up the tier from there on; keys are chained, so a block is valid
        whether or not the tier still has the ones before it.)"""
        n = start
        while n < len(keys) and keys[n] in self.blocks:
            n += 1
        return n

    def access(self, keys, gpu_hit, tier_hit, now):
        """A request found keys[:gpu_hit] on its GPU and keys[gpu_hit:tier_hit] here
        (counted in blocks), and computes the rest, which is then stored."""
        for k in keys[gpu_hit:tier_hit]:
            b = self.blocks[k]
            b.last_used, b.read = now, True
        for k in keys[tier_hit:]:
            if k not in self.blocks:
                self.blocks[k] = tier_gc.Block(k, 1, written=now, last_used=now, read=False)


def choose(policy, replicas, keys, cap=16384):
    n = range(len(replicas))
    if policy == "least":
        return min(n, key=lambda j: replicas[j].pending)
    if policy == "prefix":
        return max(n, key=lambda j: (replicas[j].match(keys), -replicas[j].pending))
    if policy == "prefix_cap":
        floor = min(r.pending for r in replicas)
        ok = [j for j in n if replicas[j].pending <= floor + cap]
        return max(ok, key=lambda j: (replicas[j].match(keys), -replicas[j].pending))
    raise ValueError(policy)
