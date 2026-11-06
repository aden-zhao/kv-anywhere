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

Caches (used by the simulator for GPUs and tiers), by eviction policy:
  lru      Replica: least recently used
  oldest   Replica(fifo=True): oldest written, like tier_gc.py --policy oldest
  chain    ChainCache: least recently used among the ends of prefix chains
  protect  ProtectCache: like tier_gc.py --policy protect
"""
import heapq
from collections import Counter, OrderedDict

BLOCK = 512


def block_keys(tokens, block=BLOCK):
    """Chained keys: key i names the whole prefix tokens[:block * (i + 1)]."""
    keys, h = [], 0
    for i in range(0, len(tokens) - block + 1, block):
        h = hash((h, tuple(tokens[i:i + block])))
        keys.append(h)
    return keys


class Replica:
    def __init__(self, capacity_blocks, fifo=False):
        self.pending = 0
        self.capacity = capacity_blocks
        self.fifo = fifo  # evict oldest-written instead of least-recently-used
        self.cache = OrderedDict()

    def match(self, keys):
        n = 0
        while n < len(keys) and keys[n] in self.cache:
            n += 1
        return n

    def insert(self, keys, now=0.0):
        for k in keys:
            if k in self.cache and self.fifo:
                continue
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

    def insert(self, keys, now=0.0):
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


class ProtectCache(ChainCache):
    """Never evict a block used within the last `window` seconds; if only such blocks
    are left, don't store the rest of the new prompt (it bypasses the cache). The
    window is 3x the median time between a session's uses: about three of its steps.
    Under pressure, some sessions stay whole instead of every session partly evicted."""

    def __init__(self, capacity_blocks):
        super().__init__(capacity_blocks)
        self.used_at, self.gaps, self.window = {}, [], 60.0

    def insert(self, keys, now=0.0):
        n = self.match(keys)  # how long since this session last used its prefix?
        if n and keys[n - 1] in self.used_at:
            self.gaps = (self.gaps + [now - self.used_at[keys[n - 1]]])[-500:]
            if len(self.gaps) >= 20:
                self.window = 3 * sorted(self.gaps)[len(self.gaps) // 2]
        stored = 0
        for i, k in enumerate(keys):
            if k not in self.parent:
                if len(self.parent) >= self.capacity and not self.make_room(now):
                    break  # full of blocks in use: the rest of this prompt isn't stored
                self.parent[k] = keys[i - 1] if i else None
                if i:
                    self.children[keys[i - 1]] += 1
            self.clock += 1
            self.last_used[k] = self.clock
            self.used_at[k] = now
            stored = i + 1
        if stored:
            heapq.heappush(self.ends, (self.last_used[keys[stored - 1]], keys[stored - 1]))

    def make_room(self, now):
        while self.ends:
            t, k = self.ends[0]
            if not (k in self.parent and self.children[k] == 0 and self.last_used[k] == t):
                heapq.heappop(self.ends)  # stale entry
                continue
            if now - self.used_at[k] < self.window:
                return False
            self.evict_one()
            del self.used_at[k]
            return True
        return False


def make_cache(capacity_blocks, eviction="lru"):
    if eviction == "chain":
        return ChainCache(capacity_blocks)
    if eviction == "protect":
        return ProtectCache(capacity_blocks)
    return Replica(capacity_blocks, fifo=eviction == "oldest")


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
