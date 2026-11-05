"""Routing policies, shared by the router and the simulator.

The router can't see inside the replicas, so it keeps its own picture of each one:
  pending  - prefill tokens sent there that haven't produced a first token yet
  cache    - an LRU of the prompt blocks it has sent there (an estimate: the real
             cache may have evicted them, or a shared tier may hold more)

Policies:
  least       send to the replica with the least pending prefill work
  prefix      send to the replica with the longest cached prefix (ties: least)
  prefix_cap  like prefix, but only among replicas whose pending work is within
              `cap` tokens of the least loaded one (SGLang's balance threshold)
"""
from collections import OrderedDict

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
