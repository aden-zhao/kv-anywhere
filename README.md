# kv-anywhere

Route LLM requests by load and let a shared KV cache supply the prefix, instead of routing each request to the replica that already caches it. 4 vLLM replicas on NERSC Perlmutter A100s, replaying real chat and coding-agent traces.

## The idea

When several replicas serve one model, each keeps its own prefix cache on the GPU. The second turn of a conversation is cheap only if it lands on the replica that served the first turn, because only that replica has the first turn's KV. So routers usually send each request to the replica caching the longest part of its prompt (*prefix affinity*). That fights with load balancing.

DeepSeek's inference system (DeepInfer, 2026) does the opposite: its KV cache lives on a storage cluster that every engine can read, and "since any engine can hit any prefix, there is no prefix-affinity routing." This repo tests that choice at small scale.

![Three designs](docs/figures/designs.svg)

- **Prefix affinity** keeps a conversation on one replica. But conversations in a real trace share a system prompt, so one replica always has the longest match.
- **Least-loaded** balances well, but the second turn usually lands on a replica that has to recompute the whole conversation.
- **Least-loaded + shared tier** balances and still reuses the first turn: replica B loads A's KV, about 4x faster than recomputing it.

Affinity brings the request to its cache. A shared tier brings the cache to the request.

## What I found

- **Chat: routing by load plus a shared tier beats every GPU-only policy.** p99 time to first token drops from 16.7 s to 5.0 s at the load where GPU-only least-loaded saturates. Pure prefix affinity sends every request to one replica.
- **Sharing matters, not capacity.** The same tier split into private per-replica pieces adds nothing.
- **The payoff scales with how cheap loading is compared with recomputing.** Qwen3-8B stores 2.6x more KV per token than Qwen2.5-7B, and its gain shrinks from 3.3x to 1.3x.
- **Agents change the calculus.** Contexts are 100k+ tokens and almost all of each prompt repeats the previous one. Without a tier, affinity becomes the best policy. With a small tier, eviction hurts: a prefix is a chain, so evicting a session's oldest blocks loses all of it. With a tier big enough to never evict (2 TB on Lustre), plain least-loaded routing wins: sessions finish 4x faster than GPU-only, with 3x the throughput.
- **A step-by-step model of vLLM, calibrated on isolated requests, predicts all of this,** e.g. agent session times within 4% (median).

## How it works

- **Replicas.** 4 × vLLM (Qwen2.5-7B-Instruct), one per A100-40GB, each with its GPU prefix cache (about 380k tokens).
- **Shared tier.** vLLM's `OffloadingConnector` with a file-system tier, one directory for all four replicas, either in node-local memory (`/dev/shm`) or on Lustre (`$SCRATCH`, a distributed file system every node can read). A block's file name is a hash of its tokens, so a replica finds blocks another replica wrote. `tier_gc.py` keeps the tier under its budget (100 GB = 1.7M tokens) by deleting the oldest blocks.
- **Router.** `router.py`, a small proxy. It can't see inside the replicas, so it keeps its own guess of each (prefill tokens sent but not yet answered, and the prompt blocks it sent there) and picks a replica with a policy from `policies.py`: least-loaded, prefix affinity, or *capped affinity* (affinity only among replicas within 16k pending tokens of the least loaded, like SGLang's balance threshold).
- **Controls.** *Private tiers* (the same capacity, split per replica) separate "more cache" from "shared cache".
- **Chat workload.** The Mooncake *conversation* trace (Kimi, FAST'25) lists, for each real request, an arrival time and a hash id per 512-token block. `trace.py` turns each id into a fixed block of random tokens, so replayed prompts share exactly what the real ones shared. It keeps 30% of the conversations (each one whole) from a 10-minute window, 481 requests, and compresses time to set the load ("5x" plays the 10 minutes in 2).
- **Agent workload.** FreeInference (Harvard, CC-BY-4.0) logs production coding agents with session ids, start/end times, and an id per 16-token block. `agents.py` keeps sessions whose context peaks between 32k and 128k tokens and replays them *closed loop*: an agent sends its next step only when the previous one finishes, plus the recorded tool time (capped at 30 s). The load is the number of concurrent agents.
- **Clean runs.** Before every run the GPU caches, CPU caches, tier files and the router's guesses are cleared.

## Chat

![TTFT vs load](docs/figures/ttft.png)

At 5x, where GPU-only least-loaded starts to saturate:

| Design | p50 TTFT | p99 TTFT | Prompt tokens from cache |
|---|---|---|---|
| Least-loaded, GPU only | 2.5 s | 16.7 s | 10.5% |
| Prefix affinity, GPU only | 217 s | 449 s | 8.4% (all 481 requests on one replica) |
| Capped affinity, GPU only | 1.3 s | 7.9 s | 18.2% |
| Least-loaded + private tiers | 2.7 s | 16.7 s | 10.5% |
| Capped affinity + private tiers | 1.1 s | 5.8 s | 21.1% |
| **Least-loaded + shared tier** | **1.0 s** | **5.0 s** | **23.8%** |
| Capped affinity + shared tier | 0.9 s | 4.8 s | 25.8% |

- Private tiers don't help because a conversation's next turn usually lands on a different replica, whose private tier doesn't have it.
- Affinity on top of the shared tier helps only a little: once any replica can load any prefix, keeping a conversation in place saves a 0.1–0.2 s copy, not a recompute.
- Above 7x every design is overloaded; the shared tier still has the lowest TTFT.

![Where cached tokens came from](docs/figures/tokens.png)

**Tool agents (Mooncake *toolagent*, short sessions).** Most reuse here is three system prompts, and every replica keeps those on its GPU anyway, so even least-loaded serves 45% of prompt tokens from cache. At 5x, capped affinity (p99 3.5 s) and least-loaded + shared tier (3.6 s) tie, against 13.2 s for least-loaded alone. Pure affinity piled the traffic onto two replicas (p99 220 s, even at 2x).

**KV bytes per token.** The same conversation sweep with Qwen3-8B (147 KB of KV per token vs 57 KB):

| | Qwen2.5-7B | Qwen3-8B |
|---|---|---|
| KV per GPU / in a 100 GB tier | 382k / 1.7M tokens | ~140k / 0.68M tokens |
| Loading vs recomputing | 5x cheaper | 2x cheaper |
| p99 at the knee, least-loaded → + shared tier | 16.7 → 5.0 s (3.3x) | 12.5 → 9.6 s (1.3x) |

Bigger KV hurts three ways: each load moves more bytes, the tier holds fewer tokens, and the GPU holds fewer requests, so the cluster saturates sooner. DeepSeek notes that V4.1 "compresses the KV cache substantially", which is part of why their design works.

## Agents

| | Conversation | Agent session |
|---|---|---|
| Context per request | ~9k tokens | median 57k, up to 128k |
| Shared with the previous request | partial | 99.6% (median) |
| Cost of a cache miss | 0.57 s (8k tokens) | 24 s (120k tokens) |
| KV per session | ~0.5 GB | 3–7 GB, so a GPU holds ~3–7 sessions |

Loading a 120k-token prompt takes 1.7 s from `/dev/shm` and 3.3 s from Lustre (2x slower), against 24 s to recompute it.

![Agents](docs/figures/agents.png)

At 48 agents (4 GPUs, Qwen2.5-7B with its 128k-context setting):

| Design | Session time | Throughput (steps/s) | p99 TTFT |
|---|---|---|---|
| Least-loaded, GPU only | 1628 s | 0.52 | 52 s |
| Prefix affinity, GPU only | 973 s | 0.75 | 53 s |
| Least-loaded + `/dev/shm` tier (100 GB) | 859 s | 0.90 | 46 s |
| Prefix affinity + `/dev/shm` tier (100 GB) | 577 s | 1.25 | 41 s |
| Least-loaded + Lustre tier (100 GB) | 836 s | 0.92 | 48 s |
| **Least-loaded + Lustre tier (2 TB, nothing evicted)** | **402 s** | **1.55** | **16 s** |

- **Without a tier, affinity flips from worst to best.** Each session's own long prefix decides the match, so affinity just keeps a session where its KV is. Least-loaded on GPUs alone recomputes constantly (16% of prompt tokens from cache).
- **A small tier is hurt by eviction.** At 48 agents the sessions' KV doesn't fit in 100 GB. `tier_gc.py` deletes the oldest-written blocks (tmpfs doesn't record reads), which for agents is the worst choice: a session's oldest blocks are the start of its prompt, and a prefix match stops at the first missing block, so evicting the start loses the whole session. The model shows this: with oldest-first eviction it reproduces the measured hit rate (65.1% vs 65.6%); with least-recently-used eviction, what DeepSeek's GC does, it predicts 79%.
- **Capacity beats distance.** With the same 100 GB, Lustre performs like `/dev/shm`. With 2 TB nothing is evicted, and plain least-loaded routing becomes the best design by far. With enough storage, DeepSeek's "no prefix-affinity routing" wins for agents too.

For agents the tier is no longer an optimization: it decides how many agents a GPU can serve, and what matters most is that it's big enough to never evict a live session. All 22,820 agent requests succeeded.

## Why: a step-by-step model of a replica

`calibrate.py` measures one replica in isolation:

| Qwen2.5-7B on one A100 | |
|---|---|
| Recompute a prompt | 59 µs per token, plus attention over the context |
| Load it from the `/dev/shm` tier | 12 µs per token (5x cheaper) |
| One decode step | 12.4 ms + 29 ns per token of context in the batch |

A decode step reads the weights once (15 GB at ~1.5 TB/s ≈ 10 ms) plus every running request's KV (57 KB per token). Prefill is limited by compute and decode by memory bandwidth, and a busy replica does both in every step. The prefill fit, made on prompts up to 15k tokens, predicts the 120k-token recompute within 3% (24.4 s vs 23.9 s).

`sim.py` replays the same workloads through these costs, stepping each replica the way vLLM does: every running request decodes one token per step, the rest of the 2048-token budget goes to prefill chunks, a request starts only once its KV fits, tier copies queue one at a time per replica, the GPU cache evicts the ends of prompts first (vLLM frees a request's blocks tail-first), and the tier evicts whatever `tier_gc.py` did in that run. The router and caches are the same code as the real ones (`policies.py`).

One number is fitted per sweep: a 1.1–1.2x slowdown for what a loaded server adds on top of isolated requests (scheduler and API-server CPU, mixed batches), fitted on the GPU-only least-loaded runs. With it fixed, the model predicts the rest:

- chat: hit rates within 4 points; TTFT typically within 10–20%, off by up to 1.6x right at the knee, where small errors in capacity become large errors in queueing;
- agents: session time within 4% (median) and 21% (worst) over 25 runs; hit rates within 1 point (median) and 5 (worst). Most of these predictions were saved before the sweep finished (`results/agents_predictions.json`). An earlier version of the model evicted GPU blocks head-first and missed by up to 9 points; matching vLLM's tail-first order fixed it.

What it misses: on the toolagent trace, which replica a hot system prompt sticks to depends on where its first request happens to land. The real run put two of the three prompts on one replica; the model spread them across three.

```
python3 validate.py results/agents_qwen25 --calib results/calib_agents/calib_shared.json \
    --calib-lustre results/calib_agents/calib_lustre.json --kv-bytes 57344
```

## Running it (Perlmutter)

```
bash perlmutter/launch.sh 03:00:00 perlmutter/jobs/conv_qwen25.sh   # chat: one 4-GPU node, ~2 node-hours
bash perlmutter/launch.sh 01:30:00 perlmutter/jobs/agents_pilot.sh  # agents: 128k calibration, both tiers
bash perlmutter/launch.sh 04:00:00 perlmutter/jobs/agents_gpu.sh    # also agents_shared, _lustre, _lustre_big
python3 analyze.py results/conv_qwen25                              # one row per run
python3 figures.py results/conv_qwen25 --load 5
python3 figures.py results/agents_qwen25 --agents
python3 docs/make_diagram.py
```

`perlmutter/run.sh` expects a vLLM venv (`$SCRATCH/vllm-dev/env.sh`) and models in `$SCRATCH/huggingface`. Everything for one node runs from a single `srun`, because Slingshot allows only 3 network applications per node.

| File | What it is |
|---|---|
| `trace.py` | Mooncake trace → conversation-sampled workload → token ids with the trace's prefix sharing |
| `agents.py` | FreeInference agent trace → closed-loop agent sessions with the same prefix structure |
| `policies.py` | Routing policies and the router's guess of each replica |
| `router.py` | The routing proxy in front of the replicas |
| `replay.py` | Sends a workload to the router; records TTFT and each replica's cache hits |
| `sweep.py` | Starts 4 replicas per server config and runs every policy × load |
| `tier_gc.py` | Keeps the KV tier under its size budget, oldest blocks first |
| `calibrate.py` | Measures prefill, tier-load, and decode-step costs on one replica |
| `sim.py`, `validate.py` | The step-by-step model and its check against a sweep |
| `analyze.py`, `figures.py`, `docs/make_diagram.py` | Tables and figures |
