# DFRR Kernel Design

> Status: Tier 1 implemented; Tiers 2-4 are design explorations
> Date: 2026-03-21
> Context: Post collection-loader-actor hardening
> See also: [Mutation & Concurrency Analysis](../research/dfrr-mutation-concurrency.md)

## Problem Statement

After incremental reindex (Merkle-based change detection + selective re-embed + upsert), the DFRR overlay state built on top of the HNSW graph becomes stale. The base VectorIndex handles incremental inserts natively, but the DFRR structures (RCM-ordered graph, binary vectors, rank index) require a full rebuild (~34s for 380K vectors).

## Current Flow

```
Incremental reindex:
  File change -> tree-sitter re-split -> Merkle diff -> re-embed -> upsert
  VectorIndex HNSW graph updated incrementally
  DFRR state stale (built from old graph topology)

First search after reindex:
  DfrrKernel::ensure_ready() detects stale state (node_count mismatch)
  -> Full DFRR rebuild: ~34s for 380K vectors
  -> Persists to disk for future sessions
```

## Why Incremental DFRR Is Hard

DFRR builds four interdependent structures:

| Structure | Purpose | Incremental Difficulty |
|---|---|---|
| FlatGraph (HNSW adjacency) | Graph traversal | Easy -- HNSW supports node insertion |
| RCM reordering | Cache-friendly memory layout | Hard -- global property, one insert can shift optimal positions for all neighbors |
| BinaryVectors (BQ1) | Hamming-distance pruning | Medium -- threshold calibration depends on dataset distribution |
| RankIndex (clustering) | Distance-free re-ranking | Medium-Hard -- cluster centroids + rank distributions shift |

The core coupling: RCM reorder invalidates vector positions, which invalidates binary vectors, which invalidates rank index positions, which invalidates node-to-ID map.

## Tiered Implementation Plan

### Tier 1: Background Rebuild via Actor Prewarm (1-2 days) -- Implemented

Wire a Reload command into the actor. After incremental reindex completes, trigger background DFRR rebuild while old state serves queries. Atomic swap when ready.

- No quality degradation
- Time to first query after reindex = 0 (old data serves)
- Fresh DFRR available ~34s later
- Uses existing actor infrastructure

### Tier 2: Two-Tier Search -- Growing + Sealed Segments (1 week)

Standard technique from production ANN systems (SPFresh/LIRE, Milvus, Pinecone):

```
Sealed segment: Full DFRR-optimized graph (old vectors)
Growing segment: Small brute-force buffer (new vectors since last build)
Search: query both -> merge top-K -> return
Compaction: periodically seal growing -> full DFRR rebuild
```

### Tier 3: Curvature-Guided Incremental RCM (2-4 weeks)

Novel approach combining:
- Levada et al. local curvature estimation for insertion sensitivity
- SPFresh LIRE bounded work and partition-local rebalancing
- Actor execution mechanism (background rebalance, atomic swap)

### Tier 4: Fully Incremental DFRR (research, open problem)

Online maintenance of all four structures with convergence guarantees.

---

## Curvature-Guided Maintenance Algorithm

> Prerequisite: [Collection Loader Actor](./collection-loader-actor.md)

### Core Idea

Not all regions of the embedding space are equally sensitive to RCM ordering staleness. Use local curvature estimation to decide:
1. **Where to insert** new nodes in the RCM ordering (approximate vs precise)
2. **Where staleness hurts** (priority rebalancing budget allocation)
3. **When to trigger compaction** (per-segment staleness threshold)

### Source Pattern Transfer

**From Levada et al. (arXiv:2409.05084):**

The paper's core mechanic: neighborhood size adapts based on local curvature estimated via the shape operator. Points with low curvature get larger neighborhoods (tangent space approximates well); high curvature gets smaller neighborhoods.

**Extracted primitive**: The curvature-adaptive decision function -- using local geometry to modulate algorithmic effort.

| Paper Primitive | DFRR Transfer | What It Buys |
|---|---|---|
| Local curvature estimation via shape operator | Estimate insertion sensitivity at a graph region | Adaptive insertion cost: cheap inserts in flat regions, careful inserts in curved regions |
| Adaptive k -- small k in high-curvature, large k in low-curvature | Adaptive RCM window -- small rebalance radius in stable regions, larger radius in sensitive regions | Bounded work per insert: O(k) local reorder instead of O(n) global RCM |
| Tangent-space approximation quality as the decision criterion | RCM bandwidth preservation as the decision criterion | Quantified staleness: measurable locality degradation per insert |

### The Shape Operator Analogue for RCM

The curvature estimation from Levada maps to variance of neighbor positions in the RCM ordering:

```
curvature(N) = var(rcm_pos[n] for n in neighbors(N))
```

- Low variance: neighbors clustered in ordering ("flat"), tolerant of stale ordering
- High variance: neighbors spread in ordering ("curved"), sensitive to staleness

### The Insertion Algorithm

```
New vector V arrives -> HNSW graph adds V as node N (standard)

1. Find N's neighbors in the RCM ordering: {n1, n2, ..., nk}
2. Compute insertion position: median of neighbor positions in ordering
3. Estimate local curvature of the ordering:
   - Low: neighbors are clustered (small variance in RCM positions)
     -> Insert at median, done. O(k)
   - High: neighbors are spread (large variance in RCM positions)
     -> Local RCM rebalance within window [min_pos - w, max_pos + w]
     -> O(w log w) but w is bounded by curvature
4. Update affected vector arrays (f32, binary) only within the window
5. Track cumulative staleness metric
```

### The Scheduling Algorithm (Priority Rebalancer)

```
On each incremental insert of vector V:
  1. Add V to HNSW graph (standard, already works)
  2. Place V at approximate RCM position (median of neighbors)
  3. Compute curvature_score(V) = var(rcm_pos of V's neighbors)
  4. For each neighbor N of V:
     recompute curvature_score(N)
     if curvature_score(N) > threshold:
       enqueue N into rebalance_priority_queue

Background rebalancer (actor-driven, bounded work per cycle):
  while budget_remaining > 0:
    N = priority_queue.pop_max()    // most curved node
    window = neighbors_of(N) within RCM range
    local_rcm_reorder(window)       // O(w log w)
    budget_remaining -= w
    update curvature_scores for affected nodes
```

### Free Telemetry During Search

Curvature measured during normal DFRR search traversal (zero extra I/O):

```rust
// As traversal visits nodes, track variance of neighbor RCM positions
let variance = statistical_variance(neighbor_rcm_positions);
self.curvature_ema[node].observe(variance);
// Same EMA infrastructure as BQ1 calibration (calibration_persist.rs)
```

### Two-Lane Strategy with Curvature Maintenance

```
Lane 1 (instant): Brute-force / BM25 over growing segment
  - New vectors since last DFRR compaction
  - Exact results, no approximation
  - Searched linearly (small buffer)

Lane 2 (promoted): Full DFRR over sealed segment
  - Curvature-guided incremental inserts maintain ~95% cache locality
  - Periodic segment compaction restores 100% when staleness budget exceeded
  - Actor manages promotion: growing -> build DFRR -> seal -> swap

Staleness-triggered prewarm:
  - On each search: check staleness counter
  - If above threshold: actor triggers background Reload (compaction)
  - Old DFRR serves until new one is ready
```

---

## Empirical Validation Required

| Assumption | Testable? | How |
|---|---|---|
| RCM position variance correlates with actual cache miss rate | Yes | `perf stat -e cache-misses` on search, stratified by node curvature |
| Most code embeddings cluster in low-curvature regions | Yes | Histogram of per-node curvature on 380K-vector index |
| Local RCM reorder preserves global ordering quality | Yes | Before/after bandwidth measurement on reordered window |
| Curvature-weighted budget outperforms uniform budget | Yes | A/B recall comparison: random vs curvature-prioritized rebalancing |

## What Exists in the Codebase vs What's New

| Piece | Exists? | Where |
|---|---|---|
| HNSW incremental insert | Yes | `vector/src/lib.rs` -- VectorIndex::insert() |
| RCM reordering | Yes | `dfrr-kernel-rs/src/kernel.rs:433-444` |
| Staleness detection (node_count mismatch) | Yes | `kernel.rs:183-186` -- binary stale/fresh check |
| Actor with atomic swap | Yes | `vectordb_local.rs` -- CollectionLoaderActor |
| EMA infrastructure | Yes | `calibration_persist.rs` -- BQ1 EMA |
| Per-segment staleness tracking | No | New: staleness counter per RCM window |
| Local curvature estimation | No | New: variance of neighbor RCM positions |
| Adaptive rebalance window | No | New: the core algorithm |
| Segment-level compaction | No | Actor reload mechanism is the trigger |
| Two-lane growing + sealed segments | No | New: brute-force buffer + sealed DFRR |

## References

- Levada, Nielsen, Haddad. "Adaptive k-nearest neighbor classifier based on the local estimation of the shape operator." arXiv:2409.05084
- Xu, Liang, et al. "SPFresh: Incremental In-Place Update for Billion-Scale Vector Search." SOSP '23. arXiv:2410.14452
