# Collection Loader Actor — Design & Rationale

> Status: Implemented (commits b62b6f0..00708bd on fix/spawn-blocking-index)
> Date: 2026-03-21
> Context: Three-layer race condition fix + lifecycle management

## Problem: Three Independent Races at Three Layers

| Layer | Race | Root Cause | Fix |
|---|---|---|---|
| Physical | macOS kqueue edge-triggered notification lost between `spawn_blocking` completion and runtime park | `current_thread` runtime has a single wake path; if the notification fires while the runtime is between two polls, the edge-triggered event is consumed/lost | `multi_thread(1)` — same single-worker scheduling but thread-safe unpark signal survives coalescing |
| Logical | TOCTOU in check-then-load allowing duplicate 800MB+ loads | `ensure_loaded` checked the map, found it empty, then raced with concurrent callers through the load path | `OnceCell<()>` per-collection gates → replaced by actor command serialization |
| Lifecycle | No reload, no hot-swap, no failure recovery after drop | `OnceCell`/`OnceLock` are set-once-forever; `drop_collection` left stale gates | Actor owns the lifecycle: Load (exactly-once), Evict (cleanup), Reload (hot-swap) |

## Architecture: Actor as Sole Initialization Path

```
ensure_loaded(name)
  ├─ Fast path: collections.contains_key(name) → return Ok(()) [zero overhead]
  └─ Actor path: loader_handle.load(name) → mpsc → actor
       └─ handle_load:
            loader.load_collection(name)    ← pure I/O (try_join! parallel)
            kernel.set_snapshot_dir(...)    ← DFRR persistence
            replay_wal(...)                ← correctness
            collections.write().insert()   ← atomic commit
```

### Key Design Decisions

1. **Actor manages lifecycle, not individual operations.** The hot read/write path (search, insert) uses `Arc<RwLock<HashMap>>` directly. The actor only controls WHEN the map is populated or cleared.

2. **Bounded channel (capacity 8).** Per module 40 sizing: 2× expected concurrent requests × burst headroom. CLI loads 1 collection; headroom for agent mode.

3. **No dual-path.** The actor IS the initialization mechanism. No `Option<Handle>`, no OnceCell fallback. Simplifies reasoning about state.

4. **`CollectionLoaderContext` separates "how" from "when."** Pure loading logic (disk I/O, format detection, try_join! parallelism) lives in `CollectionLoaderContext`. The actor decides when to invoke it.

5. **`CollectionLoadResult` carries side-effect flags.** Loading is pure; side-effects (v2 rewrite, kernel snapshot_dir, WAL replay) happen in the caller (actor's `handle_load`).

### Lifecycle Capabilities

| Operation | Mechanism | Correctness Guarantee |
|---|---|---|
| Load | Actor `handle_load`: fast-path check → loader.load_collection → WAL replay → atomic insert | Exactly-once per collection (sequential command processing) |
| Evict | Actor `handle_evict`: map.remove(name) | Immediate; old value dropped when last Arc ref dies |
| Reload | (Reserved) Load fresh → atomic swap via map.insert(). Old value stays live via Arc refcount. | No downtime gap — old data serves until swap completes |

### Failure Modes

| Failure | Behavior | Recovery |
|---|---|---|
| Load I/O error | Actor sends Err(ErrorEnvelope) back via oneshot | Caller retries (sends another Load command) |
| Actor task panic | Channel stays open but rx.recv() never returns | 5-minute timeout → `loader_timeout` error |
| Runtime drop | Channel closes → actor exits cleanly | Natural shutdown |

## Rule Compliance

| Rule | How Satisfied |
|---|---|
| 06_agent §1 (message-passing for stateful components) | Actor + mpsc channel |
| 06_agent §3 (bounded steps) | 5-min timeout on reply channel |
| 40_async §15 (bounded channels) | Capacity 8, documented rationale |
| 40_async §17 (manage task lifecycles) | JoinHandle dropped in constructor (actor exits on channel close) |
| 40_async §21 (cancellation first-class) | CancellationToken in select! loop |
| 40_async §38 (cancellation-safe) | Load completes fully before atomic map insert — no partial state |
| 30_error (typed domain errors) | ErrorEnvelope with codes: loader_channel_closed, loader_reply_dropped, loader_timeout |

## Files Modified

| File | Change |
|---|---|
| `crates/adapters/src/vectordb_local.rs` | CollectionLoaderContext, CollectionLoadResult, CollectionLoaderCommand, CollectionLoaderHandle, CollectionLoaderActor, rewired ensure_loaded/drop_collection |
| `crates/infra/src/vectordb_factory.rs` | Pass CancellationToken to LocalVectorDb::new() |
| `crates/infra/src/cli_local.rs` | multi_thread(1) for all CLI paths |

## Commit History

1. `b62b6f0` fix(infra): unify CLI runtime to multi_thread(1)
2. `b265d81` fix(adapters): discriminate panic vs cancellation in spawn_blocking JoinError
3. `1ee7d31` perf(adapters): parallel index + sidecar loading via try_join!
4. `814838c` fix(adapters): clear init gate on drop_collection
5. `50eaaa0` refactor(adapters): extract CollectionLoaderContext
6. `89a499d` refactor(adapters): define CollectionLoaderCommand + Handle
7. `7245b6c` refactor(adapters): implement CollectionLoaderActor run loop
8. `8015b17` refactor(adapters): wire actor into LocalVectorDb (dual-path)
9. `00708bd` refactor(adapters): make loader actor sole initialization path
