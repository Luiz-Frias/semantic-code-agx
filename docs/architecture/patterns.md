# Advanced Rust Pattern Opportunities — FSM, GAT, Typestate

> Analysis date: 2026-03-05
> Scope: Full 13-crate workspace scan against `.cursor/rules/rust/modules/**`
> Method: Double-blind parallel analysis (Explore agent + recursive-context-explorer)
> Purpose: Identify refactoring opportunities, not prescribe changes

---

## Already Excellent (Reference Patterns)

These are already well-implemented and can serve as internal references:

| Pattern | Location | Notes |
|---------|----------|-------|
| Index pipeline typestate | `app/src/index_codebase/mod.rs` | `IndexPipeline<Prepared/Scanned/Embedded/Inserted/Completed>` with `PhantomData<S>` |
| Build-time FSM generator | `app/build.rs` + `specs/index_pipeline.fsm` | Parses `.fsm` spec, generates `IndexPipelineState` enum at compile time |
| GAT-based lending shims | `ports/src/embedding.rs:211-290` | `EmbeddingPortLend: GAT<Future<'a, T>>` sealed trait shim over `BoxFuture` |
| Const-generic fixed-dim | `vector/src/lib.rs` | `VectorIndexFixed<const D: usize>`, `FixedVector<const D: usize>` |
| Newtype domain primitives | `domain/src/primitives.rs` | `CodebaseId`, `ChunkId`, `DocumentId`, `CollectionName` — all validated |
| Safe path newtype | `ports/src/filesystem.rs:8-26` | `SafeRelativePath` wrapping `Box<str>` with validation |
| `Arc<[f32]>` vectors | `ports/src/embedding.rs:9-55` | Shared embedding vectors avoid cloning |
| Sealed trait in lending ports | `ports/src/embedding.rs:211-251` | Private `sealed::Sealed` prevents external implementations |
| Error envelope architecture | `shared/src/` | `ErrorEnvelope` with `ErrorCode`, `ErrorClass`, structured metadata |

---

## Sprint 1 — Quick Wins (1-2 days)

### 1A. Remove Redundant Dual State in Index Pipeline

- **Location**: `app/src/index_codebase/types.rs` (lines 629-645) + `mod.rs` (lines 36-50)
- **Current**: Two parallel state representations exist for the same concept:
  1. Typestate `IndexPipeline<S>` with `PhantomData<S>` (compile-time — correct)
  2. `IndexPipelineFsm` inside it carrying `state: IndexPipelineState` runtime enum (redundant)
- **Problem**: Every transition (e.g. `scanned()`) must update BOTH the phantom type AND the runtime enum. They can silently drift.
- **Proposed**: Remove `IndexPipelineFsm` from `IndexPipeline<S>`. Derive the state label via a `trait PipelineStateLabel { fn label() -> &'static str; }` implemented per ZST. Keep runtime enum only for tracing/telemetry if needed.
- **Benefit**: Eliminates dual-representation drift risk. The typestate IS the state.
- **Complexity**: Low — internal to `index_codebase/`, no public API changes.
- **Rule ref**: Module 35 — "Typestate transitions consume `self` and return the next state type."

### 1B. Add `flush()` to `VectorDbPortLend` GAT Shim

- **Location**: `ports/src/vectordb.rs` — `VectorDbPortLend` (line 267)
- **Current**: `VectorDbPort` defines `flush()` with default no-op. The GAT lending shim `VectorDbPortLend` does NOT include `flush()`.
- **Proposed**: Add `flush()` to `VectorDbPortLend` with same default no-op. Blanket impl auto-covers.
- **Benefit**: API completeness. Prevents silent flush skips when using lending shim path.
- **Complexity**: Very low — additive only.
- **Rule ref**: Module 20 — "Public traits are small, cohesive, and object-safe when needed."

### 1C. Consolidate Index Pipeline Impl Blocks by State

- **Location**: `app/src/index_codebase/mod.rs` (lines 36-82)
- **Current**: Typestate exists but methods are scattered across modules. State-specific operations aren't grouped.
- **Proposed**: Group into per-state `impl IndexPipeline<Scanned>`, `impl IndexPipeline<Embedded>`, etc. Only expose batch scheduling on `Scanned`, commit on `Embedded`.
- **Benefit**: API is self-documenting; wrong-state method calls are compile errors.
- **Complexity**: Easy — organizational refactoring only.
- **Rule ref**: Module 35 §typestate-consolidation

---

## Sprint 2 — Structural Safety (3-5 days)

### 2A. `VectorIndex` Non-Empty Typestate Witness

- **Location**: `vector/src/lib.rs` — `VectorIndex` struct (line 585)
- **Current**: `VectorIndex::new()` creates empty index. `snapshot()` can be called on empty index — produces valid but semantically meaningless snapshot. `search()` on empty returns `Ok(vec![])` silently. DFRR kernel requires non-empty inputs but enforces only at runtime.
- **Proposed**: `NonEmpty` witness type. `insert()` returns `VectorIndex<Populated>`. `snapshot()` only available on `impl VectorIndex<Populated>`.
- **Benefit**: DFRR precondition becomes structural. No wasted I/O for empty snapshots.
- **Complexity**: Medium — `VectorKernel` trait and `LocalVectorDb` call sites need updating.
- **Rule ref**: Module 25 §phantom-data, Module 20 — "make invalid states unrepresentable"

### 2B. Snapshot Migration Explicit FSM

- **Location**: `adapters/src/vectordb_local.rs` — `load_via_v1_json()` (line 286), auto-migrate block (lines 297-319)
- **Current**: V1→V2 migration is an `if` block with best-effort non-fatal error handling. Three implicit states (`V1Only → MigrationAttempted → V2WithSidecar`) embedded in control flow.
- **Proposed**: Explicit `SnapshotMigrationState` enum:
  ```rust
  pub enum SnapshotMigrationState {
      V1Only { snapshot: CollectionSnapshot },
      V2WithSidecar { index: VectorIndex, sidecar_path: PathBuf },
      MigrationFailed { snapshot: CollectionSnapshot, error: Box<str> },
  }
  ```
- **Benefit**: Three migration outcomes are first-class variants, auditable in review.
- **Complexity**: Medium-low — private internals only.
- **Rule ref**: Module 35 — "Enum FSM with Transitions", Module 20 — "Prefer enums for sum types"

### 2C. Collection Load Path Typestate Session

- **Location**: `adapters/src/vectordb_local.rs` — `ensure_loaded()` (line 162)
- **Current**: Branches on `self.snapshot_format` at runtime. V2 path has fallback chain: try sidecar → fall back to V1 JSON → auto-migrate → reload. Four `Ok(None)` exit paths signal "caller should try other path."
- **Proposed**: `SnapshotLoadSession` typestate:
  ```
  NeedsLoad → V2Candidate → V2Loaded | FallbackToV1 → V1Loaded → MigrationDue | Ready
  ```
- **Benefit**: Eliminates implicit protocol between caller/callee. Load path auditable as single type-chain.
- **Complexity**: Medium — async path through `Arc<RwLock<...>>`.
- **Rule ref**: Module 35 — "Prevent invalid state transitions at compile time"

---

## Sprint 3 — Config & Architecture (1 week)

### 3A. Config Builder Typestate

- **Location**: `config/src/schema.rs` (lines 101-150), `config/src/load.rs` (line 50)
- **Current**: Config loading is a 4-layer merge (`default → file → overrides → env`) enforced only by private function call order. `ValidatedBackendConfig` exists but no compile-time guarantee an unvalidated `BackendConfig` can't be used in its place.
- **Proposed**: Typestate builder:
  ```rust
  ConfigBuilder<Defaults> → .with_file() → ConfigBuilder<FileApplied>
    → .with_overrides() → ConfigBuilder<OverridesApplied>
    → .validate(env) → ConfigBuilder<Validated>
    → .build() → ValidatedBackendConfig
  ```
- **Benefit**: Merge order is a compile-time guarantee. New entry points can't skip steps.
- **Complexity**: Medium — transitions must be fallible (`-> Result<ConfigBuilder<Next>>`).
- **Rule ref**: Module 35 §typestate-builders, Module 25 §phantom-data

### 3B. Bounded Numeric Types in Config Schemas

- **Location**: `shared/src/invariants.rs` → `config/src/schema.rs`
- **Current**: `BoundedU32`, `BoundedU64`, `BoundedUsize` exist but underutilized. Config fields use raw `u64`.
- **Proposed**: Replace `pub timeout_ms: u64` with `pub timeout_ms: BoundedU64<1_000, 600_000>`.
- **Benefit**: Compile-time bound verification; reduces config validation surface.
- **Complexity**: Easy — types exist; mechanical refactor.
- **Rule ref**: Module 20 §newtype-invariants

---

## Future / Research — Gate on Profiling Evidence

### 4A. Batch Lifecycle Session Types

- **Location**: `app/src/index_codebase/types.rs` — `BatchState` (line 463)
- **Current**: `InsertTask` uses `Option<BoxFuture>` to distinguish "queued but not awaitable" from "awaited." Manual sequence pointers (`next_batch_to_insert`, `next_insert_to_await`) maintain ordering invariant.
- **Proposed**: Session types: `InsertTask<Queued>` (no future field) → `InsertTask<InFlight>` (non-optional future).
- **Why deferred**: Deeply interleaved with async fan-out. `BoxFuture` lifetime constraints make typestate generics complex.
- **Rule ref**: Module 35 — "No `Option` for 'maybe present'", Module 95 — session types

### 4B. `SplitterPort` Interior Mutability Removal

- **Location**: `ports/src/splitter.rs` (line 39-41)
- **Current**: `set_chunk_size(&self)` and `set_chunk_overlap(&self)` take `&self` — requiring interior mutability in all impls. Implicit "call set_* before split()" protocol.
- **Proposed**: Move chunk size/overlap into `SplitOptions` (already exists at line 21) per-call. Or typestate `ConfiguredSplitter<HasChunkSize>`.
- **Why deferred**: Breaking change across all splitter implementations.
- **Rule ref**: Module 10 — "Avoid interior mutability abuse"

### 4C. Dense/Hybrid Collection Mode Typestate

- **Location**: `ports/src/vectordb.rs` (lines 199-213)
- **Current**: `insert()` vs `insert_hybrid()` — caller must know which to call based on runtime `IndexMode`.
- **Proposed**: `VectorDbCollectionPort<Dense>` / `VectorDbCollectionPort<Hybrid>` — mode as type parameter.
- **Why deferred**: Changes public port interface. All adapters (Milvus gRPC, REST, local) need both impls.
- **Rule ref**: Module 20 — "make invalid states unrepresentable"

### 4D. GAT Lending Iterator for `active_records()`

- **Location**: `vector/src/lib.rs` (line 639)
- **Current**: Returns `Vec<&VectorRecord>` — allocates on every call (hot path in DFRR graph build).
- **Proposed**: Custom skip-deleted iterator via GAT lending pattern. Zero allocation.
- **Why deferred**: Only material in sequential DFRR path; Rayon parallel path already uses `par_iter()`.
- **Rule ref**: Module 25 §gat-lending-iterators, Module 60

### 4E. CLI Arg Deduplication via `#[command(flatten)]`

- **Location**: `bins/cli/src/args.rs` — `Commands` enum (line 13)
- **Current**: Every command variant duplicates ~9 vector DB flags and ~8 embedding flags.
- **Proposed**: Shared `VectorDbArgs`/`EmbeddingArgs` structs with `#[command(flatten)]`.
- **Benefit**: ~200 lines of duplication collapse. New flags added in one place.
- **Complexity**: Low for logic, medium for clap integration.
- **Rule ref**: Module 20 — "Trait composition over inheritance-like patterns"

---

## Validated Non-Opportunities

Investigated and confirmed **correctly implemented** — no change needed:

- **Trait object dispatch for ports** (`Arc<dyn EmbeddingPort>`): Provider selection is config-time, not hot-path. Dynamic dispatch justified.
- **Sealed trait on lending ports**: Correctly implemented.
- **Newtype ID coverage**: Comprehensive via `validate-derive` proc macro.
- **`Arc<[f32]>` for embeddings**: Already optimized.
- **`ErrorEnvelope` builder**: Current constructor pattern (`expected/invariant/unexpected` + `with_metadata`) is adequate. Typestate builder would add complexity for minimal gain.

---

## Summary Matrix

| ID | Location | Anti-Pattern | Pattern | Impact | Risk | Sprint |
|----|----------|-------------|---------|--------|------|--------|
| 1A | `app/index_codebase/types.rs` | Dual state (typestate + runtime enum) | Remove redundant FSM; trait-derived label | High | Low | 1 |
| 1B | `ports/src/vectordb.rs` | `flush()` missing from GAT shim | Add to `VectorDbPortLend` | High | Very Low | 1 |
| 1C | `app/index_codebase/mod.rs` | Scattered impl blocks | Per-state `impl` consolidation | Medium | Low | 1 |
| 2A | `vector/src/lib.rs` | Empty index can snapshot | `NonEmpty` typestate witness | High | Medium | 2 |
| 2B | `adapters/vectordb_local.rs` | Implicit migration states | Explicit `SnapshotMigrationState` enum FSM | Medium | Medium-Low | 2 |
| 2C | `adapters/vectordb_local.rs` | 3-way fallback chain | `SnapshotLoadSession` typestate | Medium | Medium | 2 |
| 3A | `config/src/load.rs` | Merge order by convention | Typestate config builder | High | Medium | 3 |
| 3B | `config/src/schema.rs` | Raw `u64` config fields | `BoundedU64<MIN, MAX>` | Medium | Low | 3 |
| 4A | `app/index_codebase/types.rs` | `Option<BoxFuture>` + manual seq pointers | Session types for batch lifecycle | Medium | High | Future |
| 4B | `ports/src/splitter.rs` | `&self` mutators; implicit protocol | Move to per-call `SplitOptions` | Low | Medium | Future |
| 4C | `ports/src/vectordb.rs` | Runtime `insert` vs `insert_hybrid` | `VectorDbCollectionPort<Mode>` | Low | High | Future |
| 4D | `vector/src/lib.rs` | `Vec<&VectorRecord>` allocation | GAT lending iterator | Low | Low | Future |
| 4E | `bins/cli/src/args.rs` | Duplicated flag blocks | `#[command(flatten)]` shared structs | Low | Low | Future |
