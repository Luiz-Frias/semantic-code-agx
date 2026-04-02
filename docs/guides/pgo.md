# Profile-Guided Optimization (PGO)

PGO uses real execution profiles to guide LLVM's optimizer, yielding 10–20%
throughput improvement on hot paths like HNSW graph traversal, distance
computation, and embedding batch I/O.

## Quick Start

```bash
just build-pgo
```

The optimized binary lands at `target/release-pgo-use/sca`.

## How It Works

The PGO pipeline runs three phases automatically:

1. **Instrument** — Build with `-Cprofile-generate`. The binary records which
   code paths execute and how often.
2. **Train** — Run a representative workload (index, search, reindex) against
   a cloned FastAPI corpus. This generates `.profraw` files.
3. **Optimize** — Merge profiles with `llvm-profdata`, then rebuild with
   `-Cprofile-use`. LLVM inlines hot functions, optimizes branch layout, and
   deprioritizes cold paths.

### Cargo Profiles

| Profile | Purpose | LTO | Strip |
|---|---|---|---|
| `release-pgo-generate` | Instrumented build | thin | no (LLVM needs symbols) |
| `release-pgo-use` | Optimized build | fat (cross-crate inlining) | yes |

The standard `release` profile is **never modified** by PGO.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SKIP_PGO` | `false` | Set `true` to fall back to `cargo build --release` |
| `PGO_SKIP_TRAINING` | `false` | Set `true` to reuse existing `merged.profdata` |
| `SCA_EMBEDDING_PROVIDER` | `onnx` | Override embedding provider for training |
| `PGO_CORPUS_DIR` | `tmp/pgo-corpus` | Directory for training corpus |

## CI Integration

In CI where PGO training is impractical (no ONNX models, no corpus), use the
escape hatch:

```bash
SKIP_PGO=true just build-pgo
```

This falls back to a standard `cargo build --release`.

For CI pipelines that cache the merged profile data:

```bash
# First run: full PGO pipeline
just build-pgo

# Subsequent runs: reuse cached .profdata
PGO_SKIP_TRAINING=true just build-pgo
```

## DFRR Kernel

The DFRR kernel (`dfrr-kernel-rs`) is consumed as a library via `[patch]`.
When SCA builds with PGO instrumentation, DFRR code is profiled automatically
— no separate PGO script is needed for DFRR.

## Troubleshooting

### sccache conflicts

PGO is incompatible with sccache. The build script automatically runs
`unset RUSTC_WRAPPER` before invoking cargo. If you see cache-related errors,
verify that sccache is not being forced by another mechanism.

### Missing llvm-profdata

The script searches three locations in order:

1. Rustup sysroot (`llvm-tools` component)
2. Homebrew (`/opt/homebrew/opt/llvm/bin/llvm-profdata`)
3. System PATH

Install via: `rustup component add llvm-tools`

### ONNX model not found

The training workload requires an embedding model. By default it uses ONNX
(local, no API key needed). If ONNX models aren't cached, the training script
will warn but continue — profile data will be limited to non-embedding paths.

To use a cloud provider instead:

```bash
SCA_EMBEDDING_PROVIDER=openai just build-pgo
```

### Why not BOLT?

BOLT (Binary Optimization and Layout Tool) is not viable on macOS because it
requires ELF binaries, while macOS uses Mach-O. BOLT support is deferred to
future Linux CI integration.
