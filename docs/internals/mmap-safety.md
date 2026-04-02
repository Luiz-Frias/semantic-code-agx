# mmap Safety Invariants

This document defines the safety contract for read-only memory mapping in
`crates/vector/src/mmap.rs`.

## Scope

- Module: `semantic_code_vector::mmap`
- Public wrapper: `MmapBytes`
- Unsafe boundary: single call to `memmap2::MmapOptions::map`

## Invariants

1. Files are opened read-only through `open_readonly`.
2. Expected file length is validated from metadata before mapping.
3. File length is rechecked after mapping (`MappedLengthMismatch` guard).
4. Mapped bytes are exposed only as immutable slices (`&[u8]`).
5. Slice access uses checked arithmetic (`slice_at(offset, len)`).
6. Mapping lifetime is RAII-managed by `memmap2::Mmap` drop.
7. No mmap-related unsafe code exists outside `mmap.rs`.

## Failure model

All failures use typed errors (`MmapError`):

- open/metadata/map IO failures
- file length mismatch/overflow
- mapped length drift
- out-of-bounds or overflowed slice ranges

This keeps mmap behavior deterministic and audit-friendly as snapshot v2
binary loading is integrated.
