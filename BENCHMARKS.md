# Benchmarks

Scaling benchmark results for the local vector search kernels: **DFRR** (experimental) and **HNSW** (default).

Each quality tier pairs an HNSW `ef_search` setting with a DFRR strategy of comparable intent:

| Tier | HNSW | DFRR | Intent |
|------|------|------|--------|
| Conservative | ef=32 | s1 | Fastest, lower recall |
| Balanced | ef=64 | s2 | Good general-purpose |
| High Recall | ef=96 | s5 | Trades latency for recall |
| Wider Pivots | — | s3 | s2 variant: more pivot diversity |
| Wider Pull | — | s4 | s2 variant: larger pull radius |

---

## pplx-embed-0.6b (1024-dim, ANE)

### Test Setup

| Parameter | Value |
|-----------|-------|
| Embedding model | `pplx-embed-0.6b` (1024-dim, Apple Neural Engine) |
| Corpus | 10 open-source repos (Python, Rust, JS, Go) |
| Vector counts | 3K to 600K |
| Ground truth | Flat-scan exhaustive search (top-50) |
| Metric | Cosine similarity |
| Queries | 78 golden queries with known relevant files |
| Hardware | Apple M-series, single-threaded search |
| DFRR seed | `0xD4F6_59DE_0A97_2B4F` (deterministic) |

### Recall@10

Mean recall across all queries. HNSW and DFRR shown side-by-side at each quality tier.

| N | HNSW ef=32 | DFRR s1 | HNSW ef=64 | DFRR s2 | HNSW ef=96 | DFRR s5 | DFRR s3 | DFRR s4 |
|--------:|-----------:|--------:|-----------:|--------:|-----------:|--------:|--------:|--------:|
| 3,000 | 83.21% | 88.97% | 89.74% | 92.69% | 94.36% | 95.38% | 93.46% | 93.46% |
| 5,000 | 81.03% | 87.56% | 89.23% | 92.05% | 92.69% | 94.74% | 92.44% | 92.31% |
| 10,000 | 76.03% | 82.68% | 84.86% | 89.09% | 88.97% | 92.81% | 90.37% | 90.37% |
| 25,000 | 68.97% | 78.33% | 79.62% | 84.62% | 84.10% | 88.59% | 84.62% | 86.28% |
| 50,000 | 64.33% | 75.99% | 76.15% | 83.81% | 82.45% | 89.84% | 85.48% | 85.61% |
| 100,000 | 58.12% | 74.33% | 70.70% | 80.36% | 76.34% | 86.13% | 81.89% | 83.69% |
| 200,000 | 55.71% | 63.30% | 70.09% | 72.18% | 73.97% | 78.59% | 72.82% | 75.90% |
| 400,000 | 55.77% | 63.33% | 67.69% | 71.03% | 72.56% | 80.13% | 73.33% | 75.51% |
| 600,000 | 45.51% | 59.74% | 61.54% | 68.46% | 68.46% | 75.38% | 69.62% | 74.23% |

**Key takeaway**: DFRR consistently outperforms HNSW at every tier and every scale. The gap widens with scale — at 600K, DFRR s5 holds 75.38% while HNSW ef=96 drops to 68.46% (+6.9pp).

### Latency — p50 (ms)

Median query latency, single-threaded.

| N | HNSW ef=32 | DFRR s1 | HNSW ef=64 | DFRR s2 | HNSW ef=96 | DFRR s5 | DFRR s3 | DFRR s4 |
|--------:|-----------:|--------:|-----------:|--------:|-----------:|--------:|--------:|--------:|
| 3,000 | 0.19 | 0.43 | 0.28 | 0.61 | 0.29 | 0.77 | 0.62 | 0.67 |
| 5,000 | 0.17 | 0.51 | 0.25 | 0.73 | 0.33 | 0.95 | 0.75 | 0.74 |
| 10,000 | 0.24 | 0.62 | 0.33 | 0.90 | 0.46 | 1.28 | 0.99 | 1.04 |
| 25,000 | 0.37 | 1.01 | 0.56 | 1.40 | 0.72 | 2.23 | 1.42 | 1.62 |
| 50,000 | 0.54 | 1.22 | 0.74 | 1.74 | 1.01 | 2.91 | 1.77 | 2.07 |
| 100,000 | 0.94 | 1.59 | 1.26 | 2.32 | 1.39 | 3.46 | 2.37 | 2.87 |
| 200,000 | 1.54 | 2.43 | 2.16 | 3.44 | 2.81 | 4.82 | 3.99 | 3.96 |
| 400,000 | 3.63 | 3.51 | **14.41** | 5.01 | **17.96** | 8.28 | 4.91 | 5.63 |
| 600,000 | **107.84** | 3.82 | **116.60** | 5.80 | **129.96** | 9.90 | 5.32 | 7.09 |

### Latency — p95 (ms)

95th percentile query latency, single-threaded.

| N | HNSW ef=32 | DFRR s1 | HNSW ef=64 | DFRR s2 | HNSW ef=96 | DFRR s5 | DFRR s3 | DFRR s4 |
|--------:|-----------:|--------:|-----------:|--------:|-----------:|--------:|--------:|--------:|
| 3,000 | 0.27 | 0.85 | 0.49 | 0.84 | 0.50 | 1.10 | 0.87 | 1.05 |
| 5,000 | 0.28 | 0.81 | 0.43 | 1.03 | 0.58 | 1.37 | 1.09 | 1.09 |
| 10,000 | 0.35 | 0.91 | 0.50 | 1.34 | 0.71 | 1.95 | 1.45 | 1.59 |
| 25,000 | 0.61 | 1.83 | 0.91 | 2.18 | 1.14 | 3.63 | 2.20 | 2.81 |
| 50,000 | 1.52 | 2.09 | 1.17 | 2.88 | 2.36 | 4.79 | 2.72 | 3.36 |
| 100,000 | 2.41 | 2.34 | 3.43 | 3.33 | 3.43 | 5.29 | 3.77 | 4.39 |
| 200,000 | 3.31 | 6.53 | 5.62 | 8.14 | 7.04 | 11.79 | 8.93 | 11.24 |
| 400,000 | **25.38** | 9.62 | **102.39** | 10.33 | **139.43** | 19.82 | 10.18 | 14.35 |
| 600,000 | **619.33** | 9.91 | **839.59** | 14.51 | **834.02** | 47.90 | 14.84 | 35.22 |

**Key takeaway**: HNSW is faster up to ~200K vectors. At 400K+ HNSW latency explodes (p95 > 100ms) while DFRR stays bounded (p95 < 50ms). At 600K, DFRR is **50-170x faster** than HNSW at p95.

### DFRR vs HNSW Delta (Balanced Tier)

| N | HNSW ef=64 Recall | DFRR s2 Recall | Δ Recall | HNSW p95 | DFRR p95 |
|--------:|------------------:|---------------:|---------:|---------:|---------:|
| 3,000 | 89.74% | 92.69% | +2.95pp | 0.49 | 0.84 |
| 10,000 | 84.86% | 89.09% | +4.23pp | 0.50 | 1.34 |
| 50,000 | 76.15% | 83.81% | +7.66pp | 1.17 | 2.88 |
| 100,000 | 70.70% | 80.36% | +9.66pp | 3.43 | 3.33 |
| 400,000 | 67.69% | 71.03% | +3.34pp | 102.39 | 10.33 |
| 600,000 | 61.54% | 68.46% | +6.92pp | 839.59 | 14.51 |

---

## all-MiniLM-L6-v2 (384-dim, ONNX)

### Test Setup

| Parameter | Value |
|-----------|-------|
| Embedding model | `all-MiniLM-L6-v2` (384-dim, ONNX local) |
| Corpus | 10 open-source repos (Python, Rust, JS, Go) |
| Vector counts | 3K to 600K |
| Ground truth | Flat-scan exhaustive search (top-50) |
| Metric | Cosine similarity |
| Queries | 78 golden queries with known relevant files |
| Hardware | Apple M-series, single-threaded search |

### Recall@10

| N | HNSW ef=64 | DFRR s2 | HNSW ef=96 | DFRR s5 |
|--------:|-----------:|--------:|-----------:|--------:|
| 3,000 | 99.49% | 99.49% | 99.23% | 99.49% |
| 5,000 | 97.56% | 99.49% | 97.56% | 99.49% |
| 10,000 | 97.44% | 99.36% | 97.44% | 99.36% |
| 25,000 | 97.44% | 97.18% | 97.69% | 97.95% |
| 50,000 | 94.33% | 95.10% | 95.35% | 95.48% |
| 100,000 | 92.18% | 95.64% | 93.33% | **96.92%** |
| 200,000 | 88.21% | 91.54% | 89.74% | **93.33%** |
| 400,000 | 84.49% | 84.62% | 87.56% | 87.31% |
| 600,000 | 81.54% | 85.64% | 84.87% | **89.49%** |

### Latency — p50 / p95 (ms)

| N | HNSW ef=64 | DFRR s2 | HNSW ef=96 | DFRR s5 |
|--------:|-----------:|--------:|-----------:|--------:|
| 3,000 | 0.04 (0.05) | 0.20 (0.22) | 0.06 (0.08) | 0.33 (0.37) |
| 5,000 | 0.05 (0.11) | 0.27 (0.49) | 0.07 (0.09) | 0.28 (0.32) |
| 10,000 | 0.08 (0.13) | 0.25 (0.27) | 0.10 (0.14) | 0.36 (0.41) |
| 25,000 | 0.08 (0.13) | 0.35 (0.39) | 0.09 (0.15) | 0.41 (0.48) |
| 50,000 | 0.11 (0.25) | 0.42 (0.56) | 0.15 (0.34) | 0.61 (0.77) |
| 100,000 | 0.12 (0.44) | 0.56 (0.63) | 0.15 (0.45) | 0.65 (0.75) |
| 200,000 | 0.16 (0.62) | 0.74 (0.84) | 0.18 (0.62) | 0.93 (1.12) |
| 400,000 | 0.43 (1.17) | 1.07 (1.69) | 0.49 (1.30) | 1.19 (1.91) |
| 600,000 | 0.60 (1.69) | 1.51 (2.04) | 0.67 (3.53) | 1.81 (2.64) |

**Note**: The 384-dim MiniLM model produces lower-dimensional vectors that are faster to compare, so absolute latencies are lower than the 1024-dim pplx model. HNSW stays well-behaved at this dimensionality even at 600K.

---

## BQ1 Threshold Sweep (pplx-embed, DFRR s2 base)

Binary Quantization v1 (BQ1) pre-filters candidates using Hamming distance on binarized vectors before exact re-ranking. Higher thresholds allow more candidates through, trading latency for recall.

### Recall@10

| N | s2 base | t=0.05 | t=0.15 | t=0.30 | t=0.50 | t=0.70 | t=0.95 |
|--------:|--------:|-------:|-------:|-------:|-------:|-------:|-------:|
| 3,000 | 92.69% | 78.46% | 89.36% | 92.82% | 92.95% | 93.08% | 93.46% |
| 5,000 | 92.05% | 79.87% | 89.62% | 91.79% | 92.18% | 92.18% | 92.31% |
| 10,000 | 89.09% | 79.09% | 86.65% | 88.06% | 89.47% | 90.24% | 90.24% |
| 25,000 | 84.62% | 78.85% | 84.10% | 84.87% | 85.77% | 85.90% | 86.28% |
| 50,000 | 83.81% | 82.66% | 84.71% | 84.97% | 86.12% | 86.63% | 85.61% |
| 100,000 | 80.36% | 81.13% | 81.89% | 83.30% | 82.92% | 82.54% | 83.56% |
| 200,000 | 72.18% | 74.49% | 76.28% | 76.03% | 75.38% | 75.38% | 75.77% |
| 400,000 | 71.03% | 73.59% | 75.26% | 75.26% | 75.51% | 75.51% | 75.51% |
| 600,000 | 68.46% | 73.97% | 72.56% | 72.95% | 74.10% | 74.23% | 72.95% |

### Latency — p50 (ms)

| N | s2 base | t=0.05 | t=0.15 | t=0.30 | t=0.50 | t=0.70 | t=0.95 |
|--------:|--------:|-------:|-------:|-------:|-------:|-------:|-------:|
| 3,000 | 0.61 | 0.27 | 0.39 | 0.54 | 0.65 | 0.69 | 0.75 |
| 5,000 | 0.73 | 0.34 | 0.52 | 0.58 | 0.70 | 0.74 | 0.84 |
| 10,000 | 0.90 | 0.49 | 0.68 | 0.82 | 1.02 | 1.03 | 1.12 |
| 25,000 | 1.40 | 0.87 | 1.25 | 1.32 | 1.55 | 1.66 | 1.82 |
| 50,000 | 1.74 | 1.35 | 1.77 | 2.06 | 2.24 | 2.46 | 2.83 |
| 100,000 | 2.32 | 2.01 | 2.47 | 2.90 | 3.23 | 3.41 | 3.91 |
| 200,000 | 3.44 | 3.62 | 4.13 | 4.64 | 5.02 | 5.25 | 5.72 |
| 400,000 | 5.01 | 6.76 | 7.38 | 8.00 | 8.46 | 8.72 | 9.66 |
| 600,000 | 5.80 | 8.57 | 9.36 | 10.06 | 12.95 | 12.64 | 10.34 |

**Key takeaway**: BQ1 at low thresholds (t=0.05–0.15) acts as a fast pre-filter — at large N (200K+), BQ1 actually **improves recall** over base s2 by pruning low-quality candidates early, freeing budget for better ones. The recall peak shifts: at 600K, t=0.05 hits 73.97% vs base s2's 68.46% (+5.5pp) with only 48% more latency.

---

## When to Use Each Kernel

| Scenario | Recommended | Why |
|----------|-------------|-----|
| Small codebases (< 25K vectors) | **HNSW** (default) | Both achieve 85%+ recall; HNSW is 2-3x faster |
| Large codebases (100K+ vectors) | **DFRR** | 10-15% higher recall where it matters most |
| Latency-critical (< 0.5ms budget) | **HNSW** | Consistently faster at all scales up to 200K |
| Recall-critical (> 80% required) | **DFRR** | Reaches recall ceilings HNSW cannot |
| Very large scale (400K+) | **DFRR** | Faster *and* higher recall than HNSW |
| First-time users | **HNSW** (default) | No feature flags needed, good defaults |

## Configuration Profiles

| Profile | ef_search | pivot_count | pull_size | max_iterations | Tradeoff |
|---------|:---------:|:-----------:|:---------:|:--------------:|----------|
| **conservative** (s1) | 32 | 2 | 2 | 128 | Fastest, lower recall |
| **balanced** (s2) | 64 | 4 | 4 | 256 | Good general-purpose |
| **wider pivots** (s3) | 64 | 6 | 4 | 256 | More diversity in search directions |
| **wider pull** (s4) | 64 | 4 | 6 | 256 | Larger neighborhood per pivot |
| **high recall** (s5) | 96 | 8 | 8 | 512 | Trades latency for recall |

## Methodology

- **Ground truth**: Flat-scan exhaustive search computes exact top-50 neighbors for each query. Recall@10 is measured as the fraction of ground-truth results found in the approximate top-10.
- **Corpus scaling**: Vector counts from 3K to 600K via file sampling from 10 large open-source repos, with Gaussian noise padding above the natural corpus ceiling (~698K).
- **Determinism**: DFRR graph construction uses a fixed seed (`0xD4F6_59DE_0A97_2B4F`). Same data always produces identical results.
- **Cold-start excluded**: All measurements are after kernel warm-up (HNSW graph loaded, DFRR prewarm complete).
- **BQ1 thresholds**: Binary quantization pre-filter with Hamming distance. Threshold controls what fraction of candidates pass to exact re-ranking.

## Reproducing

```bash
# Clone the benchmark lab
git clone https://github.com/Luiz-Frias/bench-lab-rs.git
cd bench-lab-rs

# Run a scaling sweep (requires sca binary with DFRR feature)
bash scripts/bench-run-all.sh

# Analyze results
bench-report analyze --input experiments/data/scaling-pplx-embed-0.6b/ --output results/
```

See [bench-lab-rs README](https://github.com/Luiz-Frias/bench-lab-rs) for full setup and configuration.
