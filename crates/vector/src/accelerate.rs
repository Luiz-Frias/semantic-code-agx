//! Safe wrappers over Accelerate BLAS primitives with scalar fallbacks.
//!
//! On macOS with the `accelerate` feature enabled, distance computations use
//! `cblas_sdot` for dot products. On other platforms (or when the feature is
//! disabled), equivalent scalar Rust loops are used.
//!
//! The `accelerate` feature is **default-on**. Disable with
//! `--no-default-features` for A/B benchmarking against the scalar path.
#![cfg_attr(
    all(target_os = "macos", feature = "accelerate", not(miri)),
    expect(
        unsafe_code,
        reason = "Accelerate framework BLAS bindings require FFI calls with validated inputs"
    )
)]

#[cfg(all(target_os = "macos", feature = "accelerate", not(miri)))]
use libc::c_int;
use std::cell::Cell;

#[cfg(all(target_os = "macos", feature = "accelerate", not(miri)))]
#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" {
    fn cblas_sdot(n: c_int, x: *const f32, inc_x: c_int, y: *const f32, inc_y: c_int) -> f32;
}

/// Compute the dot product of two contiguous `f32` slices.
///
/// Uses `cblas_sdot` on macOS when the `accelerate` feature is enabled (default),
/// scalar fallback otherwise.
#[cfg(any(all(target_os = "macos", feature = "accelerate", not(miri)), test))]
#[inline]
pub fn dot_f32(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "dot_f32: length mismatch");
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }

    #[cfg(all(target_os = "macos", feature = "accelerate", not(miri)))]
    {
        if let Ok(n) = c_int::try_from(len) {
            // SAFETY:
            // - Both slices are valid, contiguous f32 data.
            // - `n` fits in c_int, stride=1, both pointers are in-bounds for `n` elements.
            // - Pointers originate from valid Rust slices and remain alive for this call.
            return unsafe { cblas_sdot(n, a.as_ptr(), 1, b.as_ptr(), 1) };
        }
    }

    dot_f32_scalar(a, b, len)
}

/// Compute the sum of squares of a contiguous `f32` slice.
///
/// Equivalent to `dot_f32(a, a)`.
#[cfg(any(all(target_os = "macos", feature = "accelerate", not(miri)), test))]
#[inline]
pub fn sum_squares_f32(a: &[f32]) -> f32 {
    dot_f32(a, a)
}

#[cfg(any(all(target_os = "macos", feature = "accelerate", not(miri)), test))]
#[inline]
fn dot_f32_scalar(a: &[f32], b: &[f32], len: usize) -> f32 {
    let mut acc = 0.0_f32;
    for i in 0..len {
        if let (Some(&av), Some(&bv)) = (a.get(i), b.get(i)) {
            acc += av * bv;
        }
    }
    acc
}

thread_local! {
    static DISTANCE_EVAL_TRACKING_DEPTH: Cell<u32> = const { Cell::new(0) };
    static DISTANCE_EVAL_COUNT: Cell<u64> = const { Cell::new(0) };
}

struct DistanceEvalTrackingGuard {
    start_count: u64,
}

impl DistanceEvalTrackingGuard {
    fn begin() -> Self {
        let start_count = DISTANCE_EVAL_COUNT.with(Cell::get);
        DISTANCE_EVAL_TRACKING_DEPTH.with(|depth| {
            depth.set(depth.get().saturating_add(1));
        });
        Self { start_count }
    }

    fn count(&self) -> u64 {
        DISTANCE_EVAL_COUNT.with(|count| count.get().saturating_sub(self.start_count))
    }
}

impl Drop for DistanceEvalTrackingGuard {
    fn drop(&mut self) {
        DISTANCE_EVAL_TRACKING_DEPTH.with(|depth| {
            depth.set(depth.get().saturating_sub(1));
        });
    }
}

#[inline]
fn record_distance_evaluation() {
    DISTANCE_EVAL_TRACKING_DEPTH.with(|depth| {
        if depth.get() == 0 {
            return;
        }
        DISTANCE_EVAL_COUNT.with(|count| {
            count.set(count.get().saturating_add(1));
        });
    });
}

pub fn with_distance_eval_tracking<T>(f: impl FnOnce() -> T) -> (T, u64) {
    let guard = DistanceEvalTrackingGuard::begin();
    let result = f();
    let count = guard.count();
    (result, count)
}

/// Accelerate-backed cosine distance metric for `hnsw_rs`.
///
/// Drop-in replacement for `hnsw_rs::prelude::DistCosine` that uses Apple
/// Accelerate `cblas_sdot` on macOS. Falls back to `DistCosine` on other
/// platforms or when the `accelerate` feature is disabled.
///
/// Returns cosine distance: `1.0 - cosine_similarity`, clamped to `[0.0, 1.0]`.
pub struct DistAccelerateCosine;

impl hnsw_rs::prelude::Distance<f32> for DistAccelerateCosine {
    fn eval(&self, va: &[f32], vb: &[f32]) -> f32 {
        record_distance_evaluation();

        #[cfg(all(target_os = "macos", feature = "accelerate", not(miri)))]
        {
            let dot = dot_f32(va, vb);
            let sq_a = sum_squares_f32(va);
            let sq_b = sum_squares_f32(vb);
            let denom = (sq_a * sq_b).sqrt();

            if denom <= 0.0 || !denom.is_finite() {
                return 1.0;
            }

            let cosine = (dot / denom).clamp(-1.0, 1.0);
            // `return` required: exits `#[cfg]` block, not the function.
            #[expect(
                clippy::needless_return,
                reason = "return exits cfg block, not function"
            )]
            return (1.0 - cosine).max(0.0);
        }

        // Fallback: delegate to the upstream DistCosine implementation.
        #[cfg(not(all(target_os = "macos", feature = "accelerate", not(miri))))]
        {
            hnsw_rs::prelude::DistCosine.eval(va, vb)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DistAccelerateCosine, dot_f32, sum_squares_f32, with_distance_eval_tracking};
    use hnsw_rs::prelude::Distance;

    #[test]
    fn dot_f32_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = dot_f32(&a, &b);
        assert!((result - 32.0).abs() < 1e-5);
    }

    #[test]
    fn dot_f32_empty() {
        assert_eq!(dot_f32(&[], &[]), 0.0);
    }

    #[test]
    fn sum_squares_basic() {
        let a = vec![3.0, 4.0];
        assert!((sum_squares_f32(&a) - 25.0).abs() < 1e-6);
    }

    #[test]
    fn dot_f32_dim384_matches_scalar() {
        let a: Vec<f32> = (0..384).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..384).map(|i| (383 - i) as f32 * 0.01).collect();
        let scalar: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let accel = dot_f32(&a, &b);
        assert!(
            (accel - scalar).abs() < 0.01,
            "scalar={scalar}, accel={accel}"
        );
    }

    #[test]
    fn accelerated_cosine_matches_dist_cosine() {
        let a: Vec<f32> = (0..384).map(|i| (i as f32) * 0.01).collect();
        let b: Vec<f32> = (0..384).map(|i| (383 - i) as f32 * 0.01).collect();

        let accel = DistAccelerateCosine.eval(&a, &b);
        let baseline = hnsw_rs::prelude::DistCosine.eval(&a, &b);

        assert!(
            (accel - baseline).abs() < 1e-4,
            "accelerated={accel}, baseline={baseline}"
        );
    }

    #[test]
    fn accelerated_cosine_identical_vectors() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let dist = DistAccelerateCosine.eval(&a, &a);
        assert!(
            dist.abs() < 1e-6,
            "identical vectors should have distance ~0, got {dist}"
        );
    }

    #[test]
    fn accelerated_cosine_zero_vector() {
        let a = vec![1.0, 2.0, 3.0];
        let zero = vec![0.0, 0.0, 0.0];
        let dist = DistAccelerateCosine.eval(&a, &zero);
        assert!(
            (dist - 1.0).abs() < 1e-6,
            "zero vector should yield distance 1.0, got {dist}"
        );
    }

    #[test]
    fn distance_eval_tracking_counts_cosine_calls() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let c = vec![7.0, 8.0, 9.0];

        let (_, count) = with_distance_eval_tracking(|| {
            let first = DistAccelerateCosine.eval(&a, &b);
            let second = DistAccelerateCosine.eval(&a, &c);
            first + second
        });

        assert_eq!(count, 2);
    }
}
