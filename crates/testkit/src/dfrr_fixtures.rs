//! Deterministic fixtures and configuration helpers for DFRR tests.

/// Deterministic seed label for DFRR milestone tests.
pub const DFRR_FIXTURE_SEED_LABEL: &str = "dfrr-fixture-v1";

/// Default milestone-level test seed. Use this when a test does not need label-level fan-out.
pub const DFRR_FIXTURE_DEFAULT_SEED: u64 = 0xB57D_D0B0_DEAD_BEEF;

/// Default number of neighbors for shared DFRR fixtures.
pub const DFRR_FIXTURE_MAX_NEIGHBORS: u32 = 8;

/// Default construction beam width for shared DFRR fixtures.
pub const DFRR_FIXTURE_EF_CONSTRUCTION: u32 = 32;

/// Default search beam width for shared DFRR fixtures.
pub const DFRR_FIXTURE_EF_SEARCH: u32 = 12;

/// Default number of levels for shared DFRR fixtures.
pub const DFRR_FIXTURE_MAX_LEVELS: u8 = 4;

/// Shared 3D vector fixture for DFRR integration tests.
pub const DFRR_FIXTURE_VECTORS: [[f32; 3]; 6] = [
    [1.00, 0.00, 0.00],
    [0.95, 0.20, 0.10],
    [0.00, 1.00, 0.00],
    [0.10, 0.90, 0.20],
    [0.00, 0.00, 1.00],
    [0.20, 0.10, 0.95],
];

/// Borrow the shared fixture rows.
#[must_use]
pub const fn fixture_vectors() -> &'static [[f32; 3]] {
    &DFRR_FIXTURE_VECTORS
}

/// Deterministic seed function with no randomness used.
pub fn deterministic_seed(label: &str) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in label.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x0100_0000_01b3);
    }

    hash
}

#[cfg(test)]
mod tests {
    use super::{
        DFRR_FIXTURE_DEFAULT_SEED, DFRR_FIXTURE_EF_CONSTRUCTION, DFRR_FIXTURE_EF_SEARCH,
        DFRR_FIXTURE_MAX_LEVELS, DFRR_FIXTURE_MAX_NEIGHBORS, deterministic_seed, fixture_vectors,
    };

    #[test]
    fn fixture_vectors_are_stable_and_non_empty() {
        let vectors = fixture_vectors();
        assert_eq!(vectors.len(), 6);
        assert!(vectors.iter().all(|vector| vector.len() == 3));
        assert_eq!(vectors[0], [1.0, 0.0, 0.0]);
    }

    #[test]
    fn fixture_seed_is_deterministic_and_label_scoped() {
        let seed_first = deterministic_seed("dfrr-fixture-v1");
        let seed_second = deterministic_seed("dfrr-fixture-v1");
        let seed_other = deterministic_seed("dfrr-fixture-v2");

        assert_eq!(seed_first, seed_second);
        assert_ne!(seed_first, seed_other);
        assert_ne!(seed_first, DFRR_FIXTURE_DEFAULT_SEED);
        assert_eq!(DFRR_FIXTURE_MAX_NEIGHBORS, 8);
        assert_eq!(DFRR_FIXTURE_MAX_LEVELS, 4);
        assert_eq!(DFRR_FIXTURE_EF_SEARCH, 12);
        assert_eq!(DFRR_FIXTURE_EF_CONSTRUCTION, 32);
    }
}
