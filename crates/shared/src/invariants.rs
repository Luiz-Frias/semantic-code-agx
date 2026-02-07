//! Compile-time helpers for carrying validated invariants.

use std::fmt;

/// Marker type for unvalidated state.
#[derive(Debug, Clone, Copy, Default)]
pub struct Unvalidated;

/// Marker type for validated state.
#[derive(Debug, Clone, Copy, Default)]
pub struct ValidatedState;

/// Proof wrapper indicating a value has been validated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Validated<T>(T);

impl<T> Validated<T> {
    /// Wrap a validated value.
    pub const fn new(value: T) -> Self {
        Self(value)
    }

    /// Borrow the inner value.
    pub const fn as_ref(&self) -> &T {
        &self.0
    }

    /// Consume and return the inner value.
    pub fn into_inner(self) -> T {
        self.0
    }
}

impl<T> std::ops::Deref for Validated<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// Out-of-range error for bounded numeric wrappers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundsError<T> {
    /// Raw value provided.
    pub value: T,
    /// Inclusive minimum.
    pub min: T,
    /// Inclusive maximum.
    pub max: T,
}

impl<T: fmt::Display> fmt::Display for BoundsError<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "value {} is outside [{}, {}]",
            self.value, self.min, self.max
        )
    }
}

impl<T: fmt::Debug + fmt::Display> std::error::Error for BoundsError<T> {}

/// Bounded `u32` with const generic limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BoundedU32<const MIN: u32, const MAX: u32>(u32);

impl<const MIN: u32, const MAX: u32> BoundedU32<MIN, MAX> {
    /// Create a bounded value when within the inclusive range.
    pub const fn new(value: u32) -> Option<Self> {
        if value < MIN || value > MAX {
            None
        } else {
            Some(Self(value))
        }
    }

    /// Create a bounded value or return a bounds error.
    pub const fn try_new(value: u32) -> Result<Self, BoundsError<u32>> {
        match Self::new(value) {
            Some(value) => Ok(value),
            None => Err(BoundsError {
                value,
                min: MIN,
                max: MAX,
            }),
        }
    }

    /// Return the wrapped value.
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Bounded `usize` with const generic limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BoundedUsize<const MIN: usize, const MAX: usize>(usize);

impl<const MIN: usize, const MAX: usize> BoundedUsize<MIN, MAX> {
    /// Create a bounded value when within the inclusive range.
    pub const fn new(value: usize) -> Option<Self> {
        if value < MIN || value > MAX {
            None
        } else {
            Some(Self(value))
        }
    }

    /// Create a bounded value or return a bounds error.
    pub const fn try_new(value: usize) -> Result<Self, BoundsError<usize>> {
        match Self::new(value) {
            Some(value) => Ok(value),
            None => Err(BoundsError {
                value,
                min: MIN,
                max: MAX,
            }),
        }
    }

    /// Return the wrapped value.
    pub const fn get(self) -> usize {
        self.0
    }
}

/// Bounded `u64` with const generic limits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BoundedU64<const MIN: u64, const MAX: u64>(u64);

impl<const MIN: u64, const MAX: u64> BoundedU64<MIN, MAX> {
    /// Create a bounded value when within the inclusive range.
    pub const fn new(value: u64) -> Option<Self> {
        if value < MIN || value > MAX {
            None
        } else {
            Some(Self(value))
        }
    }

    /// Create a bounded value or return a bounds error.
    pub const fn try_new(value: u64) -> Result<Self, BoundsError<u64>> {
        match Self::new(value) {
            Some(value) => Ok(value),
            None => Err(BoundsError {
                value,
                min: MIN,
                max: MAX,
            }),
        }
    }

    /// Return the wrapped value.
    pub const fn get(self) -> u64 {
        self.0
    }
}
