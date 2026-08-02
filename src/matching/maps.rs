//! Fast map aliases for matching hot paths.
//!
//! With the `std` feature (default), use [`std::collections::HashMap`] for
//! O(1) expected lookups. Without `std`, fall back to [`alloc::collections::BTreeMap`]
//! so the matching module stays `no_std + alloc` capable — same API surface
//! for `entry` / `get` / `retain` / `is_empty`.

#[cfg(feature = "std")]
pub(crate) use std::collections::HashMap;

#[cfg(not(feature = "std"))]
pub(crate) use alloc::collections::BTreeMap as HashMap;

/// Create a map with pre-allocated capacity (HashMap) or just empty (BTreeMap).
/// BTreeMap has no `with_capacity`, so we fall back to `new()`.
#[cfg(feature = "std")]
#[inline]
pub(crate) fn hashmap_with_capacity<K, V>(cap: usize) -> HashMap<K, V>
where
    K: core::hash::Hash + Eq,
{
    HashMap::with_capacity(cap)
}

#[cfg(not(feature = "std"))]
#[inline]
pub(crate) fn hashmap_with_capacity<K, V>(_cap: usize) -> HashMap<K, V>
where
    K: Ord,
{
    HashMap::new()
}
