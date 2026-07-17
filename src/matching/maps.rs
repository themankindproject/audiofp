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
