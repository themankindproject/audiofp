//! The [`Hash32`] trait — a common interface for 32-bit fingerprint
//! hashes across all classical fingerprinters.
//!
//! Every classical fingerprinter ([`Wang`], [`Panako`], [`Haitsma`])
//! produces hashes that fit into a 32-bit word plus an anchor time.
//! `Hash32` abstracts over the concrete hash types so that generic
//! code — the [`Matcher`](crate::matcher::Matcher), persistence layers,
//! FFI wrappers — can work with any fingerprinter's output uniformly.
//!
//! [`Wang`]: crate::classical::Wang
//! [`Panako`]: crate::classical::Panako
//! [`Haitsma`]: crate::classical::Haitsma

use core::fmt::Debug;

/// A 32-bit fingerprint hash with an associated anchor time.
///
/// All three classical fingerprinters produce hashes that satisfy this
/// trait:
///
/// - [`WangHash`]: `hash` = packed `(f_a, f_b, Δt)`, `t_anchor` = anchor
///   frame index.
/// - [`PanakoHash`]: `hash` = packed `(sign, mag_order, β, Δf_ab, Δf_bc)`,
///   `t_anchor` = anchor frame index.
/// - [`HaitsmaHash`]: `hash` = the 32 sign-bits, `t_anchor` = frame
///   index (synthesised from the `Vec<u32>` position because Haitsma
///   does not store an anchor per hash).
///
/// [`WangHash`]: crate::classical::WangHash
/// [`PanakoHash`]: crate::classical::PanakoHash
/// [`HaitsmaHash`]: crate::classical::HaitsmaHash
pub trait Hash32: Copy + Clone + Debug + Send + Sync + 'static {
    /// The 32-bit hash value used for inverted-index lookup.
    fn hash(&self) -> u32;

    /// The anchor time in STFT-frame units.
    ///
    /// This is the temporal position of the hash within the source
    /// audio, expressed as a frame index. The matcher uses it to
    /// compute the time offset `Δt` between a query hash and a
    /// reference hash.
    fn t_anchor(&self) -> u32;
}

impl Hash32 for crate::classical::WangHash {
    #[inline]
    fn hash(&self) -> u32 {
        self.hash
    }
    #[inline]
    fn t_anchor(&self) -> u32 {
        self.t_anchor
    }
}

impl Hash32 for crate::classical::PanakoHash {
    #[inline]
    fn hash(&self) -> u32 {
        self.hash
    }
    #[inline]
    fn t_anchor(&self) -> u32 {
        self.t_anchor
    }
}
