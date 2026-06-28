//! Rust backend for the ACBN0 / eACBN0 two-electron Coulomb integrals.
//!
//! The numerical core lives in [`special`] (special functions) and `eri`
//! (the contracted Coulomb integral, added in phase 2). PyO3 bindings exposing
//! a batched `eri_batch` entry point.

mod bindings;
pub mod eri;
pub mod special;
