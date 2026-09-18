//! Rust backend for PAOFLOW.
//!
//! Acceleration kernels grouped by domain. [`acbn0`] holds the ACBN0 / eACBN0
//! two-electron Coulomb integrals; [`epsilon`] holds the dielectric / JDOS
//! response kernels. The PyO3 bindings expose batched entry points. The
//! compiled extension is importable as `paoflow_rs`.

pub mod acbn0;
mod bindings;
pub mod epsilon;
