//! Dielectric / optical response kernels (Kubo–Greenwood + JDOS).
//!
//! Pure-Rust ports of the vectorised NumPy inner loops in
//! `PAOFLOW.response.do_epsilon`. They operate on the rank-local k-slice and
//! return partial sums; MPI reduction and final scaling stay in Python.

pub mod eps;
pub mod jdos;
