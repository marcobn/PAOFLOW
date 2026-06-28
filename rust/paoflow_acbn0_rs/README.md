# paoflow-acbn0-rs

Optional Rust backend for the four-centre two-electron Coulomb integrals (ERIs)
used by the ACBN0 / eACBN0 modules of [PAOFLOW](https://github.com/marcobn/PAOFLOW).

It is a numerically faithful port of `PAOFLOW.utils.pyints` (Obara–Saika / THO
recurrences + Boys function) exposing a **batched**, `rayon`-parallel API:

- `eri_batch(origins, prim_offsets, exps, coefs, norms, powers, keys)` — on-site
  `(ab|cd)` integrals over a single contracted-Gaussian basis.
- `eri_batch_2c(...i_arrays..., ...j_arrays..., keys)` — intersite `(ik|jl)`
  integrals over two bases (atoms I and J) for eACBN0.

The basis crosses the FFI boundary as flat CSR-style NumPy arrays; each MPI rank
passes only its chunk of unique keys, and Rust parallelises that chunk with
`rayon` while releasing the GIL.

PAOFLOW imports this module opportunistically and falls back to the pure-Python
`pyints` kernel when it is not installed, so it is a fully optional dependency.

## Build / install

```sh
maturin develop --release      # into the active environment
maturin build --release        # produce a wheel in target/wheels
```

## Test

```sh
# Rust golden-value parity tests (vs pyints reference in tests/golden.json)
PYO3_PYTHON=/path/to/python \
DYLD_LIBRARY_PATH=/path/to/python/lib \
cargo test

# Regenerate golden values after changing the Python reference
python scripts/gen_golden.py
```

## Performance / threads

The batched kernels parallelise over keys with `rayon` and release the GIL while
computing. Thread count resolves in this order:

1. `PAOFLOW_ACBN0_THREADS` — if set to a positive integer, a dedicated pool of
   that size is used. Set this to cap intra-rank threads and avoid
   oversubscription when running many MPI ranks per node (e.g. one thread per
   rank, or `cores_per_node / ranks_per_node`).
2. Otherwise the global `rayon` pool is used, which honours `RAYON_NUM_THREADS`
   and defaults to the number of logical cores.

`paoflow_acbn0_rs.thread_count()` returns the effective thread count.

```sh
# Benchmark native vs pyints on a d-shell ERI tensor
python scripts/bench.py --nprim 8

# Cap threads (e.g. when oversubscribed by MPI)
PAOFLOW_ACBN0_THREADS=2 python scripts/bench.py
```

To A/B a real ACBN0 run, force the pure-Python fallback without uninstalling the
wheel by setting `PAOFLOW_ACBN0_DISABLE=1` (truthy: `1`/`true`/`yes`/`on`):

```sh
PAOFLOW_ACBN0_DISABLE=1 python main.acbn0.py   # pyints reference path
python main.acbn0.py                            # native path
```
