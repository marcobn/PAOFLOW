# PAOFLOW Rust backends

This directory holds the optional, compiled Rust extensions that accelerate the
heaviest numerical kernels in PAOFLOW. They are **fully optional**: if a backend
is not installed, PAOFLOW automatically falls back to its pure-Python
implementation, producing numerically identical results (parity < 1e-12).

| Crate | Importable module | Accelerates | Python fallback |
| --- | --- | --- | --- |
| [`paoflow_acbn0_rs`](paoflow_acbn0_rs/) | `paoflow_acbn0_rs` | Four-centre two-electron Coulomb integrals (ERIs) in ACBN0 / eACBN0 | `PAOFLOW.utils.pyints` |

This document is the end-to-end guide for getting a backend running on a fresh
machine — whether you are an **end user** who just wants the speedup, or a
**developer** who wants to build, test, and modify the crate. The crate-level
[`paoflow_acbn0_rs/README.md`](paoflow_acbn0_rs/README.md) has the API reference
and internals.

---

## 1. What the backend does and why it is optional

ACBN0 / eACBN0 self-consistently compute Hubbard `U` (and intersite `V`) by
evaluating four-centre Coulomb integrals over contracted Gaussians. That ERI
sum is the compute hotspot. The Rust crate is a numerically faithful port of the
`pyints` kernel (Obara–Saika / THO recurrences + Boys function) that:

- evaluates the integrals in compiled code (no per-integral Python overhead),
- batches a whole chunk of `(ab|cd)` keys across a `rayon` thread pool,
- releases the GIL while computing, so it composes with PAOFLOW's MPI layer.

PAOFLOW imports the module opportunistically. The bridge
`PAOFLOW.acbn0_native` reports `available() == True` only when the compiled
module imports successfully **and** the backend is not force-disabled (see
§6). Everything downstream is automatic — no PAOFLOW script changes are needed.

> Measured reference (Apple M-series, 10 threads, synthetic d-shell ERI tensor):
> ~960–1100× faster than `pyints` at parity ≈ 3e-16. Real-run speedup depends on
> how ERI-dominated the system is (largest for transition-metal d/f manifolds).

---

## 2. Prerequisites

### End users (install a prebuilt wheel)

If you install a published / CI-built wheel, you need **only Python ≥ 3.10**.
The wheels are `abi3` (stable ABI) and contain no Rust source — **no Rust
toolchain is required**.

### Developers (build from source)

You additionally need a Rust toolchain and `maturin`:

- **Rust ≥ 1.71** (a transitive dependency requires it; newer is fine). Install
  via [rustup](https://rustup.rs):
  ```sh
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  rustup update stable          # if rustc is older than 1.71
  rustc --version               # verify >= 1.71
  ```
- **maturin ≥ 1.5, < 2** (the PEP 517 build backend that compiles the crate into
  a wheel):
  ```sh
  python -m pip install "maturin>=1.5,<2"
  maturin --version
  ```
- A C/Rust-capable build environment (Xcode Command Line Tools on macOS,
  `build-essential` on Linux). Windows is supported via the MSVC toolchain.

Supported platforms: Linux (x86_64 / aarch64), macOS (Intel / Apple Silicon),
Windows (x86_64).

---

## 3. Install for end users (prebuilt wheel)

A backend is just a normal Python package. Install it into the **same
environment** that runs PAOFLOW:

```sh
# From a wheel file or directory of wheels
pip install paoflow_acbn0_rs-0.1.0-cp310-abi3-<platform>.whl

# or, once published to an index
pip install paoflow-acbn0-rs
```

That is the entire installation. PAOFLOW will pick it up on the next run.

> CI builds these wheels for Linux/macOS/Windows (see §9). Until they are
> published to an index, grab the artifact from the CI run or build one yourself
> with `maturin build --release` (§4).

Verify (see §5) and you are done.

---

## 4. Build from source (developers)

All commands below run from this `rust/` directory unless noted; the crate lives
in `paoflow_acbn0_rs/`.

### 4a. Build and install into the active environment (editable dev loop)

```sh
cd paoflow_acbn0_rs
maturin develop --release
```

This compiles an optimized `abi3` extension and installs it (editable) into the
currently active Python environment. Re-run it after any change to the Rust
source so Python picks up the new build.

### 4b. Build a distributable wheel (to ship to other machines)

```sh
cd paoflow_acbn0_rs
maturin build --release            # wheel written under target/wheels/
```

Copy the resulting `*.whl` to the target machine and `pip install` it (§3). The
wheel is self-contained — the target machine needs no Rust toolchain.

### 4c. Environment gotchas when building

These caused real failures and are the most common build snags:

- **`maturin` errors: "Both VIRTUAL_ENV and CONDA_PREFIX are set."** Unset one.
  With a conda env, prefer unsetting `CONDA_PREFIX` for the command:
  ```sh
  env -u CONDA_PREFIX VIRTUAL_ENV="$CONDA_PREFIX" \
      PYO3_PYTHON="$(which python)" \
      maturin develop --release
  ```
- **Wrong interpreter picked up.** Set `PYO3_PYTHON` to the exact interpreter you
  want the extension built against (must be ≥ 3.10).
- **`rustc` too old.** If the build complains about a required rustc version,
  run `rustup update stable`.

---

## 5. Verify the install

Run this in the environment where you installed the backend:

```sh
python -c "
import PAOFLOW.acbn0_native as n
import paoflow_acbn0_rs as rs
print('available    :', n.available())     # expect: True
print('version      :', rs.__version__)
print('thread_count :', rs.thread_count())
print('module file  :', rs.__file__)
"
```

`available() == True` means PAOFLOW will use the Rust kernel automatically.

Optional numerical sanity check against the pure-Python kernel:

```sh
python rust/paoflow_acbn0_rs/scripts/bench.py --nprim 4
# prints max relative error (~1e-16) and a native-vs-pyints speedup
```

---

## 6. Use it in a PAOFLOW run

There is **nothing to change** in your PAOFLOW scripts. Any ACBN0 / eACBN0 run —
including a `main.acbn0.py` produced by `paoflow-gen` — dispatches to the Rust
kernel automatically when it is installed, and to `pyints` otherwise. The
acceleration applies inside the Hartree / intersite ERI step, which PAOFLOW runs
under its `mpi_hartree` launcher; the relevant environment variables (below) are
inherited by that subprocess.

### A/B compare native vs pure-Python in a real run

Force the pure-Python fallback without uninstalling the wheel:

```sh
PAOFLOW_ACBN0_DISABLE=1 python main.acbn0.py   # pyints reference path
python main.acbn0.py                            # native path (default)
```

`PAOFLOW_ACBN0_DISABLE` accepts `1` / `true` / `yes` / `on`. Results are
numerically identical, so this isolates pure timing.

---

## 7. Threads and oversubscription

The batched kernels parallelise over keys with `rayon`. Thread count resolves
as:

1. **`PAOFLOW_ACBN0_THREADS=N`** (positive integer) → a dedicated pool of `N`
   threads. Use this to cap intra-rank threads.
2. Otherwise the global `rayon` pool is used, honouring **`RAYON_NUM_THREADS`**
   and defaulting to the number of logical cores.

Because the Hartree step already runs under MPI (`mpi_hartree`), you now have two
parallelism layers (MPI ranks × rayon threads). To avoid oversubscribing a node:

```sh
# one thread per rank (pure MPI behaviour)
PAOFLOW_ACBN0_THREADS=1 mpirun -np <ranks> ...

# or roughly cores_per_node / ranks_per_node
PAOFLOW_ACBN0_THREADS=$((CORES / RANKS)) mpirun -np <ranks> ...
```

`paoflow_acbn0_rs.thread_count()` reports the effective count.

---

## 8. Run the test suites

### Rust golden-value parity tests

The crate ships golden values generated from the real `pyints` reference
(`paoflow_acbn0_rs/tests/golden.json`). Because plain `cargo test` links
`libpython` (the `extension-module` feature is off for tests), point it at your
interpreter and put `libpython` on the loader path:

```sh
cd paoflow_acbn0_rs

# Linux:
PYO3_PYTHON=/path/to/python \
LD_LIBRARY_PATH=/path/to/python/lib \
cargo test

# macOS:
PYO3_PYTHON=/path/to/python \
DYLD_LIBRARY_PATH=/path/to/python/lib \
cargo test

# Regenerate golden values after changing the Python reference kernel
python scripts/gen_golden.py
```

### Python parity tests

These build/import the extension and compare native output to `pyints`:

```sh
python -m pytest tests/unit/acbn0 -q
```

### Lint / format (developers)

```sh
cd paoflow_acbn0_rs
cargo fmt
PYO3_PYTHON=/path/to/python cargo clippy --all-targets
```

---

## 9. Continuous integration

`.github/workflows/acbn0-rust.yml` builds and validates the crate on every push
/ PR that touches `rust/`:

- **test job** (Ubuntu + macOS): `cargo fmt --check`, `clippy`, regenerate golden
  values, `cargo test`, `maturin develop`, then the `tests/unit/acbn0` pytest.
- **wheels job** (Ubuntu + macOS + Windows): builds `abi3` wheels with
  `PyO3/maturin-action` and uploads them as artifacts.

The wheels are not yet published to a package index, so PAOFLOW's top-level
`pyproject.toml` intentionally does **not** declare a `native` extra (adding it
prematurely would break `pip install PAOFLOW[native]`). Until publication, use a
CI artifact or build locally (§4).

---

## 10. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `available()` is `False` but import works | `PAOFLOW_ACBN0_DISABLE` is set to a truthy value — unset it. |
| `ImportError: No module named paoflow_acbn0_rs` | Wheel not installed in the active environment. Install it (§3/§4) into the same env that runs PAOFLOW. |
| `maturin failed: Both VIRTUAL_ENV and CONDA_PREFIX are set` | Unset one; see §4c. |
| Build error: requires rustc ≥ 1.71 | `rustup update stable`. |
| `cargo test`: `Library not loaded: libpython3.x.dylib` (or `.so`) | Add the interpreter's `lib` dir to `DYLD_LIBRARY_PATH` (macOS) / `LD_LIBRARY_PATH` (Linux); see §8. |
| Slower than expected on a cluster | Oversubscription — set `PAOFLOW_ACBN0_THREADS` to balance MPI ranks vs threads (§7). |
| Want to confirm it is actually being used | Compare a run with and without `PAOFLOW_ACBN0_DISABLE=1` (§6), or check `available()` (§5). |
