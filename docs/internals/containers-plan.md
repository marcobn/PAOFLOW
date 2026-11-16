# Containers for Quantum ESPRESSO and PAOFLOW (investigation)

Status: investigation of 14 November 2026. Nothing is implemented; the decisions at the end come first.

The goal is to run QE and PAOFLOW, including `paoflow-gen` and `paoflow-studio`, with no compiler, Python
environment or code installed by the user: one image, pulled and run.

## Summary

It is feasible and well-trodden. NVIDIA publishes a QE image for GPUs, and AiiDAlab publishes `aiidalab/qe`, which
bundles QE with a Python stack. A CPU image for x86_64 and arm64 is about half a day of agent work, mostly
compilation and test cycles. GPU images are a separate, larger job. The hard part is not building the image: it is
running MPI across nodes on clusters.

## State of the repository and the machines

The repository has no Dockerfile, Apptainer definition or environment file. Existing pieces:
- CI (`.github/workflows/ci.yaml`) installs Open MPI from apt and PAOFLOW with pip.
- `.github/workflows/paoflow-rs.yml` already builds the Rust extension as abi3 wheels (one wheel for every
  Python ≥ 3.10) with `PyO3/maturin-action`.
- `PSEUDOS/` (`nc-sr-05`, `nc-fr-04`) and `BASIS/` can be copied into an image as they are.

Container runtimes found on 14 November 2026:

| machine | architecture | runtime |
|---|---|---|
| Mac Studio | arm64 | none |
| music | x86_64 | none (Podman or Apptainer needs an administrator) |
| gb10 | aarch64 | Docker |
| Snellius | x86_64 | Apptainer, Podman |
| Lonestar6 | x86_64 | `module load tacc-apptainer` (TACC docs) |

## Contents of the image

Two-stage build: compilers in the builder stage only.
- Base: a minimal Ubuntu 24.04.
- QE 7.5 built with CMake: `pw.x`, `ph.x`, `projwfc.x`, `pp.x`, `q2r.x`, `matdyn.x`, and `epw.x` if the EPW coupling
  route is wanted. Libraries: MPICH, OpenBLAS with `DYNAMIC_ARCH` (kernels chosen at run time), FFTW3, ScaLAPACK.
- Python with PAOFLOW from a wheel of the repository, its core dependencies, and the extras `phonon`, `gui`,
  `pyskeaf` and the scikit-image part of `fermisurface`. Mayavi and VTK stay out: they are large and need a
  display. The `paoflow_rs` wheel is built in the builder stage.
- `PSEUDOS/` and `BASIS/`.
- Commands: `pw.x` and the other QE programs, `paoflow-gen`, `paoflow-studio`, `python`; working folder `/work`
  (a bind mount).

Expected size: about 1.5–2.5 GB for the CPU image; several GB for a GPU image.

| variant | runs on | notes |
|---|---|---|
| CPU, linux/amd64 | music, Snellius, Lonestar6, Intel PCs | one multi-architecture build |
| CPU, linux/arm64 | Mac Studio and MacBooks (Apple Silicon), gb10 | the same build; native arm64 runners in CI |
| GPU, x86_64 CUDA | music GPU | the NGC image stops at QE 7.3.1; QE 7.5 needs the NVIDIA HPC SDK |
| GPU, aarch64 | gb10 (Grace + Blackwell) | NGC images are x86_64 only, so our own build |

## How it would be used

Workstation or laptop (Docker, Podman, OrbStack or Colima):

```bash
docker run --rm --shm-size=2g --user $(id -u):$(id -g) -v $PWD:/work ghcr.io/<org>/paoflow \
    mpirun -np 8 pw.x -in scf.in
docker run --rm -v $PWD:/work -p 8780:8780 ghcr.io/<org>/paoflow \
    paoflow-studio --no-browser --port 8780
```

Clusters (Apptainer): `apptainer pull docker://ghcr.io/<org>/paoflow` gives a single `.sif` file;
`apptainer exec paoflow.sif mpirun -np 64 pw.x -in scf.in` runs on one node. Apptainer runs as the user, with
no ownership issue.

## Difficulties

1. **MPI across nodes.** The container's `mpirun` works on one node, and most QE and PAOFLOW jobs fit on one node
   (Snellius has 128–192 cores per node). Across nodes, the container MPI must work with the site launcher and
   interconnect:
   - Apptainer's *hybrid* model ships MPI in the image and needs an MPI on the host with a compatible ABI.
   - Its *bind* model mounts the host MPI into the container, which some sites forbid.
   - On Lonestar6, TACC asks for `ibrun apptainer …`, one container per rank. Intel MPI and MVAPICH there belong
     to the MPICH ABI family, so an MPICH image fits.
   - On Snellius the native MPI is Open MPI, which does not share the MPICH ABI, so multi-node runs there need the
     container's own fabric libraries (libfabric or UCX), tested on the site.

   Plan: single-node runs first; multi-node runs per site, each with a test.
2. **QE performance.** A portable build with OpenBLAS and FFTW will be slower than the site-tuned builds (MKL or
   AOCL; the "aggressive" build on music). The difference has to be measured before it is quoted. Containers are
   therefore for onboarding, teaching, reproducibility and small or medium runs; production keeps the native builds.
3. **Small code and documentation changes.**
   - `paoflow-studio` listens on 127.0.0.1 only, which a published container port cannot reach. It needs a
     bind-address option, used only inside a container, with the token and the Host/Origin checks unchanged. The
     host port and the container port must be the same number, because of the Host check.
   - Docker gives 64 MB of `/dev/shm`, too little for MPI's shared memory: `--shm-size=2g` or `--ipc=host`.
   - Docker writes files as root unless it runs with `--user $(id -u):$(id -g)`.
4. **GPU images.**
   - They need the NVIDIA HPC SDK and the CUDA compute capability of each GPU.
   - They must be tested on music's GPU and on gb10 (aarch64).
   - The NGC image needs a CUDA driver ≥ 550, or the r535 or r470 branches, and is x86_64 only.
5. **macOS.** Containers run in a Linux virtual machine. An arm64 image runs at near-native CPU speed within the
   memory given to the VM, with no Metal or GPU.

## Effort (agentic estimates)

| step | agent time | wall time |
|---|---|---|
| Two-stage Dockerfile, CPU, both architectures | ~30 min | QE compile ~20–40 min per architecture |
| CI workflow: buildx, native arm64 runner, push to ghcr.io on release tags (free for public images) | ~30 min | first build |
| Studio bind-address option and docs | ~30 min | — |
| Smoke test in the image: Si scf, then PAOFLOW bands and DOS against the test references | ~30 min | minutes |
| Tests on the Mac, gb10 and Snellius | — | about 1 h, once a runtime is on the Mac |
| GPU images | 1–2 days | mostly build and test cycles |

## Decisions for the user

- Registry and name (for example `ghcr.io/<org>/paoflow`), public or private.
- Contents: EPW or not, which PAOFLOW extras.
- MPI: MPICH (portable ABI: Lonestar6, Cray and Intel MPI sites) or Open MPI (Snellius native).
- GPU images now or later.
- The runtime for the Mac: Docker Desktop (free for education), OrbStack, Colima or Podman.
- Licences:
  - QE is GPL-2.0 and PAOFLOW GPL-3.0, so a public image must point to the corresponding source; a public Dockerfile
    and a tagged release do.
  - The redistribution terms of the pseudopotential libraries in the image need checking.
  - A GPU image redistributes NVIDIA runtime libraries under NVIDIA's licence.

## A lighter alternative

conda-forge has `qe` 7.5 for linux-64, linux-aarch64 and osx-64; on 14 November 2026 there was no osx-arm64 build.
With PAOFLOW from pip it gives a one-command environment with no compiler (an `environment.yml` or a `pixi`
manifest). It is still an installation, so it would complement the containers rather than replace them.

## Sources

- conda-forge `qe` package: <https://anaconda.org/conda-forge/qe>
- NVIDIA NGC Quantum ESPRESSO container: <https://catalog.ngc.nvidia.com/orgs/hpc/containers/quantum_espresso>
- QE through Apptainer, Sherlock (Stanford): <https://www.sherlock.stanford.edu/docs/software/using/quantum-espresso>
- Building the NGC image with Apptainer, Vega: <https://vegadocs.vega.izum.si/qe/>
- Apptainer and MPI applications: <https://apptainer.org/docs/user/1.5/mpi.html>
- MPI support in containers at TACC (Cornell CVW): <https://cvw.cac.cornell.edu/containers/tacc/mpi-support>
- Containers at TACC, MPI and GPU containers: <https://containers-at-tacc.readthedocs.io/en/latest/singularity/03.mpi_and_gpus.html>
- ARCHER2, The State of Containers (ABI compatibility): <https://www.archer2.ac.uk/training/courses/200826-containers/The%20State%20of%20Containers.pdf>
- MPI jobs with Apptainer (Carpentries): <https://carpentries-incubator.github.io/apptainer-introduction/08-apptainer-mpi/index.html>
- AiiDAlab QE app and image: <https://pypi.org/project/aiidalab-qe/>, <https://aiidalab.readthedocs.io/en/latest/usage/access/desktop.html>
