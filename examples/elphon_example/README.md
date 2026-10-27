# Moved

The electron-phonon examples now live in [`examples/elphon/`](../elphon/README.md), one folder
per system (`Si`, `Pb`, `MgB2`, `SiC`) with the QE inputs under `qe/`, one `system.py` holding the
configuration and one subfolder per workflow (`phonons/`, `coupling/`, `superconductivity/`,
`optics/`, `transport/`, `arpes/`).  `paoflow-gen elphon` writes the same layout.  This directory
will be removed in the next release.
