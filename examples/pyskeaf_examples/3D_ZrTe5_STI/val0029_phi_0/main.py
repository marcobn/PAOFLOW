from PAOFLOW.pyskeaf.config import read_config_in
from PAOFLOW.pyskeaf.io_bxsf import read_bxsf
from PAOFLOW.pyskeaf.runner import run_skeaf

cfg = read_config_in("config.in")
cfg.n_jobs = -1            # all cores; or 4, 8, etc.
bxsf = read_bxsf(cfg.filename)
run_skeaf(cfg, bxsf, output_dir=".")
