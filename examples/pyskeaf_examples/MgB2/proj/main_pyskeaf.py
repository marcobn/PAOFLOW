from PAOFLOW.pyskeaf.config import read_config_in
from PAOFLOW.pyskeaf.runner import run_paoflow_bxsf_files


def main():
    cfg = read_config_in("config.in")
    results = run_paoflow_bxsf_files(
        cfg,
        input_dir="output_paoflow",
        all_files=True,
        output_dir="output_pyskeaf",
    )
    for item in results:
        print(f"{item.path.name}: calculated" if item.calculated else f"{item.path.name}: skipped - {item.skipped_reason}")


if __name__ == "__main__":
    main()
