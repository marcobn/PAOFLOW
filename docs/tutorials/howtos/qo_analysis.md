# Agentic analysis of quantum oscillations (`PAOFLOW.qo_analysis`)

`PAOFLOW.qo_analysis` compares the extremal orbits computed by `pyskeaf` with measured dHvA, SdH or torque frequencies. It was developed on NbSn2 and is system-agnostic. The workflow is designed to be driven by an AI agent, whose playbook and prompts live in the repository folder `agentic/skeaf/` (see *Agent resources* below), but every step is an ordinary Python method.

Install the optional dependencies with `pip install -e ".[qo_analysis]"`.

## Workflow

```python
from PAOFLOW.qo_analysis import QuantumOscillationAnalysis

qo = QuantumOscillationAnalysis(skeaf_dir='output', output_dir='qo_analysis',
                                primitive_cell=cell, conventional_from_primitive=centring,
                                sweep_angle='polar', fixed_angle=29.95, system_name='NbSn2')
qo.fermi_surface_inventory()                       # bands crossing E_F, Kramers pairs, pockets
qo.field_directions(directions)                    # [hkl] <-> SKEAF angles, both conventions
qo.read_experiment(lk_table='data/exp_LK_table.csv')
qo.digitize_fft_map('data/fft_map.png', ...)       # or read_experiment(sweep_peaks=...)
qo.sweep_comparison(min_copies=50)                 # overlays, per-angle listing, chance tests
qo.direction_analysis()                            # anisotropy ratios, 1 + lambda
qo.compare_runs()                                  # convergence / SOC vs scalar-relativistic
qo.fermi_level_scan('shifted_runs', bands=[7])     # Onsager check of E_F-shifted runs
qo.breakdown_gaps(bands=[11, 13, 15, 17])          # magnetic-breakdown gaps and fields
qo.breakdown_network(bands=[11, 13, 15, 17])       # coupled-network orbits (heavy)
qo.weak_peak_analysis()                            # harmonics / mixing vs breakdown
qo.collaborator_package('package')                 # figures + data + standalone scripts
```

Every method writes tables, figures and a short report to `output_dir/<step>/`.

## Experimental input in any form

Two canonical CSV tables feed the analysis: single-direction LK values, and peaks over a rotation. They can come from:
- tables;
- free text, via `experiment_data.parse_text_peaks`, checked by hand;
- images of FFT-vs-angle maps, via `digitize_fft_map`, which validates itself against the LK values;
- slide decks, via `experiment_data.extract_slide_deck`.

Digitised or transcribed numbers must be labelled as such.

## Statistics

On a dense multi-sheet Fermi surface a ±5 % window around any frequency contains a calculated orbit 30–40 % of the time. Every match rate is therefore reported with two chance baselines: log-uniform random frequencies, and angle-shuffled measured peaks. Assignments are made from the shape of branches over the rotation.

## Agent resources

The agentic workflow lives in the top-level repository folder `agentic/skeaf/`, separate from the library code:
- `README.md`: how to start;
- the playbook (`AGENT_WORKFLOW.md`) and the lessons learned;
- 17 prompts to start, steer, challenge and resume the agent;
- templates (analysis script, canonical tables);
- `projects/`: one folder per study.

Open an agent (e.g. Claude Code) in `agentic/skeaf/` and paste `prompts/00_kickoff.md`; `CLAUDE.md` / `AGENTS.md` in that folder load the instructions automatically.

## Example

`examples/qo_analysis_examples/NbSn2/` contains the complete NbSn2 analysis on bundled, compressed data (SOC and scalar-relativistic runs, digitised torque rotation): `main.py`, `main_sr.py` and `expected_results.json`.
