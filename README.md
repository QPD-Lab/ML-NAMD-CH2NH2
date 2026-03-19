# ML-NAMD-CH2NH2 (model artifacts)

This repository contains the pre-trained machine-learning interatomic potential (MLIP) artifacts used in the paper:

* *XMCQDPT2-Fidelity Transfer-Learning Potentials and a Wavepacket Oscillation Model with Power-Law Decay for Ultrafast Photodynamics* (arXiv: https://arxiv.org/html/2512.07537v1).

The repository is organized into a few artifact folders:

`models/`:
Trained MACE weight files (`*.model`). There are single-state and multi-output variants trained with different protocols (e.g., random initialization, transfer learning, and delta-learning). See `models/README.md` for the naming convention.

`data/`:
Reference datasets in XYZ format used for training/validation/testing (e.g., `casscf_with_forces_*` and `x_with_forces_*` files for `train`/`val`/`test`, including conical-intersection-proximal samples).

`notebooks/`:
Jupyter notebooks used to generate/collect figures and results for the paper (currently the main notebook is `msani.ipynb`).

`scripts/`:
Training scripts and configuration files used to run MACE training ensembles (bash launchers like `train_ensemble_SSRI.sh`, `train_ensemble_SSTL.sh`, `train_ensemble_delta.sh`, `train_ensemble_msf.sh`, `train_ensemble_mst.sh`, `train_ensemble_mst_delta.sh`), plus `mst_x2.yml` / `mst_x_delta.yml` configs for multi-output variants. The top-level training entry point is `scripts/run_train.py` (a small wrapper around `mace.cli.run_train.main`).

`NAMD/`:
An MPI runner for non-adiabatic dynamics using `MLatom` with MACE potentials (`run_dynamics.py`), plus `requirements.txt`, an example input file (`input_example.inp`) and a simple molecular geometry file (`cnh4+.xyz`). See `NAMD/README.md` for usage.