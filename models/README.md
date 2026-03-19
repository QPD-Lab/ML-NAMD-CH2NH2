# Model Weights / Checkpoints

`models/` contains the trained MACE weight files (`*.model`) used by the repository experiments.

## How to read the filename prefixes

The part after `MACE_x_with_forces_` (before `_fixed...`) indicates the training protocol:

* `scratch` - single-state models with random initialization (single-state random initialization)
* `finetuned` - single-state transfer learning models (single-state transfer learning)
* `delta` - single-state delta-learning models (predicting `XMCQDPT2 - CASSCF` corrections)
* `mst` - multi-output random initialization models (multi-output, random initialization)
* `msf` - multi-output transfer learning models (multi-output, transfer learning)
* `mst_delta` - multi-output delta-learning models (multi-output, delta-learning against CASSCF)

## Notes on ensembles

Most variants are stored as an ensemble of multiple checkpoints, distinguished by the suffix:

* `..._fixed_0_stagetwo.model` ... `..._fixed_4_stagetwo.model` (5 checkpoints)

