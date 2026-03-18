# ML-NAMD Handler

Runs non-adiabatic molecular dynamics for the CH2NH2+ system using the [MLatom](https://github.com/dralgroup/mlatom) package with [MACE](https://github.com/acesuit/mace) potential energy surfaces.

## Requirements

Python dependencies are listed in `requirements.txt`.

## Usage

```bash
mpirun -n <number_of_processes> python3 run_dynamics.py <input_file>
```

> **Note:** At least 2 MPI ranks are required (1 master + 1 or more workers), and `traj_number` must be greater than or equal to the number of worker ranks.

## Initial Conditions

The script automatically searches the working directory for initial condition files:

- **Coordinates:** a file matching `stru*.in` — atomic positions in Angstrom
- **Velocities:** a file matching `vel*.in` — atomic velocities in Angstrom/fs

## Input File

An example input file is provided in `input_example.inp`.

### Parameters

| Parameter | Description |
|-----------|-------------|
| `maximum_propagation_time` | Maximum propagation time per trajectory (fs) |
| `time_step` | Time step for MD integration (fs) |
| `traj_number` | Total number of trajectories |
| `nstates` | Total number of electronic states |
| `nens` | Number of ML models in the ensemble — predictions are averaged at each time step |
| `initial_state` | Initial electronic state (0-based index) |
| `model_path` | Path to the folder containing ML model files |
| `model_names` | Model file names (see expansion syntax below) |
| `result_dir` | Directory where output `.xyz` trajectory files are saved |

### Model Name Expansion

`model_names` supports a simple template syntax:

- `{0..2}` expands to `0, 1, 2` (range, inclusive)
- `{0,3}` expands to `0, 3` (explicit list)
- Multiple placeholders produce the Cartesian product of their values

It is assumed that in the model filename the electronic state index comes before the ensemble model index.
