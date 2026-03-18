import re
from itertools import product
import sys
import os

import mlatom as ml
import numpy as np
import torch
from mpi4py import MPI
import uuid


# --- Input file parsing ---

def parse_blocks(filename):
    def parse_fstring_value(value_str):
        if value_str.startswith('f"') and value_str.endswith('"'):
            content = value_str[2:-1]
        elif value_str.startswith("f'") and value_str.endswith("'"):
            content = value_str[2:-1]
        else:
            return value_str.strip(r"\"\'")

        brace_pattern = r'\{([^}]+)\}'
        matches = list(re.finditer(brace_pattern, content))

        if not matches:
            return [content]

        replacements = []
        for match in matches:
            brace_content = match.group(1)
            if '..' in brace_content:
                parts = brace_content.split('..')
                if len(parts) == 2:
                    start = int(parts[0].strip())
                    end = int(parts[1].strip())
                    if start <= end:
                        range_values = [str(i) for i in range(start, end + 1)]
                    else:
                        range_values = [str(i) for i in range(start, end - 1, -1)]
                    replacements.append(range_values)
                else:
                    replacements.append([brace_content])
            else:
                values = [val.strip() for val in brace_content.split(',')]
                replacements.append(values)

        result_list = []
        for combo in product(*replacements):
            temp_content = content
            for i, match in enumerate(matches):
                pattern = match.group(0)
                replacement = str(combo[i])
                temp_content = temp_content.replace(pattern, replacement, 1)
            result_list.append(temp_content)

        return result_list

    blocks = {}
    current_block = None
    current_vars = {}

    with open(filename, 'r') as file:
        for line_num, line in enumerate(file, 1):
            if '#' in line:
                line = line.split('#', 1)[0]
            line = line.strip()

            if not line:
                continue

            if line.startswith('$') and line != '$end':
                if current_block is not None:
                    raise ValueError(f"Unclosed block '{current_block}' at line {line_num}")
                current_block = line[1:].strip()
                current_vars = {}

            elif line == '$end':
                if current_block is None:
                    raise ValueError(f"Unexpected $end at line {line_num}")
                blocks[current_block] = current_vars
                current_block = None

            elif current_block is not None:
                if '=' not in line:
                    raise ValueError(f"Invalid syntax at line {line_num}: {line}")
                var_name, var_value = line.split('=', 1)
                var_name = var_name.strip()
                var_value = var_value.strip()
                parsed_value = parse_fstring_value(var_value)
                current_vars[var_name] = parsed_value

    if current_block is not None:
        raise ValueError(f"Unclosed block '{current_block}' at end of file")

    return blocks


# --- ML models ---

class mlmodels():
    def __init__(self, nstates=3, folder_with_models='./', model_files=None, Nens=1):
        self.Nens = Nens
        self.models = [None for istate in range(nstates)]
        for istate in range(nstates):
            self.models[istate] = [ml.models.mace(model_file=folder_with_models+model_files[istate][ii], device='cpu') for ii in range(self.Nens)]

    def predict(self, molecule=None, nstates=3, current_state=0,
                calculate_energy=True, calculate_energy_gradients=True):
        molecule.electronic_states = [molecule.copy() for ii in range(nstates)]

        for istate in range(nstates):
            moltmp = molecule.electronic_states[istate]
            moltmpens = [moltmp.copy() for ii in range(self.Nens)]
            for ii in range(self.Nens):
                self.models[istate][ii].predict(molecule=moltmpens[ii], calculate_energy=True, calculate_energy_gradients=True)
            moltmp.energy = np.mean([moltmpens[ii].energy for ii in range(self.Nens)])
            moltmp.energy_gradients = np.mean([moltmpens[ii].energy_gradients for ii in range(self.Nens)], axis=0)

        molecule.energy = molecule.electronic_states[current_state].energy
        molecule.energy_gradients = molecule.electronic_states[current_state].energy_gradients


# --- MPI dynamics driver ---

class run_mpi_dynamics():
    def __init__(self, control_dict):
        if 'maximum_propagation_time' not in control_dict.keys():
            raise RuntimeError("maximum_propagation_time is not specified!!!")
        else:
            self.maximum_propagation_time = float(control_dict['maximum_propagation_time'])

        self.time_step = float(control_dict['time_step'])
        self.nstates = int(control_dict['nstates'])
        self.initial_state = int(control_dict['initial_state'])
        self.result_dir = control_dict['result_dir'].rstrip('/') + '/'
        self.nens = int(control_dict['nens'])
        self.folder_with_models = control_dict['model_path'].rstrip('/') + '/'
        self.model_names = control_dict['model_names']
        self.traj_number = int(control_dict['traj_number'])

        if self.nens * self.nstates != len(self.model_names):
            raise RuntimeError("Number of models names does not match nens*nstates!!!")
        self.model_names_list_of_lists = [self.model_names[i:i+self.nens] for i in range(0, len(self.model_names), self.nens)]

        if 'dump_trajectory_interval' in control_dict:
            val = control_dict['dump_trajectory_interval']
            self.dump_trajectory_interval = None if str(val).lower() == 'none' else int(val)
        else:
            self.dump_trajectory_interval = None

        if control_dict['multihead'].lower() == 'true':
            raise NotImplementedError('Multihead support has not been added')

    def stop_function(self, stop_check=False, mol=None, current_state=None, **kwargs):
        if np.max(np.abs(mol.energy_gradients)) > 10.0:
            return True, stop_check
        else:
            return False, stop_check

    def save_xyz(self, fname, symbols, xyz_coords_arr, comments_arr):
        traj_id = str(uuid.uuid4())
        with open(fname, 'w') as fil:
            for i, xyz in enumerate(xyz_coords_arr):
                fil.write(f'{len(symbols)}\n')
                fil.write('id = ' + traj_id + ' ' + comments_arr[i] + '\n')
                for ii in range(len(symbols)):
                    fil.write(f'{symbols[ii]}{xyz[ii,0]:22.12f}{xyz[ii,1]:22.12f}{xyz[ii,2]:22.12f}\n')

    def run(self):
        self.world_comm = MPI.COMM_WORLD
        self.world_size = self.world_comm.Get_size()
        my_rank = self.world_comm.Get_rank()

        if self.world_size < 2:
            raise RuntimeError("At least 2 MPI ranks required (1 master + 1 worker)")
        if self.traj_number < self.world_size - 1:
            raise RuntimeError(f"traj_number ({self.traj_number}) must be >= number of worker ranks ({self.world_size - 1})")

        print("World Size: " + str(self.world_size) + "   " + "Rank: " + str(my_rank))

        if my_rank == 0:
            self.main_code()
        else:
            self.subprocess_code(my_rank)

    def main_code(self):
        print("=== Dynamics Parameters ===")
        print(f"  maximum_propagation_time : {self.maximum_propagation_time} fs")
        print(f"  time_step                : {self.time_step} fs")
        print(f"  nstates                  : {self.nstates}")
        print(f"  initial_state            : {self.initial_state}")
        print(f"  nens                     : {self.nens}")
        print(f"  traj_number              : {self.traj_number}")
        print(f"  result_dir               : {self.result_dir}")
        print(f"  folder_with_models       : {self.folder_with_models}")
        print(f"  model_names              : {self.model_names}")
        print(f"  dump_trajectory_interval : {self.dump_trajectory_interval}")
        print("===========================")

        received = 0
        os.makedirs(self.result_dir, exist_ok=True)
        seeds = np.random.randint(0, 2147483647, self.traj_number)

        lis = os.listdir('.')
        stru_filename = ''
        vel_filename = ''
        for filname in lis:
            if ('.in' in filname) and ('stru' in filname):
                stru_filename = filname
            if ('.in' in filname) and ('vel' in filname):
                vel_filename = filname

        print(stru_filename, vel_filename)

        cur_ic = 0
        for i in range(1, self.world_size):
            i_np = np.array([cur_ic, 1, seeds[cur_ic]], dtype=int)
            self.world_comm.Send([i_np, MPI.INT], dest=i, tag=77)
            cur_ic = cur_ic + 1

        while True:
            curic_np = np.empty(3, dtype=int)
            status = MPI.Status()
            self.world_comm.Recv([curic_np, MPI.INT], source=MPI.ANY_SOURCE, tag=77, status=status)
            source = status.Get_source()
            print(f'(master) received from {source}, ic: {curic_np[0]}')
            received = received + 1
            if received == self.traj_number:
                i_np = np.array([0, 0, 0], dtype=int)
                for i in range(1, self.world_size):
                    self.world_comm.Send([i_np, MPI.INT], dest=i, tag=77)
                break
            if cur_ic < self.traj_number:
                i_np = np.array([cur_ic, 1, seeds[cur_ic]], dtype=int)
                self.world_comm.Send([i_np, MPI.INT], dest=source, tag=77)
                print(f'(master) sent to {source}, ic: {cur_ic}, seed: {seeds[cur_ic]}')
                cur_ic = cur_ic + 1

    def subprocess_code(self, my_rank):
        mol = ml.data.molecule()
        mol.charge = 1
        mol.read_from_xyz_file("cnh4+.xyz")

        lis = os.listdir('.')
        stru_filename = ''
        vel_filename = ''
        for filname in lis:
            if ('.in' in filname) and ('stru' in filname):
                stru_filename = filname
            if ('.in' in filname) and ('vel' in filname):
                vel_filename = filname

        print(stru_filename, vel_filename)

        init_cond_db = ml.generate_initial_conditions(molecule=mol, generation_method='user-defined',
                                                      file_with_initial_xyz_coordinates=stru_filename,
                                                      file_with_initial_xyz_velocities=vel_filename,
                                                      number_of_initial_conditions=self.traj_number)

        models = mlmodels(nstates=self.nstates, folder_with_models=self.folder_with_models,
                          model_files=self.model_names_list_of_lists, Nens=self.nens)

        while True:
            curic_np = np.empty(3, dtype=int)
            self.world_comm.Recv([curic_np, MPI.INT], source=0, tag=77)
            print(f'(slave) rank = {my_rank}, ic = {curic_np[0]}, seed = {curic_np[2]}')
            if curic_np[1] == 0:
                print('(slave) exiting...')
                break

            namd_kwargs = {
                'model': models,
                'time_step': self.time_step,
                'maximum_propagation_time': self.maximum_propagation_time,
                'dump_trajectory_interval': self.dump_trajectory_interval,
                'hopping_algorithm': 'LZBL',
                'nstates': self.nstates,
                'initial_state': self.initial_state,
                'format': 'json',
                'stop_function': self.stop_function,
                'random_seed': curic_np[2],
                'rescale_velocity_direction': 'along velocities',
                'reduce_kinetic_energy': False,
            }

            dyns = ml.simulations.run_in_parallel(molecular_database=init_cond_db[curic_np[0]:curic_np[0]+1],
                                                   task=ml.namd.surface_hopping_md, task_kwargs=namd_kwargs, nthreads=1)

            steps = dyns[0].molecular_trajectory.steps
            coords = [s.molecule.xyz_coordinates for s in steps]
            states = [s.current_state for s in steps]
            energies = [s.molecule.state_energies for s in steps]
            symbols = steps[0].molecule.element_symbols
            times = [s.time for s in steps]

            comments = [None for i in range(len(coords))]
            for i in range(len(coords)):
                e_comm = " ".join(f"E{ii} = {v:.12e}" for ii, v in enumerate(energies[i]))
                cur_comm = f"state = {states[i]} " + e_comm + f" (Hartree, ANGS) time = {times[i]:.3f} fs"
                comments[i] = cur_comm

            self.save_xyz(self.result_dir + f'traj_{curic_np[0]}.xyz', symbols, coords, comments)

            self.world_comm.Send([curic_np, MPI.INT], dest=0, tag=77)


# --- Entry point ---

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 run_dynamics.py <input_file>")
        sys.exit(1)

    inp_filename = sys.argv[1]

    blocks = parse_blocks(filename=inp_filename)

    if 'dynamics' not in blocks:
        raise RuntimeError("DYNAMICS BLOCK NOT FOUND!!!")

    dyn_block = blocks['dynamics']

    if 'parallel' not in dyn_block:
        raise RuntimeError("No parallel keyword found!!!")

    if dyn_block['parallel'] != 'MPI':
        raise RuntimeError(f"Unsupported parallel mode: {dyn_block['parallel']}. Only MPI is supported.")

    print(dyn_block['model_names'])
    dyn_driver = run_mpi_dynamics(dyn_block)
    dyn_driver.run()
