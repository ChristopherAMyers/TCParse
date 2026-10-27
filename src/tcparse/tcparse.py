from __future__ import annotations
import os
import json
import numpy as np
import io
from pprint import pprint
import sys

from pydantic import ValidationError
from typing import Optional, Union, get_args, get_origin

from .units import *
from .cis import CISParser
from .cas import CASParser
from .job_data import TCJobData

class XYZFile():
    def __init__(self, file_loc) -> None:
        self._atom_data = []
        self._coord_data = []
        self._comment_data = []

        self._n_atoms = 0
        with open(file_loc) as file:
            for line in file:
                #   sometimes a blank line is inserted between each frame
                if len(line.split()) == 0:
                    continue
                if self._n_atoms == 0:
                    self._n_atoms = int(line)
                line = next(file)
                self._comment_data.append(line)
                atoms = []
                coords = []
                for n in range(self._n_atoms):
                    line = next(file)
                    sp = line.split()
                    atoms.append(sp[0])
                    coords.append([float(x) for x in sp[1:]])
                self._atom_data.append(atoms)
                self._coord_data.append(coords)


    @property
    def n_frames(self):
        return len(self._comment_data)
    
    def get_atoms(self, frame=0):
        return self._atom_data[frame]
    def get_coords(self, frame=0):
        return self._coord_data[frame]
    def get_comment(self, frame=0):
        return self._comment_data[frame]
    
    def get_frame_num_from_comment(self, frame):
        line = self._comment_data[frame]
        sp = line.split()
        where = sp.index('frame')
        comment_frame = int(sp[where + 1])
        return comment_frame

def _assign_charge_info(new_data: dict, old_data: dict, key):
    for x in ['esp_charges', 'resp_charges']:
        new_data.setdefault(f'{key}{x}', []).append(old_data[x])
    for x in ['esp_dipole', 'resp_dipole']:
        new_data.setdefault(f'{key}{x}s', []).append(old_data[x])

def _reorder_charge_info(data: dict):
    energies = data['energy']
    n_states = len(energies)
    new_esp_data = {}

    exc_type = 'cis'
    if 'cas_states' in data:
        exc_type = 'cas'

    # all_esp_analysis_keys = [k for k in data if 'esp_analysis_' in k]
    if 'esp_analysis_S0' in data:
        # _assign_charge_info(new_esp_data, data['esp_analysis_S0'], '')
        new_esp_data['esp_charges'] = data['esp_analysis_S0']['esp_charges']
        new_esp_data['resp_charges'] = data['esp_analysis_S0']['resp_charges']
        new_esp_data['resp_dipole'] = data['esp_analysis_S0']['resp_dipole']
        new_esp_data['esp_dipole'] = data['esp_analysis_S0']['esp_dipole']
        pass
        

    if n_states > 1 and 'esp_analysis_S1' in data:
        #   if the S1 state is included, then all excited charges must have been requested
        exc_key = 'cis_unrelaxed_'
        if exc_type == 'cas':
            exc_key = 'cas_'

        for i in range(1, n_states):
            key = f'esp_analysis_S{i}'
            esp_data = data[key]
            _assign_charge_info(new_esp_data, esp_data, exc_key)
            data.pop(key)

    if n_states > 1 and 'esp_analysis_S0_S1' in data:            
        for i in range(0, n_states):
            for j in range(i+1, n_states):
                key = f'esp_analysis_S{i}_S{j}'
                if key not in data:
                    continue
                esp_data = data[key]
                _assign_charge_info(new_esp_data, esp_data, f'{exc_type}_tr_')
                data.pop(key)

    for i in range(1, n_states):
        if n_states > 1 and f'esp_analysis_S{i}_relaxed' in data: 
            key = f'esp_analysis_S{i}_relaxed'
            esp_data = data[key]
            _assign_charge_info(new_esp_data, esp_data, 'cis_relaxed_')
            data.pop(key)
    
    data.update(new_esp_data)

def correct_cis_signs(data: dict, compare_vecs=None):
    energies = data['energy']
    n_states = len(energies)

    #   use the transition dipoles relative closest to the included vectors
    #   to determine if the sign of the excited state should flip or not
    signs = {0: 1.0}
    if compare_vecs is None:
        gs_dipole = data['dipole_vector']
        compare_vecs = [gs_dipole]
    for i, vec in enumerate(compare_vecs):
        compare_vecs[i] = vec/np.linalg.norm(vec)
    
    data['cis_transition_dots'] = []
    for i in range(n_states-1):
        cis_dipole = data['cis_transition_dipoles'][i]
        cis_dipole /= np.linalg.norm(cis_dipole)

        dots = []
        for vec in compare_vecs:
            dp = np.dot(cis_dipole, vec)
            dots.append(dp)
        closest_idx = np.argmax(np.abs(dots))
        dp = dots[closest_idx]

        if dp > 0:
            signs[i+1] = 1.0
        else:
            signs[i+1] = -1.0

        data['cis_transition_dots'].append(dots)


    count = 0
    for i in range(0, n_states):
        for j in range(i+1, n_states):
            esp_key = f'esp_analysis_S{i}_S{j}'
            if esp_key in data:
                for prop_key in ['esp_charges', 'resp_charges', 'esp_dipole', 'resp_dipole']:
                    prop = np.array(data[esp_key][prop_key])
                    data[esp_key][prop_key] = (prop*signs[i]*signs[j]).tolist()
            
            tr_dipole = data['cis_transition_dipoles'][count]
            data['cis_transition_dipoles'][count] = (np.array(tr_dipole)*signs[i]*signs[j]).tolist()
            count += 1
    
def parse_from_list(file_lines: list[str], coords_file: str=None, data_output_file=None) -> TCJobData:
    '''
        Parse a list lines taken from a TeraChem output.

        Parameters
        ---------
        file_lines: list
            List if strings of each line in the output.
        coords_file: str
            location of the coordinate file. If not supplied, TCParse will try to
            use the location specified in the TeraChem output.
        data_output_file: str
            final json file location to write the parsed data to.

        Returns
        -------
        data: TCJobData
            Parsed data from the TeraChem output.
    '''
    parser = TCParser()
    data = parser._parse_from_list(file_lines, coords_file, data_output_file)
    return _dict_to_object(data)

def parse_from_file(tc_output_file: str, coords_file : str=None, data_output_file=None) -> TCJobData:
    '''
        Parse a list lines taken from a TeraChem output.

        Parameters
        ---------
        tc_output_file: str
            lcoation of the terachem output file.
        coords_file: str
            location of the coordinate file. If not supplied, TCParse will try to
            use the location specified in the TeraChem output.
        data_output_file: str
            final json file location to write the parsed data to.

        Returns
        -------
        data: TCJobData
            Parsed data from the TeraChem output.
    '''
    parser = TCParser()
    data = parser._parse_from_file(tc_output_file, coords_file, data_output_file)
    return _dict_to_object(data)

def _dict_to_object(data: dict) -> TCJobData:
    if isinstance(data, list):
        new_obj = []
        n_calcs = len(data)
        for i, x in enumerate(data):
            try:
                validated = TCJobData.model_validate(x)
                new_obj.append(validated)
            except ValidationError as e:
                print(f'ValidationError: Could not valid calculation {i + 1}/{n_calcs}')
    else:
        new_obj = _validate_dict(data)

    return new_obj

def _validate_dict(data: dict) -> TCJobData:
    try:
        new_obj = TCJobData.model_validate(data)
    except ValidationError as e:
        for error in e.errors():
            if error['type'] == 'missing' and error['msg'].lower() == 'field required':
                print(f"Error: Missing key: {'.'.join(map(str, error['loc']))}")
        print()
        raise e
        
    return new_obj

                
class TCParser():
    def __init__(self, resp_version=2, print_console=False) -> None:
        self._tc_output_file_path = None
        self._coords = None

        self._atoms = None
        self._n_atoms = None
        self._current_frame_num = 0

        self._job_type = 'energy'
        self._ex_type = None
        self._file = None

        self._vels_univ = None
        self._coords_univ = None

        self._data = {}
        self._add_frame_to_data()
        self._resp_version = resp_version

        self._n_singlets = 0

        if print_console:
            self._print = print
        else:
            self._print = lambda *args, **kwargs: None

        self._cis_parser = CISParser(self)
        self._cas_parser = CASParser(self)

    @property
    def _current_frame(self):
        return self._data[self._current_frame_num]

    def __del__(self):
        if self._file:
            self._file.close()

    def _next(self, num):
        for i in range(num):
            line = next(self._file)
        return line

    def _parse_charge_info(self, previous_line):
        data = self._current_frame

        chg_line = previous_line
        sp = chg_line.split()
        if "TrESP Charges" in chg_line:
            state = int(sp[4])
            key = f"esp_analysis_S0_S{state}"
        elif "TrVecESP" in chg_line:
            state1 = int(sp[4])
            state2 = int(sp[6])
            key = f"esp_analysis_S{state1}_S{state2}"
        elif "Excited State" in chg_line:
            state = int(sp[5])
            key = f"esp_analysis_S{state}"
            if key in data:
                key += '_relaxed'
        else:
            state = 0
            key = f"esp_analysis_S{state}"

        #   the description of the charge section
        #   has changed across TeraChem versions
        while "atom   " not in chg_line.lower():
            chg_line = next(self._file)
        next(self._file)

        n_atoms = self._n_atoms
        coords = np.zeros((n_atoms, 3))
        esp_charges = np.zeros(n_atoms)
        resp_charges = np.zeros(n_atoms)
        for n in range(n_atoms):
            line = next(self._file)
            sp = line.split()
            x, y, z, q, ex = [float(x) for x in sp[1:]]
            esp_charges[n] = q

        #   skip ahead to next charge section
        next(self._file)
        while "atom   " not in line.lower():
            line = next(self._file)
        next(self._file)
        for n in range(n_atoms):
            line = next(self._file)
            sp = line.split()
            x, y, z, q, ex = [float(x) for x in sp[1:]]
            coords[n] = [x, y, z]
            resp_charges[n] = q
        
        esp_dipole_au = np.sum(esp_charges[:, None] * coords, axis=0)
        resp_dipole_au = np.sum(resp_charges[:, None] * coords, axis=0)

        data[key] = {
            'esp_charges': esp_charges.tolist(),
            'resp_charges': resp_charges.tolist(),
            'esp_dipole': esp_dipole_au.tolist(),
            'resp_dipole': resp_dipole_au.tolist()
        }

    def _parse_columned_section(self, n_skip: int, n_lines: int, columns: int, key: Optional[str] = None, conv_factor=1.0):
        for i in range(n_skip): 
            next(self._file)

        all_vals = []
        for i in range(n_lines):
            line = next(self._file)
            sp = line.split()
            if len(sp) == 0:
                break
            if len(columns) == 1:
                vals = float(sp[columns[0]])*conv_factor
            else:
                vals = [float(sp[n])*conv_factor for n in columns]

            if key is not None:
                self._current_frame.setdefault(key, []).append(vals)
            all_vals.append(vals)

        return all_vals

    def _parse_gs_dipole_deriv(self):
        dMuX = self._parse_columned_section(3, self._n_atoms, [1,2,3])
        dMuY = self._parse_columned_section(3, self._n_atoms, [1,2,3])
        dMuZ = self._parse_columned_section(3, self._n_atoms, [1,2,3])
        self._current_frame['dipole_deriv'] = [dMuX, dMuY, dMuZ]

    def _parse_gs_gradient(self):
        vals = self._parse_columned_section(3, self._n_atoms, [1,2,3])
        self._current_frame['gs_gradient'] = vals

    def _add_frame_to_data(self):
        if self._current_frame_num not in self._data:
            self._data[self._current_frame_num] = {
                    'atoms': self._atoms,
                    'geom': None,
                    'energy': []
                }

    def _parse_data(self, end_phrase=None):
        self._add_frame_to_data()

        prev_line = ''
        for line in self._file:
            if end_phrase is not None:
                if end_phrase in line:
                    self._post_process_data()
                    if self._current_frame_num % 100 == 0:
                        self._print("Parsing frame ", self._current_frame_num)
                    self._current_frame_num += 1
                    self._add_frame_to_data()
        
            if 'Current Geometry' in line:
                self._parse_geom(line)

            if 'FINAL ENERGY' in line:
                self._parse_gs_energy(line)

            elif 'DIPOLE MOMENT' in line:
                self._parse_gs_dipole(line)

            elif 'CIS Parameters' in line or '(RPA) Parameters' in line:
                self._ex_type = 'cis'
                self._cis_parser.parse(line)

            elif 'CAS Parameters' in line:
                self._ex_type = 'cas'
                self._cas_parser.parse()

            elif 'Gradient units are Hartree/Bohr' in line:
                self._parse_gradient()
            
            elif 'Computing ESP on molecular grid' in line or 'Running Resp charge analysis...' in line:
                self._parse_charge_info(prev_line)

            elif 'Dipole X' in line or 'Dipole Derivative X' in line:
                self._parse_dipole_deriv_numerical(line)

            elif 'Ground state dipole moment derivatives' in line:
                self._parse_gs_dipole_deriv()

            elif 'Total Ground State Gradient' in line:
                self._parse_gs_gradient()

            prev_line = line

        self._post_process_data()

        return self._current_frame
    

    def _post_process_data(self):
        data = self._current_frame

        if len(data) == 0:
            return
        _reorder_charge_info(data)

        #   reorder keys in same order from TCJobData
        #   and remove empty or None values
        ordered_data = {}
        for key in TCJobData.model_fields.keys():
            val = data.get(key, None)
            if val is None:
                continue
            elif isinstance(val, list):
                if not any(val):
                    continue
                elif len(val) == 0:
                    continue

            data.pop(key)
            ordered_data[key] = val

        self._data[self._current_frame_num] = ordered_data.copy()

    def _parse_dipole_deriv_numerical(self, current_line):
        if self._ex_type == 'cis':
            self._cis_parser._parse_cis_dipole_deriv_numerical(current_line)
            # self._parse_cis_dipole_deriv_numerical(current_line)
        elif self._ex_type == 'cas':
            self._cas_parser._parse_cas_dipole_deriv_numerical(current_line)
            # self._parse_cas_dipole_deriv_numerical(current_line)
        else:
            raise ValueError(f'Unknown excited state type: {self._ex_type}')



    def _parse_gs_energy(self, line):
        self._current_frame['energy'].append(float(line.split()[2]))
    
    def _parse_gradient(self):
        next(self._file)
        next(self._file)
        grad = []
        for i in range(self._n_atoms):
            sp = next(self._file).split()
            grad.append([float(x) for x in sp])
        self._current_frame['gradient'] = grad

    def _parse_gs_dipole(self, line):
        #   as of now, we don't parse dipoles with point charges
        if line.split()[0] in ['MM', 'TOT']:
            return
        line = line.replace('QM', '')

        #   replace brackets so we can split by white space
        for x in ['{', '}', ',', '(', ')']:
            line = line.replace(x, ' ')
        dip = [float(x) for x in line.split()[2:5]]
        mag = np.linalg.norm(dip)
        self._current_frame['dipole_moment'] = mag
        self._current_frame['dipole_vector'] = dip
            
    def _parse_geom(self, line):
        #    overwrite geometry with these coordinates instead
        next(self._file)
        coords = []
        for n in range(self._n_atoms):
            sp = next(self._file).split()
            coords.append([float(x) for x in sp[1:4]])
        self._current_frame['geom'] = coords


    def _parse_from_list(self, file_lines: list[str], coords_file: str=None, data_output_file: str=None) -> dict:
        '''
            Parse a list lines taken from a TeraChem output.

            Parameters
            ---------
            file_lines: list
                List if strings of each line in the output.
            coords_file: str
                location of the coordinate file. If not supplied, TCParse will try to
                use the location specified in the TeraChem output.
            data_output_file: str
                final json file location to write the parsed data to.
        '''

        #   create temporary file
        ram_file = io.StringIO()
        for line in file_lines:
            if len(line) == 0:
                line = '\n'
            elif line[-1] != '\n':
                line += '\n'
            ram_file.write(line)
        ram_file.seek(0)
        self._file = ram_file
        return self._parse_all(coords_file, data_output_file)
    
    def _parse_from_file(self, tc_output_file: str, coords_file : str=None, data_output_file=None):
        '''
            Parse a list lines taken from a TeraChem output.

            Parameters
            ---------
            tc_output_file: str
                lcoation of the terachem output file.
            coords_file: str
                location of the coordinate file. If not supplied, TCParse will try to
                use the location specified in the TeraChem output.
            data_output_file: str
                final json file location to write the parsed data to.
        '''
        self._tc_output_file_path = os.path.split(tc_output_file)[0]
        if not os.path.isfile(tc_output_file):
            print(f"Could not find TeraChem output file: {tc_output_file}")
            sys.exit(1)

        self._file = open(tc_output_file)
        return self._parse_all(coords_file, data_output_file)

    def _parse_coords(self, coords_file):
        if not os.path.isfile(coords_file):
            print('Warning: Could not find coordinate file ', coords_file)
        else:
            coords_univ = XYZFile(coords_file)
            self._atoms = coords_univ.get_atoms()
            coords = coords_univ.get_coords()
            self._current_frame['geom'] = coords
            self._current_frame['atoms'] = self._atoms 

    def _parse_all(self, coords_file=None, data_output_file=None):

        is_md = False
        is_num_deriv = False
        coords_univ = None
        vels_file = None
        n_atoms_job = None
        scr_dir = None

        if coords_file is not None:
            self._parse_coords(coords_file)

        for line in self._file:
            if 'Total atoms:' in line:
                n_atoms_job = int(line.split()[2])
                if self._n_atoms is not None and self._n_atoms != n_atoms_job:
                    raise ValueError('Number of supplied atoms does not equal the amount specified in job output')
                self._n_atoms = n_atoms_job


            elif 'XYZ coordinates' in line and coords_file is None:
                if self._coords is not None:
                    self._current_frame['geom'] = self._coords
                elif self._tc_output_file_path is not None:
                    coords_file = os.path.join(self._tc_output_file_path, line.split()[2])
                    self._parse_coords(coords_file)
                else:
                    pass
                    # print("Could not import geometry")
            
            elif 'Velocities file:' in line:
                vels_file = os.path.join(self._tc_output_file_path, line.split()[2])
                if not os.path.isfile(vels_file):
                    self._print("Could not find ", vels_file)
                    self._print("    Velocities will not be imported")
                else:
                    vels_univ = XYZFile(vels_file)
                    self._atoms = vels_univ.get_atoms()
                    vels = vels_univ.get_coords()
                    self._current_frame['velocities'] = vels
                
            elif 'Scratch directory:' in line:
                if self._tc_output_file_path is not None:
                    scr_dir = os.path.join(self._tc_output_file_path, line.split()[2])


            if 'RUNNING AB INITIO MOLECULAR DYNAMICS' in line:
                is_md = True
            elif 'NUMERICAL DIPOLE' in line or 'NUMERICAL NONADIABATIC' in line:
                self._current_frame_num = -1
                self._data[-1] = {}
                is_num_deriv = True


            if n_atoms_job is not None:
                #   MD trajectories have multiple parses
                if is_md:
                    self._parse_data(end_phrase="MD STEP")
  
                elif is_num_deriv:
                    self._print("Parsing Numerical Derivative job")
                    self._parse_data(end_phrase="Current Geometry")

                #   single jobs need to be parsed only once
                else:
                    self._parse_data()
                    

        ###########   tc.out file parsing done   ###########

        #   frame number -1 is only used so that "Current Geometry" section
        #   can increase the frame number THEN start reading data
        if -1 in self._data:
            self._data.pop(-1)

        #   update coordinate and velocity information from traj files
        if is_md:
            if scr_dir is not None:
                coords_file = os.path.join(scr_dir, 'coors.xyz')
                coords_univ = XYZFile(coords_file)
                vels_file = os.path.join(scr_dir, 'velocities.xyz')
                vels_univ = XYZFile(vels_file)

                #   Match XYZ energies to support both initial-frame and propagated-frame numbering.
                frame_offset = 0
                best_error = float('inf')
                for offset in (0, 1):
                    energy_error = 0.0
                    for n in range(coords_univ.n_frames):
                        frame = coords_univ.get_frame_num_from_comment(n) + offset
                        data = self._data.get(frame)
                        if data is None or not data.get('energy'):
                            energy_error = float('inf')
                            break

                        comment = coords_univ.get_comment(n)
                        xyz_energy = float(comment.split()[0])
                        energy_error += abs(data['energy'][0] - xyz_energy)

                    if energy_error < best_error:
                        frame_offset = offset
                        best_error = energy_error

                for n in range(coords_univ.n_frames):
                    frame = coords_univ.get_frame_num_from_comment(n)
                    self._data[frame + frame_offset]['geom'] = coords_univ.get_coords(n)

                for n in range(vels_univ.n_frames):
                    frame = vels_univ.get_frame_num_from_comment(n)
                    self._data[frame + frame_offset]['velocities'] = vels_univ.get_coords(n)
                for frame in self._data:
                    if 'velocities' not in self._data[frame]:
                        self._data[frame]['velocities'] = None

            else:
                self._print("Scratch directory not found: cannot import coordinates and velocities")

        if is_md or is_num_deriv:
            return_data = [x for x in self._data.values()]
        else:
            return_data= self._data[0]
            # reorder_charge_info(self._data)

        # reorder_charge_info(self._data)
        # raise ValueError()


        if data_output_file is not None:
            self._print("Writing data to ", data_output_file)
            with open(data_output_file, 'w') as file:
                json.dump(return_data, file, indent=4)

        return return_data

