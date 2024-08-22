import os
import json
import numpy as np
import sys
import io
import argparse

from pydantic import BaseModel, ConfigDict, ValidationError
from typing import Optional

DEBYE_2_AU = 0.3934303
ANG_2_BOHR = 1.8897259886
BOHR_2_ANG = 1.0/ANG_2_BOHR


class TCJobData(BaseModel):
    atoms                         : list
    energy                        : list
    geom                          : list

    cas_dipole_deriv              : Optional[list] = None
    cas_dipoles                   : Optional[list] = None
    cas_states                    : Optional[int] = None
    cas_transition_dipole_deriv   : Optional[list] = None
    cas_transition_dipoles        : Optional[list] = None
    cis_dipole_deriv              : Optional[list] = None
    cis_ex_energies               : Optional[list] = None
    cis_excitations               : Optional[list] = None
    cis_gradient_2                : Optional[list] = None
    cis_osc_strengths             : Optional[list] = None
    cis_relaxed_dipole_deriv      : Optional[list] = None
    cis_relaxed_dipoles           : Optional[list] = None
    cis_relaxed_esp_charges       : Optional[list] = None
    cis_relaxed_esp_dipoles       : Optional[list] = None
    cis_relaxed_resp_charges      : Optional[list] = None
    cis_relaxed_resp_dipoles      : Optional[list] = None
    cis_states                    : Optional[int] = None
    cis_tr_esp_charges            : Optional[list] = None
    cis_tr_esp_dipoles            : Optional[list] = None
    cis_tr_resp_charges           : Optional[list] = None
    cis_tr_resp_dipoles           : Optional[list] = None
    cis_transition_dipole_deriv   : Optional[list] = None
    cis_transition_dipoles        : Optional[list] = None
    cis_unrelaxed_dipole_deriv    : Optional[list] = None
    cis_unrelaxed_dipoles         : Optional[list] = None
    cis_unrelaxed_esp_charges     : Optional[list] = None
    cis_unrelaxed_esp_dipoles     : Optional[list] = None
    cis_unrelaxed_resp_charges    : Optional[list] = None
    cis_unrelaxed_resp_dipoles    : Optional[list] = None
    dipole_deriv                  : Optional[list] = None
    dipole_moment                 : Optional[float] = None
    dipole_vector                 : Optional[list] = None
    esp_charges                   : Optional[list] = None
    esp_dipoles                   : Optional[list] = None
    gradient                      : Optional[list] = None
    resp_charges                  : Optional[list] = None
    resp_dipoles                  : Optional[list] = None
    velocities                    : Optional[None] = None
    
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

def reorder_charge_info(data: dict):
    energies = data['energy']
    n_states = len(energies)
    new_esp_data = {}

    exc_type = 'cis'
    if 'cas_states' in data:
        exc_type = 'cas'

    # all_esp_analysis_keys = [k for k in data if 'esp_analysis_' in k]
    if 'esp_analysis_S0' in data:
        _assign_charge_info(new_esp_data, data['esp_analysis_S0'], '')
        data.pop('esp_analysis_S0')
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
        # print("    ", dp, closest_idx)

        #dp = np.dot(gs_dipole, cis_dipole)
        if dp > 0:
            signs[i+1] = 1.0
        else:
            signs[i+1] = -1.0

        data['cis_transition_dots'].append(dots)

    # print(signs)

    count = 0
    for i in range(0, n_states):
        for j in range(i+1, n_states):
            esp_key = f'esp_analysis_S{i}_S{j}'
            if esp_key in data:
                # print("WORKING ON ", esp_key, signs[i], signs[j])
                for prop_key in ['esp_charges', 'resp_charges', 'esp_dipole', 'resp_dipole']:
                    prop = np.array(data[esp_key][prop_key])
                    data[esp_key][prop_key] = (prop*signs[i]*signs[j]).tolist()
            
            tr_dipole = data['cis_transition_dipoles'][count]
            data['cis_transition_dipoles'][count] = (np.array(tr_dipole)*signs[i]*signs[j]).tolist()
            count += 1
    
class TCParser():
    def __init__(self, resp_version=2) -> None:
        self._tc_output_file_path = None
        self._coords = None

        self._atoms = None
        self._n_atoms = None
        self._current_frame_num = 0

        self._job_type = 'energy'

        self._vels_univ = None
        self._coords_univ = None

        self._data = {}
        self._add_frame_to_data()
        self._resp_version = resp_version

    @property
    def _current_frame(self):
        return self._data[self._current_frame_num]

    def __del__(self):
        self._file.close()

    def _next(self, num):
        for i in range(num):
            line = next(self._file)
        return line

    def _parse_charge_info(self, data, n_atoms, previous_line):

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

        for n in range(5):
            next(self._file)

        coords = np.zeros((n_atoms, 3))
        esp_charges = np.zeros(n_atoms)
        resp_charges = np.zeros(n_atoms)
        for n in range(n_atoms):
            line = next(self._file)
            sp = line.split()
            x, y, z, q, ex = [float(x) for x in sp[1:]]
            esp_charges[n] = q

        #   skip ahead ot next charge section
        next(self._file)
        while "atom" not in line.lower():
            line = next(self._file)
        next(self._file)
        for n in range(n_atoms):
            line = next(self._file)
            sp = line.split()
            x, y, z, q, ex = [float(x) for x in sp[1:]]
            coords[n] = [x, y, z]
            resp_charges[n] = q
        
        esp_dipole_au = np.sum(esp_charges[:, None] * coords, axis=0)
        esp_dipole_debye = np.sum(esp_charges[:, None] * coords, axis=0)/0.3934303
        resp_dipole_au = np.sum(resp_charges[:, None] * coords, axis=0)
        resp_dipole_debye = np.sum(esp_charges[:, None] * coords, axis=0)/0.3934303
        # print(key)
        # print('    ESP Dipole (a.u.):   {:8.5f}  {:8.5f}  {:8.5f}    {:8.5f}'.format(*esp_dipole_au, np.linalg.norm(esp_dipole_au)))
        # print('    ESP Dipole (Debye):  {:8.5f}  {:8.5f}  {:8.5f}    {:8.5f}'.format(*esp_dipole_debye, np.linalg.norm(esp_dipole_debye)))
        # print('    RESP Dipole (a.u.):  {:8.5f}  {:8.5f}  {:8.5f}    {:8.5f}'.format(*resp_dipole_au, np.linalg.norm(resp_dipole_au)))
        # print('    RESP Dipole (Debye): {:8.5f}  {:8.5f}  {:8.5f}    {:8.5f}'.format(*resp_dipole_debye, np.linalg.norm(resp_dipole_debye)))

        data[key] = {
            'esp_charges': esp_charges.tolist(),
            'resp_charges': resp_charges.tolist(),
            'esp_dipole': esp_dipole_au.tolist(),
            'resp_dipole': resp_dipole_au.tolist()
        }

    def _parse_columned_section(self, data: dict, n_skip: int, n_lines: int, columns: int, key: str, conv_factor=1.0):
        for i in range(n_skip): 
            next(self._file)
        for i in range(n_lines):
            line = next(self._file)
            sp = line.split()
            if len(sp) == 0:
                break
            if len(columns) == 1:
                vals = float(sp[columns[0]])*conv_factor
            else:
                vals = [float(sp[n])*conv_factor for n in columns]
            data.setdefault(key, []).append(vals)

    def _parse_cas_section(self, n_atoms, data: dict):
        n_states = 0
        coefficients = []
        for line in self._file:
            # if '' in line:
            #     return
            
            if 'Total Energy (a.u.)   Ex. Energy (a.u.)     Ex. Energy (eV)' in line:
                line = next(self._file)
                line = next(self._file)
                data['energy'] = []
                while len(line.split()) > 0:
                    n_states += 1
                    energy = float(line.split()[2])
                    data['energy'].append(energy)
                    line = next(self._file)
                data['cas_states'] = n_states

            elif 'Singlet state dipole moments' in line:
                self._parse_columned_section(data, 3, n_states, [1, 2, 3], 'cas_dipoles', DEBYE_2_AU)

            elif 'Singlet state electronic transitions:' in line:
                self._parse_columned_section(data, 3, n_states**2, [3,4,5], 'cas_transition_dipoles')

            elif 'Running Resp charge analysis...' in line:
                self._parse_charge_info(data, n_atoms, prev_line)

            elif 'Singlet state electric transition quadrupole moments' in line:
                #   this is the last CAS section that we can read
                #   read past this part, we don't save the info (yet)
                line = next(self._file)
                line = next(self._file)
                while len(line.split()) > 0:
                    line = next(self._file)
                line = next(self._file)

                #   check to see if the overlaps are printed, and save them if so
                if len(line.split()) > 0:
                    if not line.split()[0].isnumeric():
                        #   if it's not a number, then this is not an overlap
                        return
                    #   first it prints the raw wavefunction overlaps
                    for i in range(n_states):
                        line = next(self._file)
                    #   then it prints the sign corercted overlaps
                    overlaps = []
                    for i in range(n_states):
                        line = next(self._file)
                        overlaps.append([float(x) for x in line.split()])
                    data['ci_overlap'] = overlaps

                #   leave the function
                return
            
            prev_line = line

    def _entry_cis_gradient(self, line):
        root = int(line.split()[1].replace(':', ''))
        for i in range(3): next(self._file)
        grad = np.zeros((self._n_atoms, 3))
        for i in range(self._n_atoms):
            line = next(self._file)
            grad[i] = np.array([float(x) for x in line.split()[1:]])

    def _parse_cis_section(self, data: dict):
        n_states = 0
        coefficients = []
        for line in self._file:
            # if '' in line:
            #     return

            if 'Number of roots' in line:
                n_states = int(line.split()[-1])
                data['cis_states'] = n_states
            
            elif 'Excited State Gradient' in line:
                root = int(line.split()[1].replace(':', ''))
                self._parse_columned_section(data, 3, self._n_atoms, [1,2,3], f'cis_gradient_{root}')

            elif 'moment derivatives' in line:
                self._parse_dipole_deriv_analytical(data, line)
            
            elif 'Total Energy (a.u.)   Ex. Energy (a.u.)     Ex. Energy (eV)' in line:
                self._parse_columned_section(data, 1, n_states, [1], 'energy')

            elif 'Unrelaxed excited state dipole moments' in line:
                self._parse_columned_section(data, 3, n_states, [1,2,3], 'cis_unrelaxed_dipoles', DEBYE_2_AU)
                
            elif 'Relaxed excited state dipole moments:' in line:
                self._parse_columned_section(data, 3, n_states, [1,2,3], 'cis_relaxed_dipoles')

            elif 'Transition dipole moments:' in line:
                self._parse_columned_section(data, 3, n_states, [1, 2, 3], 'cis_transition_dipoles')       
                
            elif 'Transition dipole moments between excited states' in line:
                n_lines = int(n_states*(n_states - 1)/2)
                self._parse_columned_section(data, 3, n_lines, [3,4,5], 'cis_transition_dipoles')

            elif 'Largest CI coefficients:' in line:
                root = int(line.split()[1][0:-1])
                coeff = []
                line = next(self._file)
                while len(line.split()) > 0:
                    sp = line.split()
                    orb1 = int(sp[0])
                    orb2 = int(sp[2])
                    c = float(sp[8])
                    coeff.append([orb1, orb2, c])
                    line = next(self._file)
                coefficients.append(coeff)
                if root == n_states - 1:
                    data['cis_excitations'] = coefficients
            
            elif 'Running Resp charge analysis...' in line:
                self._parse_charge_info(data, self._n_atoms, prev_line)

            elif 'Final Excited State Results' in line:
                for i in range(3): next(self._file)
                excitations = []
                ex_energies = []
                oscillators = []
                for i in range(n_states):
                    sp = next(self._file).split()
                    excitations.append([int(sp[6]), int(sp[8])])
                    ex_energies.append(float(sp[2]))
                    oscillators.append(float(sp[3]))
                # data['cis_excitations'] = excitations
                data['cis_ex_energies'] = ex_energies
                data['cis_osc_strengths'] = oscillators
            
                #   this is the end of the CIS section, so leave the function
                return
            
            prev_line = line

    def _add_frame_to_data(self):
        if self._current_frame_num not in self._data:
            self._data[self._current_frame_num] = {
                    'atoms': self._atoms,
                    'geom': None,
                    'energy': []
                }
            
    def _parse_data(self, list=None, end_phrase=None):

        n_atoms = self._n_atoms

        self._add_frame_to_data()
        data = self._current_frame


        prev_line = ''
        for line in self._file:

            if end_phrase is not None:
                if end_phrase in line:
                    if self._current_frame_num % 100 == 0:
                        print("Parsing frame ", self._current_frame_num)
                    self._current_frame_num += 1
                    self._add_frame_to_data()
                    data = self._current_frame
        
            if 'Current Geometry' in line:
                #    overwrite geometry with these coordinates instead
                line = next(self._file)
                coords = []
                for n in range(n_atoms):
                    sp = next(self._file).split()
                    coords.append([float(x) for x in sp[1:4]])
                data['geom'] = coords

            if 'FINAL ENERGY' in line:
                data['energy'].append(float(line.split()[2]))

            elif 'DIPOLE MOMENT' in line:
                #   as of now, we don't parse dipoles with point charges
                if line.split()[0] in ['MM', 'TOT']:
                    continue
                line = line.replace('QM', '')

                #   replace brackets so we can split by white space
                for x in ['{', '}', ',', '(', ')']:
                    line = line.replace(x, ' ')
                dip = [float(x) for x in line.split()[2:5]]
                mag = np.linalg.norm(dip)
                data['dipole_moment'] = mag
                data['dipole_vector'] = dip

            elif 'CIS Parameters' in line:
                self._parse_cis_section(data)

            elif 'CAS Parameters' in line:
                self._parse_cas_section(n_atoms, data)

            elif 'Gradient units are Hartree/Bohr' in line:
                data['gradient'] = self._parse_gradient()

            elif 'Running Resp charge analysis...' in line:
                self._parse_charge_info(data, n_atoms, prev_line)

            elif 'Dipole X' in line or 'Dipole Derivative X' in line or 'moment derivatives' in line:
                self._parse_dipole_deriv_numerical(data, line)

            # elif 'Dipole moment derivatives' in line:
            #     self._parse_dipole_deriv_anal(data, line)

            prev_line = line

        self._post_process_data(data)

        return data
    
    def _post_process_data(self, data: dict):

        # print("BEFORE")
        # for key in data:
        #     print(key)

        ex_type = 'cas'
        if 'cis_states' in data:
            ex_type = 'cis'

        #   Ground State Dipole Derivatives
        if '_S0_dipole_deriv' in data:
            mat = np.array(data['_S0_dipole_deriv']).reshape((3, -1, 3))
            data['dipole_deriv'] = mat.tolist()
            data.pop('_S0_dipole_deriv')

        #   CIS Unrelaxed Dipole and CAS Excited Dipole Derivs
        n_states = 0
        while f'_S{n_states+1}_unrelaxed_dipole_deriv' in data:
            n_states += 1
            key = f'_S{n_states}_unrelaxed_dipole_deriv'
            mat = np.array(data[key]).reshape((3, -1, 3))
            data.setdefault(f'{ex_type}_unrelaxed_dipole_deriv', []).append(mat.tolist())
            data.pop(key)
        while f'_S{n_states+1}_dipole_deriv' in data:
            n_states += 1
            key = f'_S{n_states}_dipole_deriv'
            mat = np.array(data[key]).reshape((3, -1, 3))
            data.setdefault(f'{ex_type}_dipole_deriv', []).append(mat.tolist())
            data.pop(key)

        #   CIS Transition Dipole Derivatives
        key = '_S0_S1_dipole_deriv'
        if key in data:
            for i in range(0, n_states+1):
                for j in range(i+1, n_states+1):
                    key = f'_S{i}_S{j}_dipole_deriv'
                    mat = np.array(data[key]).reshape((3, -1, 3))
                    data.setdefault(f'{ex_type}_transition_dipole_deriv', []).append(mat.tolist())
                    data.pop(key)

        #   CIS Relaxed Dipole Derivatives
        for i in range(1, n_states+1):
            key = f'_S{i}_relaxed_dipole_deriv'
            if key in data:
                mat = np.array(data[key]).reshape((3, -1, 3))
                data.setdefault(f'{ex_type}_relaxed_dipole_deriv', []).append(mat.tolist())
                data.pop(key)
 
        # print("AFTER")
        # for key in data:
        #     print(key)

    def _parse_dipole_deriv_analytical(self, data, current_line):
        current_line = current_line.lower()

        if 'ground' in current_line:
            key = f'_S0_dipole_deriv'
            self._parse_columned_section(data, 3, self._n_atoms, [1,2,3], key)
            self._parse_columned_section(data, 3, self._n_atoms, [1,2,3], key)
            self._parse_columned_section(data, 3, self._n_atoms, [1,2,3], key)

        elif 'ground' not in current_line and 'transition' not in current_line:
            current_line = self._next(2)
            while current_line.split()[1].isnumeric():
                sp = current_line.split()
                state = int(sp[1])
                key = f'_S{state}_dipole_deriv'
                self._parse_columned_section(data, 3, self._n_atoms, [1,2,3], key)
                self._parse_columned_section(data, 3, self._n_atoms, [1,2,3], key)
                self._parse_columned_section(data, 3, self._n_atoms, [1,2,3], key)
                current_line = self._next(2).lower()

        if 'transition dipole' in current_line:
            current_line = self._next(2)
            while current_line.split()[0].isnumeric():
                sp = current_line.split()
                state_1 = int(sp[0])
                state_2 = int(sp[2])
                key = f'_S{state_1}_S{state_2}_dipole_deriv'
                self._parse_columned_section(data, 3, self._n_atoms, [1,2,3], key)
                self._parse_columned_section(data, 3, self._n_atoms, [1,2,3], key)
                self._parse_columned_section(data, 3, self._n_atoms, [1,2,3], key)
                current_line = self._next(2)



    def _parse_dipole_deriv_numerical(self, data, current_line):

        #   analytical derivatives
        columns = [0,1,2]
        skips = [4,7,7]
        if 'moment derivatives' in current_line:
            #   numerical derivatives
            columns = [1,2,3]
            skips = [3,3,3]

        #   figure out what derivatives these are
        for x in [':', 'Root', 'State', '->', 'Singlet', 'Doublet', 'Tripplet', '-']:
            current_line = current_line.replace(x, '')
        current_line = current_line.lower()

        sp = current_line.split()
        if 'ground' in current_line or '0 Dipole' in current_line:
            key = f'_S0_dipole_deriv'
        elif 'unrelaxed' in current_line:
            state = int(sp[0])
            key = f'_S{state}_unrelaxed_dipole_deriv'
        elif 'relaxed' in current_line:
            state = int(sp[0])
            key = f'_S{state}_relaxed_dipole_deriv'
        elif 'transition dipole' in current_line:
            state_1 = int(sp[0])
            state_2 = int(sp[1])
            key = f'_S{state_1}_S{state_2}_dipole_deriv'
        else:
            state = int(sp[0])
            key = f'_S{state}_dipole_deriv'

        self._parse_columned_section(data, skips[0], self._n_atoms, columns, key)
        self._parse_columned_section(data, skips[1], self._n_atoms, columns, key)
        self._parse_columned_section(data, skips[2], self._n_atoms, columns, key)


        
    
    def _parse_gradient(self):
        next(self._file)
        next(self._file)
        grad = []
        for i in range(self._n_atoms):
            sp = next(self._file).split()
            grad.append([float(x) for x in sp])
        return grad

    def parse_from_list(self, file_lines: list[str], coords_file: str=None, data_output_file: str=None):
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
    
    def parse_from_file(self, tc_output_file: str, coords_file : str=None, data_output_file=None):
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
        self._file = open(tc_output_file)
        return self._parse_all(coords_file, data_output_file)

    def _parse_coords(self, coords_file):
        if not os.path.isfile(coords_file):
            raise FileNotFoundError(f'Could not find coordinate file {coords_file}')
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
                    # if not os.path.isfile(coords_file):
                    #     print("Could not find ", coords_file)
                    #     print("    Coordinates will not be imported")
                    # else:
                    #     coords_univ = XYZFile(coords_file)
                    #     self._atoms = coords_univ.get_atoms()
                    #     coords = coords_univ.get_coords()
                    #     self._data[self._current_frame]['geom'] = coords
                    #     self._data[self._current_frame]['atoms'] = self._atoms 
                else:
                    pass
                    # print("Could not import geometry")
            
            elif 'Velocities file:' in line:
                vels_file = os.path.join(self._tc_output_file_path, line.split()[2])
                if not os.path.isfile(vels_file):
                    print("Could not find ", vels_file)
                    print("    Velocities will not be imported")
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
                    print("PARSING STEP")
                    self._parse_data(end_phrase="MD STEP")
  
                elif is_num_deriv:
                    print("Parsing Numerical Derivative job")
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

                for n in range(coords_univ.n_frames):
                    frame = coords_univ.get_frame_num_from_comment(n)
                    self._data[frame + 1]['geom'] = coords_univ.get_coords(n)

                for n in range(vels_univ.n_frames):
                    frame = vels_univ.get_frame_num_from_comment(n)
                    self._data[frame + 1]['velocities'] = vels_univ.get_coords(n)
                for frame in self._data:
                    if 'velocities' not in self._data[frame]:
                        self._data[frame]['velocities'] = None

            else:
                print("Scratch directory not found: cannot import coordinates and velocities")

        if is_md or is_num_deriv:
            for frame in self._data:
                reorder_charge_info(self._data[frame])
        else:
            self._data = self._data[0]
            reorder_charge_info(self._data)

        # reorder_charge_info(self._data)
        # raise ValueError()


        if data_output_file is not None:
            print("Writing data to ", data_output_file)
            with open(data_output_file, 'w') as file:
                json.dump(self._data, file, indent=4)

        return self._data



def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-t', type=str, required=True,  default=None, help='TeraChem output file')
    arg_parser.add_argument('-j', type=str, required=True,  default=None, help='.json file to save parsed output to')
    arg_parser.add_argument('-x', type=str, required=False, default=None, help='.xyz coordinate file')
    args = arg_parser.parse_args()

    parser = TCParser()
    data = parser.parse_from_file(args.t, args.x, args.j)

    new_obj = TCJobData.parse_obj(data)
    return new_obj

if __name__ == '__main__':
    job_data = main()