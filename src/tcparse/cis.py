from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .tcparse import TCParser

import numpy as np
from .units import *

def delegate_property(attr):
    return property(lambda self: getattr(self.root, attr))

class CISParser:
    
    _file =                     delegate_property('_file')
    _current_frame =            delegate_property('_current_frame')
    _n_atoms =                  delegate_property('_n_atoms')
    _data =                     delegate_property('_data')
    _parse_columned_section =   delegate_property('_parse_columned_section')
    _next =                     delegate_property('_next')

    def __init__(self, root_parser: TCParser):
        self.root = root_parser
    
    def _init_cis_data(self, current_line: str):

        if 'Number of roots' in current_line:
            n_states = int(current_line.split()[-1])
            self._n_cis_states = n_states

        n = self._n_cis_states
        n_combos = int(n*(n+1)/2)

        data = self._current_frame
        data['cis_states'] =                     n_states
        data['cis_excitations'] =                [None]*n
        data['cis_osc_strengths'] =              [None]*n
        data['cis_gradients'] =                  [None]*n
        data['cis_couplings'] =                  [None]*n_combos
        data['cis_transition_dipole_deriv'] =    [None]*n_combos
        data['cis_transition_dipoles'] =         [None]*n_combos
        data['cis_relaxed_transition_dipoles'] = [None]*n
        data['cis_unrelaxed_dipoles'] =          [None]*n
        data['cis_relaxed_dipoles'] =            [None]*n
        data['cis_unrelaxed_dipole_deriv'] =     [None]*n
        data['cis_relaxed_dipole_deriv'] =       [None]*n
        data['cis_transition_dots'] =            [None]*n

    def parse(self):
        '''
            Main parsing function. Should be called from the root parser.
        '''
        self._ex_type = 'cis'

        for line in self._file:
            if 'Number of roots' in line:
                self._init_cis_data(line)
            
            elif 'Excited State Gradient' in line:
                self._parse_cis_gradients(line)

            elif 'moment derivatives' in line:
                self._parse_cis_dipole_deriv_analytical(line)
            
            elif 'Total Energy (a.u.)   Ex. Energy (a.u.)     Ex. Energy (eV)' in line:
                self._parse_cis_energy(line)

            elif 'excited state dipole moments' in line:
                self._parse_cis_dipoles(line)

            elif 'transition dipole moments' in line.lower():
                self._parse_cis_transition_dipoles(line)

            elif 'Nonadiabatic coupling between' in line:
                self._parse_cis_couplings(line)

            elif 'Largest CI coefficients:' in line:
                self._parse_cis_coefficients(line)

            elif 'Computing ESP on molecular grid' in line or 'Running Resp charge analysis...' in line:
                self.root._parse_charge_info(prev_line)

            elif 'Final Excited State Results' in line:
                self._parse_final_cis_results(line)
                #   this is the end of the CIS section, so leave the function
                return
            
            prev_line = line

    def _parse_cis_couplings(self, current_line: str):
        current_line = current_line.lower()
        if 'ground' in current_line:
            sp = current_line.split()
            state_1 = 0
            state_2 = int(sp[-2])
            vals = self._parse_columned_section(4, self._n_atoms, [1,2,3])
            index = self._cis_state_pair_to_index(state_1, state_2)
            self._current_frame['cis_couplings'][index] = vals

        else:    
            n_states = self._current_frame['cis_states']
            for i in range(1, n_states + 1):
                for j in range(i+1, n_states + 1):
                    if f'excited state {i}' in current_line and f'excited state {j}' in current_line:
                        idx = self._cis_state_pair_to_index(i, j)
                        vals = self._parse_columned_section(4, self._n_atoms, [1,2,3])
                        self._current_frame['cis_couplings'][idx] = vals

    def _parse_cis_coefficients(self, current_line: str):
        root = int(current_line.split()[1][0:-1])
        coefficients = []
        for line in self._file:
            coeff = []
            while len(line.split()) > 0:
                sp = line.split()
                orb1 = int(sp[0])
                orb2 = int(sp[2])
                c = float(sp[8])
                coeff.append([orb1, orb2, c])
                line = next(self._file)
            coefficients.append(coeff)
            if root == self._current_frame['cis_states']:
                self._current_frame['cis_excitations'] = coefficients
                return

            line = next(self._file)
            root = int(line.split()[1][0:-1])

    def _parse_cis_transition_dipoles(self, current_line: str):
        n_states = self._n_cis_states

        #   Ground state transition dipoles
        if 'Transition dipole moments:' in current_line:
            vals = self._parse_columned_section(3, n_states, [1,2,3])
            for i in range(n_states):
                self._current_frame['cis_transition_dipoles'][i] = vals[i]


        elif 'Transition dipole moments between' in current_line:
            n_lines = int(n_states*(n_states - 1)/2)
            vals = self._parse_columned_section(3, n_lines, [3, 4, 5])
            vals = list(reversed(vals))
            for i in range(1, n_states + 1):
                for j in range(i+1, n_states + 1):
                    index = self._cis_state_pair_to_index(i, j)
                    self._current_frame['cis_transition_dipoles'][index] = vals.pop()

        elif 'Relaxed transition dipole moments' in current_line:
            #   Note: Relaxed transition moments are not computed between excited states
            vals = self._parse_columned_section(3, n_states, [1,2,3])
            for i in range(n_states):
                self._current_frame['cis_relaxed_transition_dipoles'][i] = vals[i]

    def _parse_cis_dipoles(self, line):
        vals = self._parse_columned_section(3, self._n_cis_states, [0, 1,2,3])
        for i, mu_x, mu_y, mu_z in vals:
            i = int(i) - 1
            mu = np.array([mu_x, mu_y, mu_z])*DEBYE_2_AU
            if 'Relaxed' in line:
                self._current_frame['cis_relaxed_dipoles'][i] =  list(mu)
            else:
                self._current_frame['cis_unrelaxed_dipoles'][i] = list(mu)

    def _parse_cis_gradients(self, current_line: str):
        root = int(current_line.split()[1].replace(':', ''))
        self._current_frame['cis_gradients'][root-1] = self._parse_columned_section(3, self._n_atoms, [1,2,3])

    def _parse_cis_energy(self, line):
        self._parse_columned_section(1, self._n_cis_states, [1], 'energy')


    def _cis_state_pair_to_index(self, i, j):
        if i > j:
            i, j = j, i
        n_states = self._n_cis_states
        return int((i*n_states) + (j - 1 - (i*(i+1)/2)))


    def _parse_cis_dipole_deriv_analytical(self, current_line: str):
        data = self._current_frame
        current_line = current_line.lower()

        if 'ground' in current_line:
            key = 'dipole_deriv'
            self._parse_columned_section(3, self._n_atoms, [1,2,3], key)
            self._parse_columned_section(3, self._n_atoms, [1,2,3], key)
            self._parse_columned_section(3, self._n_atoms, [1,2,3], key)
            return

        elif 'ground' not in current_line and 'transition' not in current_line:
            current_line = self._next(2)
            while current_line.split()[1].isnumeric():
                sp = current_line.split()
                state = int(sp[1])
                dMuX = self._parse_columned_section(3, self._n_atoms, [1,2,3])
                dMuY = self._parse_columned_section(3, self._n_atoms, [1,2,3])
                dMuZ = self._parse_columned_section(3, self._n_atoms, [1,2,3])
                data['cis_unrelaxed_dipole_deriv'][state - 1] = [dMuX, dMuY, dMuZ]
                current_line = self._next(2).lower()


        if 'transition dipole moment derivative' in current_line:
            current_line = self._next(2)
            sp = current_line.split()
            state_1 = int(sp[0])
            state_2 = int(sp[2])
            if state_1 == 0:
                while current_line.split()[0].isnumeric():
                    sp = current_line.split()
                    state_2 = int(sp[2])
                    dMuX = self._parse_columned_section(3, self._n_atoms, [1,2,3])
                    dMuY = self._parse_columned_section(3, self._n_atoms, [1,2,3])
                    dMuZ = self._parse_columned_section(3, self._n_atoms, [1,2,3])
                    idx = self._cis_state_pair_to_index(state_1, state_2)
                    data['cis_transition_dipole_deriv'][idx] = [dMuX, dMuY, dMuZ]
                    current_line = self._next(2)
            else:
                    dMuX = self._parse_columned_section(3, self._n_atoms, [1,2,3])
                    dMuY = self._parse_columned_section(3, self._n_atoms, [1,2,3])
                    dMuZ = self._parse_columned_section(3, self._n_atoms, [1,2,3])
                    idx = self._cis_state_pair_to_index(state_1, state_2)
                    data['cis_transition_dipole_deriv'][idx] = [dMuX, dMuY, dMuZ]

    def _parse_final_cis_results(self, line):
        n_states = self._current_frame['cis_states']
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
        self._current_frame['cis_ex_energies'] = ex_energies
        self._current_frame['cis_osc_strengths'] = oscillators


    def _parse_cis_dipole_deriv_numerical(self, current_line):

        #   analytical derivatives
        columns = [0,1,2]
        skips = [4,7,7]

        #   figure out what derivatives these are
        for x in [':', 'Root', 'State', '->', 'Singlet', 'Doublet', 'Tripplet', '-']:
            current_line = current_line.replace(x, '')
        current_line = current_line.lower()

        dipole_type_line = current_line
        
        dMuX = self._parse_columned_section(skips[0], self._n_atoms, columns)
        dMuY = self._parse_columned_section(skips[1], self._n_atoms, columns)
        dMuZ = self._parse_columned_section(skips[2], self._n_atoms, columns)

        sp = dipole_type_line.split()
        if 'ground' in dipole_type_line or '0 Dipole' in dipole_type_line:
            self._current_frame['dipole_deriv'] = [dMuX, dMuY, dMuZ]
        elif 'unrelaxed' in dipole_type_line:
            state = int(sp[0])
            self._current_frame['cis_unrelaxed_dipole_deriv'][state - 1] = [dMuX, dMuY, dMuZ]
        elif 'relaxed' in dipole_type_line:
            state = int(sp[0])
            self._current_frame['cis_relaxed_dipole_deriv'][state - 1] = [dMuX, dMuY, dMuZ]
        elif 'transition dipole' in dipole_type_line:
            state_1 = int(sp[0])
            state_2 = int(sp[1])
            index = self._cis_state_pair_to_index(state_1, state_2)
            self._current_frame['cis_transition_dipole_deriv'][index] = [dMuX, dMuY, dMuZ]
        else:
            self._current_frame['cis_unrelaxed_dipole_deriv'][state - 1] = [dMuX, dMuY, dMuZ]