from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .tcparse import TCParser

import numpy as np
from .units import *

def delegate_property(attr):
    return property(lambda self: getattr(self.root, attr))

class CASParser:
    
    _file =                     delegate_property('_file')
    _current_frame =            delegate_property('_current_frame')
    _n_atoms =                  delegate_property('_n_atoms')
    _data =                     delegate_property('_data')
    _parse_columned_section =   delegate_property('_parse_columned_section')
    _next =                     delegate_property('_next')

    def __init__(self, root_parser: TCParser):
        self.root = root_parser

    def _init_cas_data(self, line):
        self._n_singlets = int(line.split()[-1])
        n = self._n_singlets
        n_combos = int(n*(n-1)/2)

        data = self._current_frame
        data['cas_singlet_states'] =              n
        data['cas_gradients'] =                  [None]*n
        data['cas_couplings'] =                  [None]*n_combos
        data['cas_transition_dipole_deriv'] =    [None]*n_combos
        data['cas_transition_dipoles'] =         [None]*n_combos
        data['cas_dipoles'] =                    [None]*n
        data['cas_dipole_deriv'] =               [None]*n


    def parse(self):

        data = self._current_frame
        n_states = 0
        for line in self._file:
            if 'Number of singlet CSFs' in line:
                self._init_cas_data(line)
            
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
                self._parse_cas_dipoles(line)

            elif 'Singlet state electronic transitions:' in line:
                self._parse_cas_transition_dipoles(line)

            elif 'Running Resp charge analysis...' in line:
                self._parse_charge_info(prev_line)

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

    def _parse_cas_transition_dipoles(self, line):
        n = self._n_singlets
        n_combos = int(n*(n-1)/2)
        vals = self._parse_columned_section(3, n_combos, [0, 2, 3, 4, 5])
        for i, j, x, y, z in vals:
            index = self._cas_state_pair_to_index(int(i-1), int(j-1))
            self._current_frame['cas_transition_dipoles'][index] = [x, y, z]

    def _parse_cas_dipoles(self, line):
        vecs = self._parse_columned_section(3, self._n_singlets, [1, 2, 3], conv_factor=DEBYE_2_AU)
        for i, vec in enumerate(vecs):
            self._current_frame['cas_dipoles'][i] = vec

    def _cas_state_pair_to_index(self, i, j):
        if i > j:
            i, j = j, i
        n_states = self._n_singlets - 1
        return int((i*n_states) + (j - 1 - (i*(i+1)/2)))
    
    def _parse_cas_dipole_deriv_numerical(self, current_line):

        #   analytical derivatives
        columns = [0,1,2]
        skips = [4,7,7]

        dipole_line = current_line

        dMuX = self._parse_columned_section(skips[0], self._n_atoms, columns)
        dMuY = self._parse_columned_section(skips[1], self._n_atoms, columns)
        dMuZ = self._parse_columned_section(skips[2], self._n_atoms, columns)

        for x in [':', '->', '-']:
            dipole_line = dipole_line.replace(x, '')
        dipole_line = dipole_line.lower()

        #TODO: Add explicit fields for Singlet, Doublet, Triplet, etc.
        state_type = dipole_line.split()[0]
        n_states = dipole_line.count('state')
        if n_states == 1:
            state = int(dipole_line.split()[2])
            key = f'cas_dipole_deriv'
            self._current_frame[key][state] = [dMuX, dMuY, dMuZ]
        elif n_states == 2:
            state_1 = int(dipole_line.split()[2])
            state_2 = int(dipole_line.split()[4])
            key = f'cas_transition_dipole_deriv'
            index = self._cas_state_pair_to_index(state_1, state_2)
            self._current_frame[key][index] = [dMuX, dMuY, dMuZ]