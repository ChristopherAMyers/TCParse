from tcpb import TCProtobufClient
import MDAnalysis as mda
import pickle
import os
import json
import time
import numpy as np
import sys

DEBYE_2_AU = 0.3934303
ANG_2_BOHR = 1.8897259886
BOHR_2_ANG = 1.0/ANG_2_BOHR

def reorder_charge_info(data):
    ''' Never finished this function!!!!!'''
    energies = data['energy']
    n_states = len(energies)
    new_esp_data = {}

    all_esp_analysis_keys = [k for k in data if 'esp_analysis_' in k]
    if 'esp_analysis_S0' in data:
        all_esp_analysis_keys.pop('esp_analysis_S0')
        pass

    if n_states > 1:
        new_esp_data['cis_unrelaxed_esp_charges'] = []
        new_esp_data['cis_unrelaxed_esp_dipoles'] = []
        new_esp_data['cis_unrelaxed_resp_charges'] = []
        new_esp_data['cis_unrelaxed_resp_dipoles'] = []

        for i in range(1, n_states):
            key = f'esp_analysis_S{i}'
            esp_data = data[key]
            all_esp_analysis_keys.pop(key)
            new_esp_data['cis_unrelaxed_esp_charges'].append(esp_data['esp_charges'])
            new_esp_data['cis_unrelaxed_esp_dipoles'].append(esp_data['esp_dipole'])
            new_esp_data['cis_unrelaxed_resp_charges'].append(esp_data['resp_charges'])
            new_esp_data['cis_unrelaxed_resp_dipoles'].append(esp_data['resp_dipole'])

        for i in range(1, n_states):
            for j in range(i+1, n_states):
                key = f'esp_analysis_S{i}_S{j}'
                esp_data = data[key]
                all_esp_analysis_keys.pop(key)
                new_esp_data['cis_unrelaxed_TrESP_charges'].append(esp_data['esp_charges'])
                new_esp_data['cis_unrelaxed_TrESP_dipoles'].append(esp_data['esp_dipole'])
                new_esp_data['cis_unrelaxed_TrRESP_charges'].append(esp_data['resp_charges'])
                new_esp_data['cis_unrelaxed_TrRESP_dipoles'].append(esp_data['resp_dipole'])

    unrelaxed_dipoles = data['cis_unrelaxed_dipoles']
    tr_dipoles = data['cis_transition_dipoles']

def correct_cis_signs(data, compare_vecs=None):
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
    
class TCParcer():
    def __init__(self, tc_output_file) -> None:
        self._tc_output_file_path = os.path.split(tc_output_file)[0]
        self._file = open(tc_output_file)
        self._atoms = None

    def __del__(self):
        self._file.close()

    def parse_charge_info(self, data, n_atoms, previous_line):

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

    def parse_cis_section(self, n_atoms, data: dict):
        n_states = 0
        coefficients = []
        for line in self._file:
            # if '' in line:
            #     return
            
            if 'Excited State Gradient' in line:
                root = int(line.split()[1].replace(':', ''))
                for i in range(3): next(self._file)
                grad = np.zeros((n_atoms, 3))
                for i in range(n_atoms):
                    line = next(self._file)
                    grad[i] = np.array([float(x) for x in line.split()[1:]])
                data[f'cis_gradient_{root}'] = grad.tolist()
            
            elif 'Total Energy (a.u.)   Ex. Energy (a.u.)     Ex. Energy (eV)' in line:
                line = next(self._file)
                line = next(self._file)
                while len(line.split()) > 0:
                    n_states += 1
                    energy = float(line.split()[1])
                    data['energy'].append(energy)
                    line = next(self._file)
                data['cis_states'] = n_states
            
            elif 'Unrelaxed excited state dipole moments' in line:
                dipoles = []
                for i in range(3): next(self._file)
                for i in range(n_states):
                    line = next(self._file)
                    dip = [float(x)*DEBYE_2_AU for x in line.split()[1:4]]
                    dipoles.append(dip)
                data['cis_unrelaxed_dipoles'] = dipoles

            elif 'Relaxed excited state dipole moments:' in line:
                data['cis_relaxed_dipoles'] = []
                for i in range(3): next(self._file)
                line = next(self._file)
                while len(line.split()) > 0:
                    dip = [float(x) for x in line.split()[1:4]]
                    data['cis_relaxed_dipoles'].append(dip)
                    line = next(self._file)
            
            elif 'Transition dipole moments:' in line:
                data['cis_transition_dipoles'] = []
                for i in range(3): next(self._file)
                for i in range(n_states):
                    line = next(self._file)
                    dip = [float(x) for x in line.split()[1:4]]
                    data['cis_transition_dipoles'].append(dip)

            elif 'Transition dipole moments between excited states' in line:
                for i in range(3): next(self._file)
                n_lines = int((n_states - 1)*(n_states - 2)/2)
                for i in range(n_states):
                    line = next(self._file)
                    dip = [float(x) for x in line.split()[3:6]]
                    data['cis_transition_dipoles'].append(dip)


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

            elif 'Final Excited State Results' in line:
                for i in range(3): next(self._file)
                excitations = []
                ex_energies = []
                for i in range(n_states):
                    sp = next(self._file).split()
                    excitations.append([int(sp[6]), int(sp[8])])
                    ex_energies.append(float(sp[2]))
                # data['cis_excitations'] = excitations
                data['cis_ex_energies'] = ex_energies

                #   this is the end of the CIS section, so leave the function
                return

    def parse_data(self, positions: list, vels: list=None, end_phrase=None):

        n_atoms = len(self._atoms)

        data = {
            'atoms': self._atoms,
            'geom': positions
        }
        if vels is not None:
            data['velocities'] = vels
        data['energy'] = []

        prev_line = ''
        for line in self._file:

            if end_phrase is not None:
                if end_phrase in line:
                    # print("PHARASE FOUND: ", data.keys())
                    # input()
                    break
        
            if 'Current Geometry' in line:
                print("CURRENT")
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
                for x in ['{', '}', ',', '(', ')']:
                    line = line.replace(x, ' ')
                dip = [float(x) for x in line.split()[2:5]]
                mag = np.linalg.norm(dip)
                data['dipole_moment'] = mag
                data['dipole_vector'] = dip

            elif 'CIS Parameters' in line:
                self.parse_cis_section(n_atoms, data)

            elif 'Gradient units are Hartree/Bohr' in line:
                next(self._file)
                next(self._file)
                grad = []
                for i in range(n_atoms):
                    sp = next(self._file).split()
                    grad.append([float(x) for x in sp])
                data['gradient'] = grad

            elif 'Running Resp charge analysis...' in line:
                self.parse_charge_info(data, n_atoms, prev_line)

            prev_line = line

        return data

    def parse_file(self, data_output_file=None):

        # tc_output_file = os.path.abspath('ex_1_high_res/tc.out')
        # data_output_file = os.path.abspath('ex_1_high_res/data.json')

        is_md = False
        is_num_dipole = False
        frame_data = {}
        props = {}
        coords_univ = None
        vels_file = None
        # atoms = None
        # with open(tc_output_file) as file:
        n_atoms = None
        for line in self._file:
            if 'Total atoms:' in line:
                n_atoms = int(line.split()[2])

            elif 'XYZ coordinates' in line:
                coords_file = os.path.join(self._tc_output_file_path, line.split()[2])
                coords_univ = mda.Universe(coords_file)
                
            elif 'Scratch directory:' in line:
                print(self._tc_output_file_path, line.split()[2])
                scr_dir = os.path.join(self._tc_output_file_path, line.split()[2])
                
            if 'RUNNING AB INITIO MOLECULAR DYNAMICS' in line:
                is_md = True
                coords_file = os.path.join(scr_dir, 'coors.xyz')
                coords_univ = mda.Universe(coords_file)
                vels_file = os.path.join(scr_dir, 'velocities.xyz')
                vels_univ = mda.Universe(vels_file)
            elif 'NUMERICAL DIPOLE DERIVATIVES' in line:
                is_num_dipole = True


            if n_atoms is not None:
                self._atoms = coords_univ.atoms.elements.tolist()
                coords = coords_univ.atoms.positions.tolist()

                #   MD trajectories have multiple parses
                if is_md:
                    vels = vels_univ.atoms.positions.tolist()
                    data = self.parse_data(coords, vels, "MD STEP")
                    frame_data[coords_univ.trajectory.frame] = data
                    # print(json.dumps(data, indent=4))
                    if coords_univ.trajectory.frame == coords_univ.trajectory.n_frames - 1:
                        break
                    coords_univ.trajectory.next()
                    vels_univ.trajectory.next()
                    print(coords_univ.trajectory.frame)
                elif is_num_dipole:
                    data = self.parse_data(coords, end_phrase="Current Geometry")
                    frame_data[len(frame_data)] = data

                #   single jobs need to be parsed only once
                else:
                    data = self.parse_data(coords)

        if data_output_file is not None:
            print("Writing data to ", data_output_file)
            with open(data_output_file, 'w') as file:
                if is_md or is_num_dipole:
                    json.dump(frame_data, file, indent=4)
                else:
                    json.dump(data, file, indent=4)
        
        if is_md:
            return frame_data
        else:
            return data

def main():
    parser = TCParcer(sys.argv[1])
    parser.parse_file(sys.argv[2])
    # parse_file(sys.argv[1], sys.argv[2])

if __name__ == '__main__':
    main()