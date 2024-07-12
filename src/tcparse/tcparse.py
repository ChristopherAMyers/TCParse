import os
import json
import numpy as np
import sys
import io

DEBYE_2_AU = 0.3934303
ANG_2_BOHR = 1.8897259886
BOHR_2_ANG = 1.0/ANG_2_BOHR

class XYZFile():
    def __init__(self, file_loc) -> None:
        self._atom_data = []
        self._coord_data = []
        self._comment_data = []
        self._current_frame = 0

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
    def __init__(self) -> None:
        self._tc_output_file_path = None
        self._coords = None

        self._atoms = None
        self._n_atoms = None
        self._current_frame = 0
        self._job_type = 'energy'

        self._vels_univ = None
        self._coords_univ = None

        self._data = {}
        self._add_frame_to_data()

    def __del__(self):
        self._file.close()

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
                dipoles = []
                for i in range(3): next(self._file)
                for i in range(n_states):
                    line = next(self._file)
                    dip = [float(x)*DEBYE_2_AU for x in line.split()[1:4]]
                    dipoles.append(dip)
                data['cas_dipoles'] = dipoles

            elif 'Singlet state electronic transitions:' in line:
                data['cas_transition_dipoles'] = []
                for i in range(3): next(self._file)
                line = next(self._file)
                while len(line.split()) > 0:
                    dip = [float(x) for x in line.split()[3:6]]
                    data['cas_transition_dipoles'].append(dip)
                    line = next(self._file)

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

    def _parse_cis_section(self, n_atoms, data: dict):
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

    def _add_frame_to_data(self):
        if self._current_frame not in self._data:
            self._data[self._current_frame] = {
                    'atoms': self._atoms,
                    'geom': None,
                    'energy': []
                }

    def _parse_data(self, list=None, end_phrase=None):

        n_atoms = self._n_atoms

        self._add_frame_to_data()
        data = self._data[self._current_frame]

        prev_line = ''
        for line in self._file:

            if end_phrase is not None:
                if end_phrase in line:
                    if self._current_frame % 100 == 0:
                        print("Parsing frame ", self._current_frame)
                    self._current_frame += 1
                    self._add_frame_to_data()
                    data = self._data[self._current_frame]

        
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
                for x in ['{', '}', ',', '(', ')']:
                    line = line.replace(x, ' ')
                dip = [float(x) for x in line.split()[2:5]]
                mag = np.linalg.norm(dip)
                data['dipole_moment'] = mag
                data['dipole_vector'] = dip

            elif 'CIS Parameters' in line:
                self._parse_cis_section(n_atoms, data)

            elif 'CAS Parameters' in line:
                self._parse_cas_section(n_atoms, data)

            elif 'Dipole Derivative' in line:
                self._parse_dipole_deriv(data, line)

            elif 'Gradient units are Hartree/Bohr' in line:
                data['gradient'] = self._parse_gradient()

            elif 'Running Resp charge analysis...' in line:
                self._parse_charge_info(data, n_atoms, prev_line)

            prev_line = line

        self._post_process_data(data)

        return data
    
    def _post_process_data(self, data: dict):
        n_states = len(data['energy'])
        if 'num_dipole_derivatives' in data:
            old_data = data['num_dipole_derivatives']
            new_data = []
            for i in range(int(n_states*(n_states+1)/2)):
                dMu_X = old_data.pop(0)
                dMu_Y = old_data.pop(0)
                dMu_Z = old_data.pop(0)
                new_data.append([dMu_X, dMu_Y, dMu_Z])
            data['num_dipole_derivatives'] = new_data

    def _parse_dipole_deriv(self, data, current_line):
        #   add entry to data if not present
        if 'num_dipole_derivatives' not in data:
            data['num_dipole_derivatives'] = []
        section = data['num_dipole_derivatives']
        derivatives = []
  
        #   figure out what derivatives these are
        sp = current_line.split()
        idx = sp.index('State')
        state_1 = int(sp[idx+1].replace(':', ''))
        for j in range(idx+1):
            sp.pop(0)
        if 'State' in sp:
            idx = sp.index('State')
            state_2 = int(sp[idx+1].replace(':', ''))
        else:
            state_2 = state_1
        direction = sp[-1]

        #   read in derivatives
        while 'dE' not in current_line:
            current_line = next(self._file)
        current_line = next(self._file)
        while '---' not in current_line:
            derivatives.append([float(x) for x in current_line.split()])
            current_line = next(self._file)

        data['num_dipole_derivatives'].append(derivatives)
        
    
    def _parse_gradient(self):
        next(self._file)
        next(self._file)
        grad = []
        for i in range(self._n_atoms):
            sp = next(self._file).split()
            grad.append([float(x) for x in sp])
        return grad

    def parse_from_list(self, file_lines: list[str], data_output_file=None):
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
        return self._parse_all(data_output_file)
    
    def parse_from_file(self, tc_output_file: str, data_output_file=None):
        self._tc_output_file_path = os.path.split(tc_output_file)[0]
        self._file = open(tc_output_file)
        return self._parse_all(data_output_file)

    def _parse_all(self, data_output_file=None):

        is_md = False
        is_num_deriv = False
        coords_univ = None
        vels_file = None
        n_atoms_job = None
        scr_dir = None
        for line in self._file:
            if 'Total atoms:' in line:
                n_atoms_job = int(line.split()[2])
                if self._n_atoms is not None and self._n_atoms != n_atoms_job:
                    raise ValueError('Number of supplied atoms does not equal the amount specified in job output')
                self._n_atoms = n_atoms_job


            elif 'XYZ coordinates' in line:
                if self._coords is not None:
                    self._data[self._current_frame]['geom'] = self._coords
                elif self._tc_output_file_path is not None:
                    coords_file = os.path.join(self._tc_output_file_path, line.split()[2])
                    if not os.path.isfile(coords_file):
                        print("Could not find ", coords_file)
                        print("    Coordinates will not be imported")
                    else:
                        coords_univ = XYZFile(coords_file)
                        self._atoms = coords_univ.get_atoms()
                        coords = coords_univ.get_coords()
                        self._data[self._current_frame]['geom'] = coords
                        self._data[self._current_frame]['atoms'] = self._atoms 
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
                    self._data[self._current_frame]['velocities'] = vels
                
            elif 'Scratch directory:' in line:
                if self._tc_output_file_path is not None:
                    scr_dir = os.path.join(self._tc_output_file_path, line.split()[2])


            if 'RUNNING AB INITIO MOLECULAR DYNAMICS' in line:
                is_md = True
            elif 'NUMERICAL DIPOLE' in line or 'NUMERICAL NONADIABATIC' in line:
                self._current_frame = -1
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

        if data_output_file is not None:
            print("Writing data to ", data_output_file)
            with open(data_output_file, 'w') as file:
                if is_md or is_num_deriv:
                    json.dump(self._data, file, indent=4)
                else:
                    json.dump(self._data[0], file, indent=4)
        
        if is_md or is_num_deriv: return self._data
        else: return self._data[0]


def main():
    parser = TCParcer()
    parser.parse_from_file(sys.argv[1], sys.argv[2])

if __name__ == '__main__':
    main()