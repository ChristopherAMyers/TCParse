from __future__ import annotations

from pydantic import BaseModel
from typing import List
from typing import Optional, Union, get_args, get_origin, TypeAlias
import numpy as np

class TCJobData(BaseModel):

    energy                         : List[float]
    atoms                          : List[str]         | None = None
    geom                           : list[list | None] | None = None
    dipole_moment                  : float             | None = None
    dipole_vector                  : List[float]       | None = None

    #   DFT dataInput Option Dependent Fields
    dipole_deriv                   : list[list | None]  | None = None
    gs_gradient                    : list[list | None]  | None = None
    esp_charges                    : List[float]        | None = None
    esp_dipole                     : List[float]        | None = None
    gradient                       : list[list | None]  | None = None
    resp_charges                   : List[float]        | None = None
    resp_dipole                    : List[float]        | None = None
    velocities                     : list[list | None]  | None = None
 
    #   CIS-Specific Fields
    cis_states                     : int                | None = None
    cis_unrelaxed_dipole_deriv     : list[list | None]  | None = None
    cis_ex_energies                : List[float]        | None = None
    cis_excitations                : list[list | None]  | None = None
    cis_gradients                  : list[list | None]  | None = None
    cis_couplings                  : list[list | None]  | None = None
    cis_osc_strengths              : List[float]        | None = None
    cis_relaxed_dipole_deriv       : list[list | None]  | None = None
    cis_relaxed_dipoles            : list[list | None]  | None = None
    cis_relaxed_esp_charges        : list[list | None]  | None = None
    cis_relaxed_esp_dipoles        : list[list | None]  | None = None
    cis_relaxed_resp_charges       : list[list | None]  | None = None
    cis_relaxed_resp_dipoles       : list[list | None]  | None = None
    cis_tr_esp_charges             : list[list | None]  | None = None
    cis_tr_esp_dipoles             : list[list | None]  | None = None
    cis_tr_resp_charges            : list[list | None]  | None = None
    cis_tr_resp_dipoles            : list[list | None]  | None = None
    cis_transition_dipole_deriv    : list[list | None]  | None = None
    cis_transition_dipoles         : list[list | None]  | None = None
    cis_relaxed_transition_dipoles : list[list | None]  | None = None
    cis_unrelaxed_dipoles          : list[list | None]  | None = None
    cis_unrelaxed_esp_charges      : list[list | None]  | None = None
    cis_unrelaxed_esp_dipoles      : list[list | None]  | None = None
    cis_unrelaxed_resp_charges     : list[list | None]  | None = None
    cis_unrelaxed_resp_dipoles     : list[list | None]  | None = None
    cis_transition_dots            : list[list | None]  | None = None
    
    #   CAS-Specific Fields (Only for CAS-singles for now)
    cas_dipole_deriv               : list[list | None]  | None = None
    cas_dipoles                    : list[list | None]  | None = None
    cas_states                     : int                | None = None
    cas_transition_dipole_deriv    : list[list | None]  | None = None
    cas_transition_dipoles         : list[list | None]  | None = None

    def append_results(self, *datasets: TCJobData) -> TCJobData:
        '''
            Combine multiple TCJobData objects into a single object. This is most useful
            for multiple CIS jobs that have different computations (e.g. one has CIS 
            gradients, the other has CIS dipole derivatives). Preference for identical
            fields (e.g. energy, dipole_moment, etc.) is given in reverse order of
            datasets. The last dataset will be used for the final value of the field.

            Parameters
            ----------
            *datasets: TCJobData
                The datasets to combine.

            Returns
            -------
            combined_data: TCJobData
                Combined TCJobData object.
        '''


        #   check if the datasets are all TCJobData objects of the same type of job
        if self.geom is not None:
            min_geom_rmsd = np.inf
            for dataset in datasets:
                if dataset.geom is None:
                    continue
                geom_rmsd = np.linalg.norm(np.array(self.geom) - np.array(dataset.geom))/np.sqrt(len(self.geom))
                if geom_rmsd < min_geom_rmsd:
                    min_geom_rmsd = geom_rmsd
            
            if min_geom_rmsd > 1e-3:
                raise ValueError(f"Datasets have geometries differing by {min_geom_rmsd:.3e}. Cannot combine datasets with different geometries with an RMSD > 1e-3.")
            
        if self.atoms is not None:
            for dataset in datasets:
                if dataset.atoms is None:
                    continue
                if len(self.atoms) != len(dataset.atoms):
                    raise ValueError(f"Datasets have different number of atoms. Cannot combine datasets with different number of atoms.")
                if set(self.atoms) != set(dataset.atoms):
                    raise ValueError(f"Datasets have different atoms. Cannot combine datasets with different atoms.")
        

        combined_data = {}
        for dataset in (self,) + datasets:
            for key, val in dataset.model_dump().items():
                if key not in combined_data:
                    combined_data[key] = val
                    continue

                if combined_data[key] is None:
                    combined_data[key] = val
                    continue

                #   for all list fields, we don't overwrite with None values
                if isinstance(val, list):
                    if combined_data[key] is None:
                        combined_data[key] = [None for _ in range(len(val))]

                    for i in range(len(val)):
                        if val[i] is None:
                            continue
                        combined_data[key][i] = val[i]
                elif val is not None:
                    combined_data[key] = val

        data = TCJobData.model_validate(combined_data)
        self.__dict__.update(data.__dict__)

                    