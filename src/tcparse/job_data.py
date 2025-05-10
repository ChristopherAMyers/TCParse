from __future__ import annotations

from pydantic import BaseModel
from typing import List

class TCJobData(BaseModel):

    energy                         : List[float]
    atoms                          : List[str]                        | None = None
    geom                           : List[List[float]]                | None = None
    dipole_moment                  : float                            | None = None
    dipole_vector                  : List[float]                      | None = None

    #   DFT dataInput Option Dependent Fields
    dipole_deriv                   : List[List[List[float]]]          | None = None
    esp_charges                    : List[float]                      | None = None
    esp_dipoles                    : List[float]                      | None = None
    gradient                       : List[List[float]]                | None = None
    resp_charges                   : List[float]                      | None = None
    resp_dipoles                   : List[float]                      | None = None
    velocities                     : List[List[float]]                | None = None
 
    #   CIS-Specific Fields
    cis_states                     : int                              | None = None
    cis_unrelaxed_dipole_deriv     : List[List[List[List[float]]]]    | None = None
    cis_ex_energies                : List[float]                      | None = None
    cis_excitations                : List[List[List[float]]]          | None = None
    cis_gradients                  : List[List[List[float]]]          | None = None
    cis_couplings                  : List[List[List[float]]]          | None = None
    cis_osc_strengths              : List[float]                      | None = None
    cis_relaxed_dipole_deriv       : List[List[List[List[float]]]]    | None = None
    cis_relaxed_dipoles            : List[List[float]]                | None = None
    cis_relaxed_esp_charges        : List[List[float]]                | None = None
    cis_relaxed_esp_dipoles        : List[List[float]]                | None = None
    cis_relaxed_resp_charges       : List[List[float]]                | None = None
    cis_relaxed_resp_dipoles       : List[List[float]]                | None = None
    cis_tr_esp_charges             : List[List[float]]                | None = None
    cis_tr_esp_dipoles             : List[List[float]]                | None = None
    cis_tr_resp_charges            : List[List[float]]                | None = None
    cis_tr_resp_dipoles            : List[List[float]]                | None = None
    cis_transition_dipole_deriv    : List[List[List[List[float]]]]    | None = None
    cis_transition_dipoles         : List[List[float]]                | None = None
    cis_relaxed_transition_dipoles : List[List[float]]                | None = None
    cis_unrelaxed_dipoles          : List[List[float]]                | None = None
    cis_unrelaxed_esp_charges      : List[List[float]]                | None = None
    cis_unrelaxed_esp_dipoles      : List[List[float]]                | None = None
    cis_unrelaxed_resp_charges     : List[List[float]]                | None = None
    cis_unrelaxed_resp_dipoles     : List[List[float]]                | None = None
    cis_transition_dots            : List[list[float]]                | None = None
    
    #   CAS-Specific Fields (Only for CAS-singles for now)
    cas_dipole_deriv               : List[List[List[List[float]]]]    | None = None
    cas_dipoles                    : List[List[float]]                | None = None
    cas_states                     : int                              | None = None
    cas_transition_dipole_deriv    : List[List[List[List[float]]]]    | None = None
    cas_transition_dipoles         : List[List[float]]                | None = None