from __future__ import annotations

from pydantic import BaseModel
from typing import Optional

class TCJobData(BaseModel):

    energy                        : list

    atoms                         : Optional[list] =  None
    geom                          : Optional[list] =  None
    cas_dipole_deriv              : Optional[list] =  None
    cas_dipoles                   : Optional[list] =  None
    cas_states                    : Optional[int]  =  None
    cas_transition_dipole_deriv   : Optional[list] =  None
    cas_transition_dipoles        : Optional[list] =  None
    cis_unrelaxed_dipole_deriv    : Optional[list] =  None
    cis_ex_energies               : Optional[list] =  None
    cis_excitations               : Optional[list] =  None
    cis_gradients                 : Optional[list] =  None
    cis_couplings                 : Optional[list] =  None
    cis_osc_strengths             : Optional[list] =  None
    cis_relaxed_dipole_deriv      : Optional[list] =  None
    cis_relaxed_dipoles           : Optional[list] =  None
    cis_relaxed_esp_charges       : Optional[list] =  None
    cis_relaxed_esp_dipoles       : Optional[list] =  None
    cis_relaxed_resp_charges      : Optional[list] =  None
    cis_relaxed_resp_dipoles      : Optional[list] =  None
    cis_states                    : Optional[int]  =  None
    cis_tr_esp_charges            : Optional[list] =  None
    cis_tr_esp_dipoles            : Optional[list] =  None
    cis_tr_resp_charges           : Optional[list] =  None
    cis_tr_resp_dipoles           : Optional[list] =  None
    cis_transition_dipole_deriv   : Optional[list] =  None
    cis_transition_dipoles        : Optional[list] =  None
    cis_relaxed_transition_dipoles: Optional[list] =  None
    cis_unrelaxed_dipole_deriv    : Optional[list] =  None
    cis_unrelaxed_dipoles         : Optional[list] =  None
    cis_unrelaxed_esp_charges     : Optional[list] =  None
    cis_unrelaxed_esp_dipoles     : Optional[list] =  None
    cis_unrelaxed_resp_charges    : Optional[list] =  None
    cis_unrelaxed_resp_dipoles    : Optional[list] =  None
    cis_transition_dots           : Optional[list] =  None
    dipole_deriv                  : Optional[list] =  None
    dipole_moment                 : Optional[float] = None
    dipole_vector                 : Optional[list] =  None
    esp_charges                   : Optional[list] =  None
    esp_dipoles                   : Optional[list] =  None
    gradient                      : Optional[list] =  None
    resp_charges                  : Optional[list] =  None
    resp_dipoles                  : Optional[list] =  None
    velocities                    : Optional[list] =  None