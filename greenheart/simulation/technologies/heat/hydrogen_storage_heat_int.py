from scipy.constants import R, convert_temperature
from scipy.constants import liter #as cubic meters
from scipy.constants import atm, bar #converts to Pa
from greenheart.simulation.technologies.hydrogen.h2_storage.salt_cavern.salt_cavern import SaltCavernStorage
# - needs to have outlet_pressure as attribute (max is 120 bar)
from greenheart.simulation.technologies.hydrogen.h2_storage.pressure_vessel.tankinator import Tank,LinedTank
# - no idea if this is used
from greenheart.simulation.technologies.hydrogen.h2_storage.pipe_storage.underground_pipe_storage import UndergroundPipeStorage
# - self.compressor_output_pressure [bar] is an attribute
from greenheart.simulation.technologies.hydrogen.h2_storage.lined_rock_cavern.lined_rock_cavern import LinedRockCavernStorage
# - needs to have outlet_pressure as attribute (max is 200 bar)
# def ideal_gas_law_solve_for_T():
#     """
#     P: pressure in Pa
#     V: volume in m^3
#     n: # of moles
#     R: gas constant (8.314 J/K-mol)
#     T: temperature in Kelvin
#     """
#     T_k = pv/nR
#     T_c = convert_temperature([T_k], "K", "C")[0]
#     return T_c

def estimate_h2_storage_volume(h2_storage_size_kg):
    density_h2_kg_pr_m3 = 0.083 #kg/m^3
    h2_volume_m3 = h2_storage_size_kg/density_h2_kg_pr_m3
    return h2_volume_m3
def calc_moles_from_mass(h2_storage_size_kg):
    m_H2 = 20.16 #g/mol
    h2_storage_size_mol = h2_storage_size_kg*1e3/m_H2
    return h2_storage_size_mol
def estimate_h2_storage_temp(h2_storage_outlet_pressure_bar,h2_storage_size_kg):
    P_pa = h2_storage_outlet_pressure_bar/bar
    n = calc_moles_from_mass(h2_storage_size_kg)
    v = estimate_h2_storage_volume(h2_storage_size_kg)
    T_k = (P_pa*v)/(n*R)
    T_c = convert_temperature([T_k], "K", "C")[0]
    return T_c


