import numpy as np
import pytest

from greenheart.simulation.technologies.heat.materials import Quartz
from greenheart.simulation.technologies.heat.heat_storage.thermal_energy_storage import ThermalEnergyStorage


np.set_printoptions(legacy="1.25")


def config():
    # thermal_energy_storage:
    config = dict(
        M_hot_capacity= 22500.0e+3, # [kg]
        M_buffer_capacity= 22500.0e+3, # [kg]
        mdot_max_charge= 1082057, # [kg h^-1]
        mdot_max_discharge= 1082057, # [kg h^-1]
        T_hot_target= 1200, # [C]
        T_buffer_target= 300, # [C]
        initial_SOC= 0.5,
        M_total= 22500.0e+3, # [kg]
    )
    return config


def test_init_from_config():
    tes_config = config()
    tes = ThermalEnergyStorage(**tes_config)


    assert tes.C2K == 273.15
    assert tes.kWhpkJ == 1/3600
    assert tes.kJpkWh == 3600

    assert tes.M_hot_max ==22500000.0
    assert tes.M_hot_min ==225000.0
    
    assert tes.M_buffer_max ==22500000.0
    assert tes.M_buffer_min ==225000.0

    assert tes.H_hot_max_kWh == 8228681.312702628
    assert tes.H_buffer_max_kWh == 1751523.4867248915
    assert tes.H_capacity_kWh == 6477157.825977737

    assert tes.M_hot == 11250000.0




def test_storage_state():
    tes_config = config()
    tes = ThermalEnergyStorage(**tes_config)


    assert tes.tank_H(which="hot") == 4114340.656351314
    assert tes.tank_H(which="buffer") == 875761.7433624457
    assert tes._SOC() == 0.4949494949494949

    # So bookkeeping doesn't throw an error
    tes.P_used = 0
    tes.Q_out_kWh = 0
    tes.P_charge_desired_kWh = 0
    tes.Q_discharge_desired_kWh = 0
    tes.m_charge = 1000
    tes.m_discharge = 0
    tes.unused_power = 0

    tes.step_model(m_charge = 1000, m_discharge=0, step_index=0)

    assert tes.tank_H(which="hot") == 4112992.0668914863
    assert tes.tank_H(which="buffer") == 875319.0295687817
    assert tes._SOC() == 0.4947839506145259

    assert tes.M_hot == 11251000.0
    assert tes.M_buffer == 11249000.0




def test_get_state_measurement():    
    tes_config = config()
    tes = ThermalEnergyStorage(**tes_config)

    x0 = tes.get_state_measurement(0)
    assert x0[0] == 4114340.656351314
    assert x0[1] == 11250000.0





def test_low_level_controller():
    tes_config = config()
    tes = ThermalEnergyStorage(**tes_config)

    []
#     h2s = HydrogenStorage(config())

#     # Trivial charge with 30 kg/hr
#     inp = dict(available_massflow=100, desired_massflow=30)
#     model_output, model_input, passthrough, curtail = h2s.low_level_controller(**inp)
#     assert model_output == 0
#     assert model_input == 30
#     assert passthrough == 70

#     # Trivial discharge with 30 kg/hr
#     inp = dict(available_massflow=100, desired_massflow=-30)
#     model_output, model_input, passthrough, curtail = h2s.low_level_controller(**inp)
#     assert model_output == 30
#     assert model_input == -30
#     assert passthrough == 100

#     # Upper bounded by max charge rate
#     inp = dict(available_massflow=10000, desired_massflow=10000)
#     model_output, model_input, passthrough, curtail = h2s.low_level_controller(**inp)
#     assert model_output == 0
#     assert model_input == 9695
#     assert passthrough == 305

#     # Upper bounded by storage state
#     h2s.storage_state = 1566000 - 2500
#     inp = dict(available_massflow=5000, desired_massflow=5000)
#     model_output, model_input, passthrough, curtail = h2s.low_level_controller(**inp)
#     assert model_output == 0
#     assert model_input == 2500
#     assert passthrough == 2500

#     # Upper bounded by available massflow
#     h2s.storage_state = 783000
#     inp = dict(available_massflow=2000, desired_massflow=5000)
#     model_output, model_input, passthrough, curtail = h2s.low_level_controller(**inp)
#     assert model_output == 0
#     assert model_input == 2000
#     assert passthrough == 0

#     # Lower bounded by max discharge rate
#     inp = dict(available_massflow=0, desired_massflow=-10000)
#     model_output, model_input, passthrough, curtail = h2s.low_level_controller(**inp)
#     assert model_output == 9695
#     assert model_input == -9695
#     assert passthrough == 0

#     # Lower bounded by storage state
#     h2s.storage_state = 2500
#     inp = dict(available_massflow=0, desired_massflow=-5000)
#     model_output, model_input, passthrough, curtail = h2s.low_level_controller(**inp)
#     assert model_output == 2500
#     assert model_input == -2500
#     assert passthrough == 0


def test_step():

    tes_config = config()
    tes = ThermalEnergyStorage(**tes_config)


    []
#     h2s = HydrogenStorage(config())

#     inp = dict(h2_input=np.array([100, 80]), dispatch=np.array([100, 0]), step_index=0)

#     # Charge for 10 steps
#     for i in range(0, 10):
#         inp["step_index"] = i
#         model_output, passthrough, curtail = h2s.step(**inp)

#     assert model_output == 0
#     assert passthrough == 0
#     assert curtail == 0

#     assert h2s.storage_state == 784000
#     assert np.all(h2s.store_charge[0:10] == 100)
#     assert h2s.store_charge[11] == 0
#     assert np.all(h2s.store_storage_state[0:10] != 0)

#     inp["h2_input"] = np.array([0, 80])
#     inp["dispatch"] = np.array([0, 100])
#     for i in range(10, 20):
#         inp["step_index"] = i
#         model_output, passthrough, curtail = h2s.step(**inp)

#     assert model_output == 100
#     assert passthrough == 0
#     assert curtail == 0

#     assert h2s.storage_state == 783000
#     assert np.all(h2s.store_charge[11:20] == -100)
#     assert h2s.store_charge[21] == 0
#     assert np.all(h2s.store_storage_state[11:20] != 0)







class TestQuartz:
    def test_quartz_params(self):
        q = Quartz()

        assert q.molar_mass == 60.0843
        assert q.T_max == 1996
        assert q.T_max == np.max(q.T)


    def test_quartz_temperature_bounds(self):
        q = Quartz()

        # Assert that quartz throws a value error when temperature is too high
        with pytest.raises(ValueError):
            q.H(temperature=2000)



    def test_quartz_Cp(self):
        # quartz heat capacity calculation
        q = Quartz()

        T1 = 500 # [K]
        T2 = 1500 # [K]
        
        Cp1 = q.Cp(T1)
        Cp2 = q.Cp(T2)

        assert Cp1 == 59.642301
        assert Cp2 == 73.97312297222221

 

    def test_quartz_H(self):
        # quartz enthalpy calculation
        q = Quartz()

        T1 = 500 # [K]
        T2 = 1500 # [K]
        
        H1 = q.H(T1)
        H2 = q.H(T2)
        
        assert H1 == 10.684118916666648
        assert H2 == 81.08888832291666
        
        Hk1 = q.H_kwhpkg(T1)
        Hk2 = q.H_kwhpkg(T2)
        
        assert Hk1 == 0.05609213417609704
        assert Hk2 == 0.3748848068598649

    def test_quartz_S(self):
        # quarts entropy calculation
        q = Quartz()

        T1 = 500 # [K]
        T2 = 1500 # [K]
        
        S1 = q.S(T1)
        S2 = q.S(T2)

        assert S1 == 68.49880591906594
        assert S2 == 144.92539382161112

