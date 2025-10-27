import numpy as np
import pytest

from greenheart.simulation.technologies.heat.materials import Quartz
from greenheart.simulation.technologies.heat.heat_storage.thermal_energy_storage import (
    ThermalEnergyStorage,
)


np.set_printoptions(legacy="1.25")


def config():
    # thermal_energy_storage:
    config = dict(
        M_hot_capacity=22500.0e3,  # [kg]
        M_buffer_capacity=22500.0e3,  # [kg]
        mdot_max_charge=1082057,  # [kg h^-1]
        mdot_max_discharge=1082057,  # [kg h^-1]
        T_hot_target=1200,  # [C]
        T_buffer_target=300,  # [C]
        initial_SOC=0.5,
        M_total=22500.0e3,  # [kg]
    )
    return config


def test_init_from_config():
    tes_config = config()
    tes = ThermalEnergyStorage(**tes_config)

    assert tes.C2K == 273.15
    assert tes.kWhpkJ == 1 / 3600
    assert tes.kJpkWh == 3600

    assert tes.M_hot_max == 22500000.0
    assert tes.M_hot_min == 225000.0

    assert tes.M_buffer_max == 22500000.0
    assert tes.M_buffer_min == 225000.0

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

    tes.step_model(m_charge=1000, m_discharge=0, step_index=0)

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

    kwargs = dict(
        available_power=200e3,
        P_charge_desired_kWh=200e3,
        Q_discharge_desired_kWh=75e3,
        step_index=0,
    )

    m_charge, m_discharge, unused_power = tes.low_level_controller(**kwargs)

    assert m_charge == 679801.0253398636
    assert m_discharge == 260127.94336984577
    assert unused_power == 0


    kwargs["available_power"] = 300e3
    m_charge, m_discharge, unused_power = tes.low_level_controller(**kwargs)

    assert m_charge == 679801.0253398636
    assert m_discharge == 260127.94336984577
    assert unused_power == 100000.0
    
    kwargs["available_power"] = 200e3
    kwargs["P_charge_desired_kWh"] = 300e3
    m_charge, m_discharge, unused_power = tes.low_level_controller(**kwargs)

    assert m_charge == 679801.0253398636
    assert m_discharge == 260127.94336984577
    assert unused_power == 0
    

    # Requesting charge above maximum charge rate
    kwargs["available_power"] = 500e3
    kwargs["P_charge_desired_kWh"] = 500e3
    m_charge, m_discharge, unused_power = tes.low_level_controller(**kwargs)

    assert m_charge == 1082057.0
    assert m_discharge == 260127.94336984577
    assert unused_power == 177746.32723402118




def test_step():

    tes_config = config()
    tes = ThermalEnergyStorage(**tes_config)

    kwargs = dict(
        available_power= np.array([100e3]),
        dispatch=np.array([0, 50e3]),
        step_index=0
    )

    y_model, u_passthrough, u_curtail = tes.step(**kwargs)

    assert y_model[0] == 49922.659086029336
    assert u_passthrough == 0
    assert u_curtail == 100000.0

    kwargs["dispatch"] = np.array([0, 100e3])


    assert tes._SOC() == 0.48695695303422076

    for i in range(10):
        y_model, u_passthrough, u_curtail = tes.step(**kwargs)

    assert tes._SOC() == 0.3294433118258059

    for i in range(22):
        y_model, u_passthrough, u_curtail = tes.step(**kwargs)

    assert y_model[0] == 2283.9520293826636
    assert u_passthrough == 0
    assert u_curtail == 100000.0


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

        T1 = 500  # [K]
        T2 = 1500  # [K]

        Cp1 = q.Cp(T1)
        Cp2 = q.Cp(T2)

        assert Cp1 == 59.642301
        assert Cp2 == 73.97312297222221

    def test_quartz_H(self):
        # quartz enthalpy calculation
        q = Quartz()

        T1 = 500  # [K]
        T2 = 1500  # [K]

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

        T1 = 500  # [K]
        T2 = 1500  # [K]

        S1 = q.S(T1)
        S2 = q.S(T2)

        assert S1 == 68.49880591906594
        assert S2 == 144.92539382161112
