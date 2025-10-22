import numpy as np

from greenheart.simulation.technologies.hydrogen.h2_storage.hydrogen_storage import (
    HydrogenStorage,
)


np.set_printoptions(legacy="1.25")


def config():
    return {
        "max_capacity_kg": 1566000,
        "min_capacity_kg": 0.0,
        "max_charge_rate_kgphr": 9695,
        "initial_state": 783000,
    }


def test_init_from_config():
    h2s_config = config()

    h2s = HydrogenStorage(h2s_config)

    assert h2s.max_capacity_kg == 1566000
    assert h2s.max_charge_rate_kg_hr == 9695
    assert hasattr(h2s, "control_model")
    assert h2s.control_model.bounds_dict["x_ub"] == 1566000


def test_init_defaults():
    h2s = HydrogenStorage()

    assert h2s.max_capacity_kg == 3345172.1687337956
    assert h2s.max_charge_rate_kg_hr == 20710.174804159713
    assert hasattr(h2s, "control_model")
    assert h2s.control_model.bounds_dict["x_ub"] == 3345172.1687337956


def test_update_storage_state():
    h2s = HydrogenStorage(config())
    assert h2s.storage_state == 783000
    h2s.update_storage_state(input_massflow=10)
    assert h2s.storage_state == 783010
    h2s.update_storage_state(input_massflow=-25)
    assert h2s.storage_state == 782985


def test_get_state_measurement():
    h2s = HydrogenStorage(config())

    x_h2s = h2s.get_state_measurement(step_index=0)
    assert x_h2s == 783000

    h2s.update_storage_state(input_massflow=10)
    x_h2s = h2s.get_state_measurement(step_index=0)
    assert x_h2s == 783010

    h2s.update_storage_state(input_massflow=-25)
    x_h2s = h2s.get_state_measurement(step_index=0)
    assert x_h2s == 782985


def test_low_level_controller():
    h2s = HydrogenStorage(config())

    # Trivial charge with 30 kg/hr
    inp = dict(available_massflow=100, desired_massflow=30)
    model_output, model_input, passthrough, curtail = h2s.low_level_controller(**inp)
    assert model_output == 0
    assert model_input == 30
    assert passthrough == 70

    # Trivial discharge with 30 kg/hr
    inp = dict(available_massflow=100, desired_massflow=-30)
    model_output, model_input, passthrough, curtail = h2s.low_level_controller(**inp)
    assert model_output == 30
    assert model_input == -30
    assert passthrough == 100

    # Upper bounded by max charge rate
    inp = dict(available_massflow=10000, desired_massflow=10000)
    model_output, model_input, passthrough, curtail = h2s.low_level_controller(**inp)
    assert model_output == 0
    assert model_input == 9695
    assert passthrough == 305

    # Upper bounded by storage state
    h2s.storage_state = 1566000 - 2500
    inp = dict(available_massflow=5000, desired_massflow=5000)
    model_output, model_input, passthrough, curtail = h2s.low_level_controller(**inp)
    assert model_output == 0
    assert model_input == 2500
    assert passthrough == 2500

    # Upper bounded by available massflow
    h2s.storage_state = 783000
    inp = dict(available_massflow=2000, desired_massflow=5000)
    model_output, model_input, passthrough, curtail = h2s.low_level_controller(**inp)
    assert model_output == 0
    assert model_input == 2000
    assert passthrough == 0

    # Lower bounded by max discharge rate
    inp = dict(available_massflow=0, desired_massflow=-10000)
    model_output, model_input, passthrough, curtail = h2s.low_level_controller(**inp)
    assert model_output == 9695
    assert model_input == -9695
    assert passthrough == 0

    # Lower bounded by storage state
    h2s.storage_state = 2500
    inp = dict(available_massflow=0, desired_massflow=-5000)
    model_output, model_input, passthrough, curtail = h2s.low_level_controller(**inp)
    assert model_output == 2500
    assert model_input == -2500
    assert passthrough == 0


def test_step():
    h2s = HydrogenStorage(config())

    inp = dict(h2_input=np.array([100, 80]), dispatch=np.array([100, 0]), step_index=0)

    # Charge for 10 steps
    for i in range(0, 10):
        inp["step_index"] = i
        model_output, passthrough, curtail = h2s.step(**inp)

    assert model_output == 0
    assert passthrough == 0
    assert curtail == 0

    assert h2s.storage_state == 784000
    assert np.all(h2s.store_charge[0:10] == 100)
    assert h2s.store_charge[11] == 0
    assert np.all(h2s.store_storage_state[0:10] != 0)

    inp["h2_input"] = np.array([0, 80])
    inp["dispatch"] = np.array([0, 100])
    for i in range(10, 20):
        inp["step_index"] = i
        model_output, passthrough, curtail = h2s.step(**inp)

    assert model_output == 100
    assert passthrough == 0
    assert curtail == 0

    assert h2s.storage_state == 783000
    assert np.all(h2s.store_charge[11:20] == -100)
    assert h2s.store_charge[21] == 0
    assert np.all(h2s.store_storage_state[11:20] != 0)
