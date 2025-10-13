import pytest
from pathlib import Path

from greenheart.simulation.greenheart_simulation import GreenHeartSimulationConfig
from greenheart.simulation.realtime_simulation import RealTimeSimulation
from greenheart.simulation.technologies.dispatch.dispatch import GreenheartDispatch
import greenheart.tools.eco.hopp_mgmt as he_hopp
from greenheart.simulation.greenheart_simulation import setup_greenheart_simulation


def fpaths():
    config_root = Path(__file__).parent / "input_files"

    return dict(
        hopp=config_root / "plant/hopp_config_mn.yaml",
        greenheart=config_root / "plant/greenheart_config_onshore_mn.yaml",
        turbine=config_root / "turbines/ATB2024_6MW_170RD_floris_turbine.yaml",
        floris=config_root / "floris/floris_input_lbw_6MW.yaml",
    )


def make_config():
    configs = fpaths()
    config = GreenHeartSimulationConfig(
        str(configs["hopp"]),
        str(configs["greenheart"]),
        str(configs["turbine"]),
        str(configs["floris"]),
        verbose=False,
        show_plots=False,
        save_plots=False,
        use_profast=True,
        post_processing=True,
        incentive_option=1,
        plant_design_scenario=1,
        output_level=8,
    )
    config.realtime_simulation = True
    return config


def make_realtime_simulator():
    config = make_config()
    config, hi, _ = setup_greenheart_simulation(config)

    simulator = RealTimeSimulation(config, hi)
    dispatcher = GreenheartDispatch(
        hi,
        config,
        simulator,
        dispatch_config=config.greenheart_config["realtime_simulation"]["dispatch"],
    )

    hopp_results = he_hopp.run_hopp(
        hi,
        project_lifetime=config.greenheart_config["project_parameters"][
            "project_lifetime"
        ],
        verbose=config.verbose,
    )
    return simulator, dispatcher, hopp_results


@pytest.fixture(scope="module")
def full_system_RTS_run():
    rts, dispatcher, hopp_results = make_realtime_simulator()
    rts.simulate(dispatcher, hopp_results)
    yield rts

@pytest.fixture(scope="module")
def simple_system_RTS_run():
    rts, dispatcher, hopp_results = make_realtime_simulator()
    rts.simulate(dispatcher, hopp_results)
    yield rts


def test_RTS_full_system_runs(full_system_RTS_run):

    # Shouldn't raise
    pass


def test_RTS_simple_system():
    pass
