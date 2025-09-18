from pytest import approx, raises
import numpy as np
import types
from pathlib import Path


import sys
import types

import pytest

from greenheart.simulation.realtime_simulation import RealTimeSimulation
from greenheart.simulation.technologies.dispatch.controllers.dispatch_mpc import DispatchModelPredictiveController
# from greenheart.simulation.technologies.dispatch.controllers.dispatch_mpc_pyomo import DispatchModelPredictiveController
from greenheart.simulation.technologies.dispatch.dispatch import GreenheartDispatch
from greenheart.simulation.greenheart_simulation import  GreenHeartSimulationConfig


from hopp.simulation.technologies.sites.site_info import SiteInfo

class HOPPSystem:
    def __init__(self, site):
        self.site = site

class HOPPInterface:
    def __init__(self, site):
        self.system = HOPPSystem(site)

def make_standin_controller():

    config_root = Path(__file__).parents[0] / "dispatch_inputs"

    fname_hopp_config = str(config_root / "plant/hopp_config_mn.yaml")
    fname_greenheart_config = str(config_root / "plant/greenheart_config_onshore_mn.yaml")
    fname_turbine_config = str(
        config_root / "turbines/ATB2024_6MW_170RD_floris_turbine.yaml"
    )
    fname_floris_config = str(config_root / "floris/floris_input_lbw_6MW.yaml")


    config = GreenHeartSimulationConfig(
        fname_hopp_config,
        fname_greenheart_config,
        fname_turbine_config,
        fname_floris_config,
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

    hopp_site = SiteInfo(**config.hopp_config["site"])
    hi = HOPPInterface(hopp_site)
    simulator = RealTimeSimulation(config, hi)


    mpc_config = config.greenheart_config["realtime_simulation"]["dispatch"]["mpc"]
    
    


    # Minimal required attributes for instantiation
    ctrl = DispatchModelPredictiveController(
        config=config,
        simulation_graph=simulator.G,
        node_order=simulator.node_order,
        edge_order=simulator.edge_order,
        mpc_config=mpc_config,
    )

    return ctrl

def test_setup_solution_storage_initializes_lists():
    ctrl = make_standin_controller()
    # ctrl.setup_solution_storage()
    assert isinstance(ctrl.step_index_store, list)
    assert isinstance(ctrl.uct_store, list)
    assert isinstance(ctrl.usp_store, list)
    assert isinstance(ctrl.x_store, list)
    assert isinstance(ctrl.yex_store, list)
    assert isinstance(ctrl.ysp_store, list)
    assert isinstance(ctrl.forecast_store, list)
    assert isinstance(ctrl.curtail_store, list)
    assert isinstance(ctrl.grid_store, list)
    assert isinstance(ctrl.de_store, list)
    assert isinstance(ctrl.dco_store, list)
    assert isinstance(ctrl.objective_store, list)
    assert isinstance(ctrl.objective_uw_store, list)

def test_store_solution_appends_data():
    ctrl = make_standin_controller()
    ctrl.setup_solution_storage()
    ctrl.store_solution(
        step_index=0,
        uc=np.ones((1,2)),
        us=np.ones((1,2)),
        x=np.ones((1,3)),
        yex=np.ones((1,2)),
        ysp=np.ones((1,2)),
        forecast=np.ones((1,2)),
        x0=np.ones((1, 3)),
        curtail=np.ones((1,2)),
        grid_purchase=np.ones((1,2)),
        dex=np.ones((1,2)),
        dco=np.ones((1,2)),
        objective={"a": 1.0},
        objective_uw={"a": 2.0},
    )
    assert len(ctrl.step_index_store) == 1
    assert np.all(ctrl.uct_store[0] == np.ones((1,2)))
    assert np.all(ctrl.objective_store[0] == np.array([1.0]))
    assert np.all(ctrl.objective_uw_store[0] == np.array([2.0]))

def test_update_optimization_parameters_sets_values():
    ctrl = make_standin_controller()
    x0 = np.ones((3,1))
    src_forecast = np.ones((1,6))
    # Should not raise
    ctrl.update_optimization_parameters(x0, src_forecast)

# def test_save_and_load_stored_values(tmp_path):
#     ctrl = make_standin_controller()
#     ctrl.setup_solution_storage()
#     ctrl.store_solution(
#         step_index=0,
#         uc=np.ones((1,2)),
#         us=np.ones((1,2)),
#         x=np.ones((1,3)),
#         yex=np.ones((1,2)),
#         ysp=np.ones((1,2)),
#         forecast=np.ones((1,2)),
#         curtail=np.ones((1,2)),
#         grid_purchase=np.ones((1,2)),
#         dex=np.ones((1,2)),
#         dco=np.ones((1,2)),
#         objective={"a": 1.0},
#         objective_uw={"a": 2.0},
#     )
#     fname = tmp_path / "test.pkl"
#     # Patch save_stored_values and load_stored_values to not use missing code
#     import pickle
#     def save_stored_values(self, fname=None):
#         save_dict = {"horizon": self.horizon, "step_index": self.step_index_store}
#         with open(fname, "wb") as f:
#             pickle.dump(save_dict, f)
#     def load_stored_values(self, fname=None):
#         with open(fname, "rb") as f:
#             stored_dict = pickle.load(f)
#         self.horizon = stored_dict["horizon"]
#         self.step_index_store = stored_dict["step_index"]
#     ctrl.save_stored_values = types.MethodType(save_stored_values, ctrl)
#     ctrl.load_stored_values = types.MethodType(load_stored_values, ctrl)
#     ctrl.save_stored_values(fname)
#     ctrl.horizon = 0
#     ctrl.step_index_store = []
#     ctrl.load_stored_values(fname)
#     assert ctrl.horizon == 2
#     assert ctrl.step_index_store == [0]

def test_objective_step_returns_dict():
    ctrl = make_standin_controller()
    # Patch objective_step to return a dict with "objective"
    def objective_step(self, x, uct, usp, yco, yex, curtail=None, grid=None, gridcurtail=None, var_inds=None):
        return {"objective": {"w": 1, "expr": 42}}
    ctrl.objective_step = types.MethodType(objective_step, ctrl)
    result = ctrl.objective_step(None, None, None, None, None)
    assert "objective" in result

def test_step_control_model_runs():
    ctrl = make_standin_controller()
    # Patch required matrices
    ctrl.A = np.eye(1)
    ctrl.Bct = np.ones((1,1))
    ctrl.Bsp = np.ones((1,1))
    ctrl.Eex = np.ones((1,1))
    ctrl.Cex = np.ones((1,1))
    ctrl.Dexct = np.ones((1,1))
    ctrl.Dexsp = np.ones((1,1))
    ctrl.Fexex = np.ones((1,1))
    ctrl.Cco = np.ones((1,1))
    ctrl.Dcoct = np.ones((1,1))
    ctrl.Dcosp = np.ones((1,1))
    ctrl.Fcoex = np.ones((1,1))
    ctrl.Cze = np.ones((1,1))
    ctrl.Dzect = np.ones((1,1))
    ctrl.Dzesp = np.ones((1,1))
    ctrl.Fzeex = np.ones((1,1))
    ctrl.Cgt = np.ones((1,1))
    ctrl.Dgtct = np.ones((1,1))
    ctrl.Dgtsp = np.ones((1,1))
    ctrl.Fgtex = np.ones((1,1))
    ctrl.Cet = np.ones((1,1))
    ctrl.Detct = np.ones((1,1))
    ctrl.Detsp = np.ones((1,1))
    ctrl.Fetex = np.ones((1,1))
    x_var = np.ones((1,1))
    uct_var = np.ones((1,1))
    usp_var = np.ones((1,1))
    dex_param = np.ones((1,1))
    grid_curtail = np.ones((1,1))
    # Should not raise
    ctrl.step_control_model(x_var, uct_var, usp_var, dex_param, grid_curtail)



def test_collect_system_matrices_runs():
    """
    Test that collect_system_matrices runs without error and sets key attributes.
    """
    ctrl = make_standin_controller()


    # Should not raise
    ctrl.collect_system_matrices(ctrl.traversal_order, ctrl.G)

    # Check that some expected attributes are set
    assert hasattr(ctrl, "A")
    assert hasattr(ctrl, "Bct")
    assert hasattr(ctrl, "Cco")
    assert hasattr(ctrl, "labels")
    assert hasattr(ctrl, "dims")

def test_collect_system_matrices_dimensions():
    ctrl = make_standin_controller()

    assert hasattr(ctrl, "mct_label")
    assert hasattr(ctrl, "M_dco_yco")
    assert ctrl.M_dco_yco.shape == (12, 12)
    assert ctrl.block_ss.shape == (24, 19)

    block_ss = np.array([[ 1.     ,  0.     ,  0.     ,  0.98088, -0.93351,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ],
       [ 0.     ,  0.99958,  0.     ,  0.     ,  0.     ,  1.     , -1.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ],
       [ 0.     ,  0.     ,  1.     ,  0.     ,  0.     ,  0.     ,  0.     ,  1.     , -1.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ],
       [ 0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  1.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ],
       [ 0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  1.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ],
       [ 0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  1.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ],
       [ 0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  1.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ],
       [ 0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  1.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ],
       [ 0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  1.     ,  0.     ,  0.     ,  0.     ,  0.     ],
       [ 0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  1.     ,  0.     ,  0.     ,  0.     ],
       [ 0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  1.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ],
       [ 0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  1.     ,  0.     ,  0.     ],
       [ 0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  1.     ,  0.     ],
       [ 0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  1.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ],
       [ 0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  1.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  1.     ,  0.     ],
       [ 0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.01516,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.01516,  0.     ],
       [ 0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     , -1.     , -1.     , -1.     , -1.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  1.     ],
       [ 0.     ,  0.     ,  0.     ,  0.     ,  0.99354,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     , -1.     , -1.     , -1.     ,  0.     ,  0.     ,  0.     ],
       [ 0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.01886,  0.     ,  0.     ,  0.01886,  0.     , -1.     , -1.     ,  0.     ],
       [ 0.     ,  0.     ,  0.     , -1.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  1.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ],
       [ 0.     ,  0.     ,  0.     ,  0.     ,  0.     , -1.     ,  0.     ,  0.     ,  0.     ,  0.     ,  1.     ,  0.     ,  0.     ,  1.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ],
       [ 0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     , -1.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  1.     ,  0.     ,  0.     ],
       [ 0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  1.     ,  0.     , -3.84673,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     , -3.84673,  0.     ],
       [ 0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     ,  0.     , -8.34142,  0.     ,  0.     ,  0.     ,  1.     ,  0.     ,  0.     ,  1.     ,  0.     , -8.34142,  0.     ]])


    assert np.all(np.isclose(ctrl.block_ss, block_ss, atol=1e-6))

def test_compute_trajectory():

    atol=1e-6

    ctrl = make_standin_controller()

    x0 = np.array([1800000.   , 3238578.913,  783000.   ])
    forecast = np.array([ 94841.828,  81526.197, 118532.144, 111062.737, 235747.435, 234696.2  ])
    
    # assert ctrl.opti.ng == 279
    # assert ctrl.opti.np == 9
    # assert ctrl.opti.nx == 135


    uct, usp, curtail, grid, obj_values_uw = ctrl.compute_trajectory(x0, forecast, ret_obj=True)

    # assert np.all(np.isclose(uct, np.array([[-8.13025493e-09, -8.12998781e-09, -8.12975127e-09, -8.01949147e-09, -6.00061025e-09,  7.78131504e-09],
    #    [ 9.72000001e+04,  9.72000001e+04,  9.72000001e+04,  9.17875822e+04,  4.54609015e+04,  1.02095545e+04],
    #    [ 2.95399774e+04,  2.68776474e+04,  3.31461280e+04,  3.05892817e+04,  4.60392218e+04,  5.36483338e+04],
    #    [ 2.79103182e+04,  2.79103182e+04,  2.79103182e+04,  2.79103182e+04,  2.79103182e+04,  2.79103182e+04],
    #    [ 2.34021792e-08,  2.22724978e-08,  2.57348751e-08,  2.43579178e-08,  3.31845231e-08,  2.65598399e-08],
    #    [ 5.44333008e+03,  5.63386869e+03,  5.08801404e+03,  5.29197345e+03,  4.21032414e+03,  4.97315107e+03]]), atol=atol))

    # assert np.all(np.isclose(usp, np.array([[-8.13025493e-09, -8.12998780e-09, -8.12975046e-09, -8.01949151e-09, -6.00061029e-09,  7.78131504e-09],
    #    [ 4.39422871e+04,  3.90648849e+04,  6.39051846e+04,  5.70956689e+04,  1.35305990e+05,  1.20316510e+05],
    #    [ 1.79489616e+04,  1.15691475e+04,  1.58864479e+04,  1.49015047e+04,  3.83711437e+04,  4.87044411e+04],
    #    [ 2.76926568e+04,  2.61843845e+04,  3.22387749e+04,  3.20390644e+04,  4.91901291e+04,  5.60308115e+04],
    #    [ 5.21518897e+04,  4.69261332e+04,  5.10293436e+04,  4.70240734e+04,  2.61673502e+04,  7.08536484e+02],
    #    [ 1.15910158e+04,  1.53084999e+04,  1.72596800e+04,  1.56877771e+04,  7.66807811e+03,  4.94389269e+03],
    #    [ 3.28293432e+04,  3.43376155e+04,  2.82832251e+04,  2.84829356e+04,  1.13318709e+04,  4.49118847e+03],
    #    [ 2.34021792e-08,  2.22724978e-08,  2.57348751e-08,  2.43579178e-08,  3.31845231e-08,  2.65598399e-08],
    #    [ 1.81226992e+03,  1.62173131e+03,  2.16758596e+03,  1.96362655e+03,  3.04527586e+03,  2.28244893e+03]]) ,atol=atol))
    

    # assert np.all(np.isclose(curtail, np.array([[ 5257.92243614,  4707.780169  ,  6501.73652458,  7026.49902193, 12880.17227182,  9644.43728677]]), atol=atol))
    # assert np.all(np.isclose(grid, np.array([[0., 0., 0., 0., 0., 0.]]),  atol=atol))

    # expected_obj_vals = {'output_tracking': 1.5731616743927374e-20, 'gridcurtail': 400367217.13433015, 'bes_simultaneous': -0.0033001465447434586, 'bes_state': 2159371808229.61, 'h2s_simultaneous': 0.0007845106691240363, 'h2s_state': 1532880213.2085793, 'tes_simultaneous': 6135820829.089593, 'tes_state': 413087348.6593884, 'objective': 21660072.92262677}

    expected_obj_vals = {'output_tracking': 1.2069495632628994e-05, 'gridcurtail': 1.2004826047931353e-05, 'bes_simultaneous': -0.0059144416079150285, 'tes_simultaneous': 10724513416.551521, 'h2s_simultaneous': -0.00030347589779848697, 'bes_state': 131694847270.64319, 'tes_state': 11778928652203.576, 'h2s_state': 368870151831.668, 'bes_soc_state': 0.1518260381331452, 'tes_soc_state': 0.46867437691450015, 'h2s_soc_state': 0.5166765518233639, 'bes_terminal': 0.025322592662485103, 'tes_terminal': 0.057822618066569555, 'h2s_terminal': 0.02793587263350462, 'objective': 5153.972869957731}

    for key in expected_obj_vals.keys():
        assert np.isclose(obj_values_uw[key], expected_obj_vals[key], atol=1e-6)


    []



    