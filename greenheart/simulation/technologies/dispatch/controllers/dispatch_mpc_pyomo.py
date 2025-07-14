import numpy as np
import scipy
import networkx as nx
import matplotlib.pyplot as plt
import casadi as ca
import pyomo.environ as pyo
import pprint
import sys
from io import StringIO
import pickle

import time

from hopp.utilities import load_yaml


class DispatchModelPredictiveController:

    step_index_store: list
    uct_store: list
    usp_store: list
    dco_store: list

    def __init__(
        self,
        config,
        simulation_graph,
        saved_state=None,
        node_order=None,
        edge_order=None,
        mpc_config=None,
        p_opts = None,
        s_opts = None,
        debug_mode=False,
    ):

        # Option flags
        self.allow_curtail_forecast = True
        self.allow_grid_purchase = True
        self.include_edges = True
        self.use_sparsity_constraint = False
        self.use_config_weights = True
        self.debug_mode = debug_mode
        self.warm_start_with_previous_solution = True
        self.grid_curtail_mod = True
        self.use_NL_electrolzyer = False
        self.NL_EL_order = 1
        self.only_bounded_yco = True
        self.no_shortfall = True

        # if self.no_shortfall:
        #     print(f"{self.no_shortfall = }")

        # if self.use_NL_electrolzyer:
        #     print(f"{self.use_NL_electrolzyer = }, {self.NL_EL_order = }")

        if p_opts is None:
            self.p_opts = {"print_time": False, "verbose": False}
        else:
            self.p_opts = p_opts

        if s_opts is None:

            self.s_opts = {
                "print_level": 0,
                "compl_inf_tol": 1e-3,
                # "linear_solver": "ma27",
                # "max_iter": 10000,
                # "tol": 1e-3
                # "jac_c_constant": "yes",
                # "jac_d_constant": "yes",
                # "acceptable_compl_inf_tol": 0.5,
                # "print_user_options": "yes",
                # "print_options_documentation": "yes",
                # "print_timing_statistics": "yes"
            }
        else:
            self.s_opts = s_opts

        if self.debug_mode:
            self.load_state_for_debug(saved_state)
            self.use_saved_solution = False
            self.setup_optimization()
            self.setup_solution_storage()
        else:

            system_graph = load_yaml(
                config.greenheart_config["realtime_simulation"]["system"][
                    "system_graph_config"
                ]
            )

            self.config = config

            nodes = system_graph["traversal_order"]
            traversal_order = system_graph["traversal_order"]

            self.traversal_order = traversal_order
            self.node_order = node_order
            self.edge_order = edge_order

            if mpc_config is not None:
                self.horizon = mpc_config["horizon"]
            else:
                self.horizon = 5
            self.G = simulation_graph

            if "reference" in mpc_config:
                self.reference = mpc_config["reference"]
            else:
                # ref_steel = 45.48e3
                # ref_steel = 50
                # ref_steel = 35
                # ref_steel = 170.3413
                ref_steel = 165
                self.reference = ref_steel

            if "weights" in mpc_config:
                self.use_config_weights = True
                self.weights = mpc_config["weights"]
            if "terms" in mpc_config:
                self.term_keys = mpc_config["terms"]

            if "battery" in self.node_order:
                if "references" in mpc_config:
                    bes_soc_ref = mpc_config["references"]["bes"]
                else:
                    bes_soc_ref = 0.7   
                self.ref_bes_state = (
                    bes_soc_ref
                    * simulation_graph.nodes["battery"]["ionode"].model.max_capacity_kWh
                )
                self.weight_bes_state = 1e-4 / self.ref_bes_state

            if "hydrogen_storage" in self.node_order:
                if "references" in mpc_config:
                    h2s_soc_ref = mpc_config["references"]["h2s"]
                else:
                    h2s_soc_ref = 0.7   
                self.ref_h2s_state = (
                    h2s_soc_ref
                    * simulation_graph.nodes["hydrogen_storage"][
                        "ionode"
                    ].model.max_capacity_kg
                )
                self.weight_h2s_state = 1e-1 / self.ref_h2s_state

            if "thermal_energy_storage" in self.node_order:
                if "references" in mpc_config:
                    tes_soc_ref = mpc_config["references"]["tes"]
                else:
                    tes_soc_ref = 0.7                    
                self.ref_tes_state = (
                    tes_soc_ref
                    * simulation_graph.nodes["thermal_energy_storage"][
                        "ionode"
                    ].model.H_capacity_kWh
                )
                self.weight_tes_state = 1e-4 / self.ref_tes_state

            self.use_saved_solution = (
                "use_saved_solution"
                in config.greenheart_config["realtime_simulation"]["dispatch"]["mpc"]
            )
            if self.use_saved_solution:
                self.load_stored_values(
                    config.greenheart_config["realtime_simulation"]["dispatch"]["mpc"][
                        "use_saved_solution"
                    ]
                )

            # self.build_control_model(traversal_order, simulation_graph)
            self.collect_system_matrices(traversal_order, simulation_graph)
            self.setup_optimization()
            self.setup_solution_storage()
        self.curtail_storage = np.zeros(8760 + self.horizon)

        if self.horizon == 1:
            self.warm_start_with_previous_solution = False

        self.bad_solve_count = 0
        self.bad_solve_step = []
        self.bad_solve_violation = []
        self.prev_sol = None

    def setup_solution_storage(self):
        self.step_index_store = []
        self.uct_store = []
        self.usp_store = []
        self.x_store = []
        self.yex_store = []
        self.ysp_store = []
        self.forecast_store = []
        self.curtail_store = []
        self.grid_store = []
        self.de_store = []
        self.dco_store = []
        self.objective_store = []
        self.objective_uw_store = []

    def store_solution(
        self,
        step_index,
        uc,
        us,
        x,
        yex,
        ysp,
        forecast,
        curtail,
        grid_purchase,
        dex,
        dco,
        objective,
        objective_uw,
    ):
        self.step_index_store.append(step_index)
        self.uct_store.append(np.atleast_2d(uc))
        self.usp_store.append(np.atleast_2d(us))
        self.x_store.append(np.atleast_2d(x))
        self.yex_store.append(np.atleast_2d(yex))
        self.ysp_store.append(np.atleast_2d(ysp))
        self.forecast_store.append(np.atleast_2d(forecast))
        self.curtail_store.append(np.atleast_2d(curtail))
        self.grid_store.append(np.atleast_2d(grid_purchase))
        self.de_store.append(np.atleast_2d(dex))
        self.dco_store.append(np.atleast_2d(dco))
        self.objective_store.append(np.array(list(objective.values())))
        self.objective_uw_store.append(np.array(list(objective_uw.values())))

    def save_stored_values(self, fname=None):
        save_dict = {
            "horizon": self.horizon,
            "step_index": self.step_index_store,
            "uct": self.uct_store,
            "usp": self.usp_store,
            "x": self.x_store,
            "yex": self.yex_store,
            "ysp": self.ysp_store,
            "forecast": self.forecast_store,
            "curtail": self.curtail_store,
            "grid": self.grid_store,
            "de": self.de_store,
            "dco": self.dco_store,
            "objective": self.objective_store,
            "objective_uw": self.objective_uw_store,
        }

        with open(fname, "wb") as f:
            pickle.dump(save_dict, f)

    def load_stored_values(self, fname=None):

        with open(fname, "rb") as f:
            stored_dict = pickle.load(f)

        for key in stored_dict.keys():
            setattr(self, f"{key}_saved", stored_dict[key])

    def setup_optimization(self):
        # Create abstract model
        model = pyo.ConcreteModel()
        solver = pyo.SolverFactory("ipopt", options=self.s_opts)
        self.solver = solver

        # Setup sets

        # Time steps
        model.k = pyo.Set(dimen=1, doc="time step k", initialize = np.arange(0, self.horizon, 1), ordered=True)
        model.k_x = pyo.Set(dimen=1, doc="time step for x k", initialize = np.arange(0, self.horizon + 1, 1), ordered=True)

        model.n_set = pyo.Set(dimen=1, doc="states", initialize = np.arange(0, self.n), ordered=True)

        model.mct_set = pyo.Set(dimen=1, doc="inputs, control", initialize = np.arange(0, self.mct), ordered=True)
        model.msp_set = pyo.Set(dimen=1, doc="inputs, splitting", initialize = np.arange(0, self.msp), ordered=True)

        model.pex_set = pyo.Set(dimen=1, doc="outputs, external", initialize = np.arange(0, self.pex), ordered=True)
        model.pco_set = pyo.Set(dimen=1, doc="outputs, coupling", initialize = np.arange(0, self.pco), ordered=True)
        model.pco_bd_set = pyo.Set(dimen=1, doc="outputs, coupling", initialize = self.yco_ub_ind, ordered=True)
        model.pgt_set = pyo.Set(dimen=1, doc="outputs, greater than constraint", initialize = np.arange(0, self.pgt), ordered=True)
        model.pet_set = pyo.Set(dimen=1, doc="outputs, equal to constraint", initialize = np.arange(0, self.pet), ordered=True)
        model.pze_set = pyo.Set(dimen=1, doc="outputs, equal to zero", initialize = np.arange(0, self.pze), ordered=True)

        model.oco_set = pyo.Set(dimen=1, doc="disturbances, coupling", initialize = np.arange(0, self.oco), ordered=True)
        model.oex_set = pyo.Set(dimen=1, doc="disturbances, external", initialize = np.arange(0, self.oex), ordered=True)

        def init_zero(model, i, j=None):
            return 0


        # Parameters
        model.dex = pyo.Param(model.oex_set, model.k, domain = pyo.NonNegativeReals, initialize=init_zero, mutable=True)
        model.x0 = pyo.Param(model.n_set, domain=pyo.NonNegativeReals, initialize=init_zero, mutable=True)

        # Variables
        def xbd(model, n, k):
            return (self.bounds["x_lb"][n], self.bounds["x_ub"][n])
        model.x = pyo.Var(model.n_set, model.k_x, domain=pyo.Reals, bounds = xbd)        

    

        def uctbd(model, m, k):
            return (self.bounds["u_lb"][m], self.bounds["u_ub"][m])
        model.uct = pyo.Var(model.mct_set, model.k, domain = pyo.Reals, bounds = uctbd)

        def uspbd(model, m, k):
            return (0, None)
        model.usp = pyo.Var(model.msp_set, model.k, domain=pyo.NonNegativeReals, bounds = uspbd)


        model.yex = pyo.Var(model.pex_set, model.k, domain = pyo.NonNegativeReals)

        if self.only_bounded_yco:

            def pcobd(model, p, k):
                return (self.bounds_verbose[self.node_order[self.yco_ub_node_ind[0]]]["y_lb"], self.bounds_verbose[self.node_order[self.yco_ub_node_ind[0]]]["y_ub"])
            model.yco = pyo.Var(model.pco_bd_set * model.k, domain= pyo.Reals, bounds = pcobd)
        else:
            pass

        # Grid curtail variable
        if self.no_shortfall:
            def ucurbd(model, o, k):
                return (None, 0)
        else:
            def ucurbd(model, o, k):
                return (None, 2e6)
        model.ucur = pyo.Var(model.oex_set, model.k, domain = pyo.Reals, bounds = ucurbd )

        for k in model.k:
            model.add_component(name=f"dex_ucur_cons_{k}", val=pyo.Constraint(rule = model.ucur[0, k] >= -model.dex[0,k]))



        model.construct()

        # model.x0_con = pyo.Constraint(rule=)

        for n in model.n_set:
            model.add_component(name=f"x0_con_{n}",  val=pyo.Constraint(rule=model.x[n, 0]==model.x0[n]))

        objective = 0
        objective_terms = []


        objective_var_inds = self.get_objective_var_inds()

        for k in range(self.horizon):

            # if k == 0:
            #     x_k = model.x0
            # else:
            #     x_k = [model.x[n,k] for n in model.n_set]
            x_k = [model.x[n,k] for n in model.n_set]
            uct_k = [model.uct[m, k] for m in model.mct_set]
            usp_k = [model.usp[m, k] for m in model.msp_set]
            dex_k = [model.dex[o, k] for o in model.oex_set]
            ucur_k = model.ucur[0, k]

            xkp1, yexk, yco, yze, ygt, yet = self.step_control_model(
                x_k,
                uct_k,
                usp_k,
                dex_k,
                ucur_k,
            )
            # xkp1, yexk, yco, yze, ygt, yet = self.step_control_model(xk, model.uct[:, k], model.usp[:, k], model.dex[:, k], model.ucur)

            for n in model.n_set:
                model.add_component(f"dyn_con_i{n}_k{k}", pyo.Constraint(rule = model.x[n, k+1] == xkp1[n]))
            model.add_component(f"yex_con{k}", pyo.Constraint(rule = model.yex[0, k] == yexk[0]))

            if self.only_bounded_yco:
                for i, p in enumerate(model.pco_bd_set):
                    model.add_component(name=f"yco_con_p{p}_k{k}", val=pyo.Constraint(rule=model.yco[p, k] == yco[p]))
            else:
                pass

            for p in model.pze_set:
                model.add_component(f"pze_con_{k}_{p}", pyo.Constraint(rule = yze[p] == 0))

            for i in range(self.pgt):
                model.add_component(f"pgt_con_{k}_{i}", pyo.Constraint(rule = ygt[i] == 0))

            for i in range(self.pet):
                model.add_component(f"pet_con_{k}_{i}", pyo.Constraint(rule = yet[i] == 0))

            if self.grid_curtail_mod:
                step_obj, step_obj_terms = self.objective_step(
                    x_k,
                    uct_k,
                    usp_k,
                    yco,
                    yexk,
                    gridcurtail=ucur_k,
                    var_inds=objective_var_inds,
                )
            else:
                pass

            objective += step_obj
            objective_terms.append(step_obj_terms)

        model.obj = pyo.Objective(expr = objective, sense = "minimize")

        self.obj_terms = {}
        self.obj_terms_uw = {}

        for term in objective_terms[0].keys():
            obj_term = 0
            obj_term_uw = 0
            for i in range(self.horizon):
                obj_term += (
                    objective_terms[i][term]["w"] * objective_terms[i][term]["expr"]
                )
                obj_term_uw += objective_terms[i][term]["expr"]

            self.obj_terms.update({term: obj_term})
            self.obj_terms_uw.update({term: obj_term_uw})

        self.model = model
        []
        
    def matmul(self, A, x):
        y = []
        for i in range(A.shape[0]):
            row = 0
            for j in range(A.shape[1]):
                row += A[i, j] * x[j]
            y.append(row)
        return(y)
    
    def vecsum(self, vecs):
        sumvec = []
        for i in range(len(vecs[0])):
            vsum = 0
            for j in range(len(vecs)):
                vsum += vecs[j][i]
            sumvec.append(vsum)
        return sumvec

    def get_objective_var_inds(self):
        def find_index(label, index_list):
            indices = [i for i in range(len(index_list)) if label in index_list[i]]
            # print(indices)
            assert len(indices) == 1
            return indices[0]

        objective_var_inds = {}

        if "battery" in self.node_order:
            objective_var_inds.update(
                {
                    "uct_charge_bes": find_index("uct 0 battery", self.mct_label),
                    "uct_discharge_bes": find_index("uct 1 battery", self.mct_label),
                    "x_bes": find_index("x 0 battery", self.n_label),
                }
            )

        if "hydrogen_storage" in self.node_order:
            objective_var_inds.update(
                {
                    "uct_charge_h2s": find_index(
                        "uct 0 hydrogen_storage", self.mct_label
                    ),
                    "uct_discharge_h2s": find_index(
                        "uct 1 hydrogen_storage", self.mct_label
                    ),
                    "x_h2s": find_index("x 0 hydrogen_storage", self.n_label),
                }
            )

        if "thermal_energy_storage" in self.node_order:
            objective_var_inds.update(
                {
                    "uct_charge_tes": find_index(
                        "uct 0 thermal_energy_storage", self.mct_label
                    ),
                    "uct_discharge_tes": find_index(
                        "uct 1 thermal_energy_storage", self.mct_label
                    ),
                    "x_tes": find_index("x 0 thermal_energy_storage", self.n_label),
                }
            )
        return objective_var_inds
            
    def step_control_model(self, x_var, uct_var, usp_var, dex_param, grid_curtail):

        if self.use_NL_electrolzyer:
            return self.step_control_model_NL(x_var, uct_var, usp_var, dex_param, grid_curtail)

        xkp1 = self.vecsum([
            self.matmul(self.A, x_var),
            self.matmul(self.Bct, uct_var),
            self.matmul(self.Bsp, usp_var),
            self.matmul(self.Eex, [(dex_param[0] + grid_curtail)]),
        ])
        # external outputs
        yexk = self.vecsum([
            self.matmul(self.Cex, x_var),
            self.matmul(self.Dexct, uct_var),
            self.matmul(self.Dexsp, usp_var),
            self.matmul(self.Fexex, [(dex_param[0] + grid_curtail)]),
        ])

        # coupling outputs
        yco = self.vecsum([
            self.matmul(self.Cco, x_var),
            self.matmul(self.Dcoct, uct_var),
            self.matmul(self.Dcosp, usp_var),
            self.matmul(self.Fcoex, [(dex_param[0] + grid_curtail)]),
        ])

        # Splitting constraint zero outputs
        yze = self.vecsum([
            self.matmul(self.Cze, x_var),
            self.matmul(self.Dzect, uct_var),
            self.matmul(self.Dzesp, usp_var),
            self.matmul(self.Fzeex, [(dex_param[0] + grid_curtail)]),
        ])

        # greater than 0 constraint outputs
        ygt = self.vecsum([
            self.matmul(self.Cgt, x_var),
            self.matmul(self.Dgtct, uct_var),
            self.matmul(self.Dgtsp, usp_var),
            self.matmul(self.Fgtex, [(dex_param[0] + grid_curtail)]),
        ])

        # equal to 0 constraint outputs
        yet = self.vecsum([
            self.matmul(self.Cet, x_var),
            self.matmul(self.Detct, uct_var),
            self.matmul(self.Detsp, usp_var),
            self.matmul(self.Fetex, [(dex_param[0] + grid_curtail)]),
        ])
        return xkp1, yexk, yco, yze, ygt, yet

    def nonlinear_block(self, X):

        P_el = np.ones((1, 2)) @ X

        # Hacky for electrolyzer only right now

        if self.NL_EL_order == 1:

            # 1st order fit
            popt = np.array([0.01885931])
            Y = popt[0] * P_el

        elif self.NL_EL_order == 2:
            # 2nd order fit
            popt = np.array([-2.28481418e-09,  2.08294629e-02])
            Y = popt[0] * P_el ** 2 + popt[1] * P_el
        elif self.NL_EL_order == 3:
            # 3rd order fit
            popt = np.array([ 1.28840632e-15, -4.33591254e-09,  2.15782895e-02])
            Y = popt[0] * P_el ** 3 + popt[1] * P_el**2 + popt[2] * P_el

        return Y

    def step_control_model_NL(self, x_var, uct_var, usp_var, dex_param, grid_curtail):

        X_block = ca.vertcat(x_var, uct_var, usp_var, dex_param+ grid_curtail)
        X_block_li = X_block[self.cols_li]
        X_block_nl = X_block[self.cols_nl]

        ss_lili = self.block_ss[self.rows_li, self.cols_li]
        ss_linl = self.block_ss[self.rows_li, self.cols_nl]
        ss_nlli = self.block_ss[self.rows_nl, self.cols_li]
        ss_nlnl = self.block_ss[self.rows_nl, self.cols_nl]

        Y_block_li = ss_lili @ X_block_li + ss_linl @ X_block_nl
        if self.use_NL_electrolzyer:
            Y_block_nl = ss_nlli @ X_block_li + self.nonlinear_block(X_block_nl)
        else:
            Y_block_nl = ss_nlli @ X_block_li + ss_nlnl @ X_block_nl

        Y_block = ca.MX(len(self.rows_li) + len(self.rows_nl), 1)
        Y_block[self.rows_li, :] = Y_block_li
        Y_block[self.rows_nl, :] = Y_block_nl

        # Y_block = self.block_ss @ ca.vertcat(x_var, uct_var, usp_var, dex_param+ grid_curtail)

        row_inds = [self.n, self.pco, self.pex, self.pze, self.pgt, self.pet]
        previous = 0
        y_parts = []
        for rows in row_inds: 
            y_parts.append(Y_block[previous: previous + rows])
            previous += rows

        xkp1, yco, yexk, yze, ygt, yet = y_parts[0], y_parts[1], y_parts[2], y_parts[3], y_parts[4], y_parts[5] 
        return xkp1, yexk, yco, yze, ygt, yet

    def objective_step(self, x, uct, usp, yco, yex, curtail=None, grid=None, gridcurtail=None, var_inds=None):

        # =============================================================================
        # ==                                                                         ==
        # ==                                Objective                                ==
        # ==                                                                         ==
        # =============================================================================
        obj_terms = {}

        if self.grid_curtail_mod:
            term_keys = ["output_tracking","gridcurtail"]
        else:
            term_keys = [
                "output_tracking",
                "curtail",
                "grid_purchase",
            ]

        # output_tracking = (ref_steel - yex) ** 2
        obj_terms.update(
            {"output_tracking": {"w": 1e9, "expr": (self.reference - yex[0]) ** 2}}
        )

        if self.grid_curtail_mod:
            obj_terms.update({"gridcurtail": {"w": 1e-4, "expr": gridcurtail**2}})
        else:
            obj_terms.update({"curtail": {"w": 1e-4, "expr": curtail**2}})
            obj_terms.update({"grid_purchase": {"w": 1e-4, "expr": grid**2}})
            obj_terms.update({"gen_simultaneous": {"w": 1, "expr": curtail * grid}})

        if "battery" in self.node_order:
            simu = uct[var_inds["uct_charge_bes"]] * uct[var_inds["uct_discharge_bes"]]
            obj_terms.update({"bes_simultaneous": {"w": 1e0, "expr": simu}})

            state = (x[var_inds["x_bes"]] - self.ref_bes_state) ** 2
            obj_terms.update({"bes_state": {"w": self.weight_bes_state, "expr": state}})

            term_keys.append("bes_simultaneous")

        if "hydrogen_storage" in self.node_order:
            simu = uct[var_inds["uct_charge_h2s"]] * uct[var_inds["uct_discharge_h2s"]]
            obj_terms.update({"h2s_simultaneous": {"w": 1e0, "expr": simu}})

            state = (x[var_inds["x_h2s"]] - self.ref_h2s_state) ** 2
            obj_terms.update({"h2s_state": {"w": self.weight_h2s_state, "expr": state}})

            term_keys.append("h2s_simultaneous")

        if "thermal_energy_storage" in self.node_order:
            simu = uct[var_inds["uct_charge_tes"]] * uct[var_inds["uct_discharge_tes"]]
            obj_terms.update({"tes_simultaneous": {"w": 1e0, "expr": simu}})

            state = (x[var_inds["x_tes"]] - self.ref_tes_state) ** 2
            obj_terms.update({"tes_state": {"w": self.weight_tes_state, "expr": state}})

            if self.weights["tes_simultaneous"] > 0:
                term_keys.append("tes_simultaneous")
            term_keys.append("tes_state")

        if self.use_config_weights:

            for key in self.weights.keys():
                if key in obj_terms:
                    obj_terms[key]["w"] = self.weights[key]


        if hasattr(self, "term_keys"):
            obj_term_keys = self.term_keys
        else:
            obj_term_keys = term_keys

        objective = 0
        for term in obj_term_keys:
            objective += obj_terms[term]["w"] * obj_terms[term]["expr"]

        obj_terms.update({"objective": {"w": 1, "expr": objective}})
        return objective, obj_terms

    def update_optimization_parameters(self, x0, src_forecast):

        for n in self.model.n_set:
            self.model.x0[n].set_value(x0[n])

        for k in self.model.k:
            self.model.dex[0,k].set_value(src_forecast[k])

    def compute_trajectory(self, x0, forecast, step_index=0, ret_obj=False):
        # =============================================================================
        # ==                                                                         ==
        # ==                            Compute Trajectory                           ==
        # ==                                                                         ==
        # =============================================================================

        def get_sol_value(prob:ca.Opti, var):
            val = prob.value(var)
            val = np.reshape(val, var.shape)
            return val

        if self.use_saved_solution:

            # find the right index

            save_index = [
                i
                for i in range(len(self.step_index_saved))
                if step_index == self.step_index_saved[i]
            ]
            assert len(save_index) == 1
            save_index = save_index[0]

            uct = self.uct_saved[save_index]
            usp = self.usp_saved[save_index]
            x = self.x_saved[save_index]
            yex = self.yex_saved[save_index]
            ysp = self.ysp_saved[save_index]
            saved_forecast = self.forecast_saved[save_index]
            curtail = self.curtail_saved[save_index]
            grid = self.grid_saved[save_index]
            dex = self.de_saved[save_index]
            dco = self.dco_saved[save_index]
            obj_values = self.objective_saved[save_index]
            obj_values = {
                key: obj_values[i] for i, key in enumerate(list(self.obj_terms.keys()))
            }

            obj_values_uw = self.objective_uw_saved[save_index]
            obj_values_uw = {
                key: obj_values_uw[i]
                for i, key in enumerate(list(self.obj_terms_uw.keys()))
            }

            # check saved forecast is the same as given forecast?

            self.store_solution(
                step_index=step_index,
                uc=uct,
                us=usp,
                x=x,
                yex=yex,
                ysp=ysp,
                forecast=forecast,
                curtail=curtail,
                grid_purchase=grid,
                dex=dex,
                dco=dco,
                objective=obj_values,
                objective_uw=obj_values_uw,
            )

            return uct, usp, curtail, grid

        else:

            if len(self.x_store) > 0:
                # Error between where the MPC planned for the state to be and where the measure state is
                state_error = x0 - self.x_store[-1][:, step_index - self.step_index_store[-1]]

            self.update_optimization_parameters(x0, forecast)
            if self.warm_start_with_previous_solution:
                if hasattr(self, "x_init"):  # then try to update initial guess

                #     self.opti.set_initial(self.opt_vars["uct"][:, :overlap] , self.uc_init[:, -overlap:])
                #     self.opti.set_initial(self.opt_vars["usp"][:, :overlap] , self.us_init[:, -overlap:])
                #     self.opti.set_initial(self.opt_vars["x"][:, :overlap] , self.x_init[:, -overlap:])
                #     self.opti.set_initial(self.opt_vars["yex"][:, :overlap] , self.ys_init[:, -overlap:])

                    overlap = self.horizon - (step_index - self.step_index_store[-1])
                    for k in range(overlap):
                        for n in self.model.n_set:
                            self.model.x[n, k].set_value(self.x_init[n, -overlap + k])
                        
                        for m in self.model.mct_set:
                            self.model.uct[m, k].set_value(self.uc_init[m, -overlap + k])
                    
                        for m in self.model.msp_set:
                            self.model.usp[m, k].set_value(self.us_init[m, -overlap + k])
                        
                        for p in self.model.pex_set:
                            self.model.x[p, k].set_value(self.ys_init[p, -overlap + k])



                pass    


            self.solver.solve(self.model)



            uct = self.format_output(self.model.uct)
            usp = self.format_output(self.model.usp)
            x = self.format_output(self.model.x)
            yex = self.format_output(self.model.yex)
            yco = self.format_output(self.model.yco)
            dex = self.format_output(self.model.dex)


            if self.grid_curtail_mod:
                gridcurtail = self.format_output(self.model.ucur)
                grid = np.where(gridcurtail >= 0, gridcurtail, 0)
                curtail = np.where(gridcurtail <= 0, -gridcurtail, 0)
            else:
                pass

            self.curtail_storage[step_index : step_index + self.horizon] = curtail

            self.uc_init = uct
            self.us_init = usp
            self.x_init = x
            self.ys_init = yex
            self.curtail_init = curtail

            ysp = (
                self.Cco @ x[:, :-1]
                + self.Dcoct @ uct
                + self.Dcosp @ usp
                + self.Fcoex @ (dex - curtail)
            )
            dco = self.M_dco_yco @ ysp

            ysp = np.concatenate([ysp, yex])
            obj_values = {key: pyo.value(self.obj_terms[key]) for key in self.obj_terms.keys()}
            obj_values_uw = {key: pyo.value(self.obj_terms_uw[key]) for key in self.obj_terms_uw.keys()}

            self.store_solution(
                step_index=step_index,
                uc=uct,
                us=usp,
                x=x,
                yex=yex,
                ysp=ysp,
                forecast=forecast,
                curtail=curtail,
                grid_purchase=grid,
                dex=dex,
                dco=dco,
                objective=obj_values,
                objective_uw=obj_values_uw,
            )

            # u_split = usp[:, 0]
            # if not self.debug_mode:
            #     self.save_state_for_debug(x0, forecast, step_index)

            if ret_obj: 

                return uct, usp, curtail, grid, obj_values_uw
            else:
                return uct, usp, curtail, grid
            
    def format_output(self, var):
        if not isinstance(var, dict):
            try:
                var = var.extract_values()
            except:
                AssertionError("error")

        key_list = list(var.keys())

        var_shape = np.max(np.stack(list(var.keys())), axis=0) + 1
        var_out = np.zeros(var_shape)

        for key in var.keys():
            var_out[*key] = var[key]

        return var_out


    def save_state_for_debug(self, x0, forecast, step_index):

        assert not self.debug_mode

        import datetime
        from pathlib import Path
        import json

        datetime_string = datetime.datetime.now().strftime("%Y_%m_%d--%H_%M_%S")
        dir_path = "/Users/ztully/Documents/hybrids_code/GH_scripts/greenheart_scripts/minnesota_reference_design/01-minnesota-steel/saved_data/optimization_data"
        dir = f"{dir_path}/mpcstate_{datetime_string}_step{step_index}"
        Path(dir).mkdir(parents=True, exist_ok=True)
        fpath = f"{dir}/mpc_data.json"

        js_kw = dict(ensure_ascii=True, indent=4)

        with open(fpath, "w", encoding="utf-8") as f:

            plant_SS = [
                [self.A, self.Bct, self.Bsp, self.Eex],
                [self.Cco, self.Dcoct, self.Dcosp, self.Fcoex],
                [self.Cex, self.Dexct, self.Dexsp, self.Fexex],
                [self.Cze, self.Dzect, self.Dzesp, self.Fzeex],
                [self.Cgt, self.Dgtct, self.Dgtsp, self.Fgtex],
                [self.Cet, self.Detct, self.Detsp, self.Fetex],
            ]

            save_dict = dict(
                horizon=self.horizon,
                statespace=[[mat.tolist() for mat in row] for row in plant_SS],
                x0=x0.tolist(),
                forecast=forecast.tolist(),
                step_index=step_index,
                bounds={key: self.bounds[key].tolist() for key in self.bounds.keys()},
                bounds_verbose = {node:{key: self.bounds_verbose[node][key].tolist() for key in self.bounds_verbose[node].keys()} for node in self.bounds_verbose.keys()},
                dimensions=self.dims,
                labels=self.labels,
                node_order=self.node_order,
                edge_order=self.edge_order,
                weights=self.weights,
                reference=self.reference,
                ref_bes_state=self.ref_bes_state,
                weight_bes_state=self.weight_bes_state,
                ref_h2s_state=self.ref_h2s_state,
                weight_h2s_state=self.weight_h2s_state,
                ref_tes_state=self.ref_tes_state,
                weight_tes_state=self.weight_tes_state,
                M_dco_yco=self.M_dco_yco.tolist(),
                yco_ub_ind = self.yco_ub_ind.tolist(),
                yco_ub_node_ind = self.yco_ub_node_ind.tolist(),
                cols_li = self.cols_li.tolist(),
                cols_nl = self.cols_nl.tolist(),
                rows_li = self.rows_li.tolist(),
                rows_nl = self.rows_nl.tolist(),
                block_ss = self.block_ss.tolist(),
            )

            if hasattr(self, "x_init"):  # then try to update initial guess
                save_dict.update(dict(uct_init = self.uc_init.tolist(), usp_init = self.us_init.tolist(), x_init = self.x_init.tolist(), yex_init = self.ys_init.tolist()))

                # self.opti.set_initial(self.opt_vars["uct"], self.uc_init)
                # self.opti.set_initial(self.opt_vars["usp"], self.us_init)
                # self.opti.set_initial(self.opt_vars["x"], self.x_init)
                # self.opti.set_initial(self.opt_vars["yex"], self.ys_init)

            json.dump(save_dict, f, **js_kw)

        pass

    def load_state_for_debug(self, state_dict):
        self.horizon = state_dict["horizon"]

        self.bounds = {
            key: np.array(state_dict["bounds"][key])
            for key in state_dict["bounds"].keys()
        }
        self.bounds_verbose = {node:{key: np.array(state_dict["bounds_verbose"][node][key]) for key in state_dict["bounds_verbose"][node].keys()} for node in state_dict["bounds_verbose"].keys()}

        for key in state_dict["labels"].keys():
            setattr(self, f"{key}_label", state_dict["labels"][key])

        for key in state_dict["dimensions"].keys():
            setattr(self, key, np.sum(state_dict["dimensions"][key]))

        self.node_order = state_dict["node_order"]
        self.edge_order = state_dict["edge_order"]

        self.reference = state_dict["reference"]
        self.weights = state_dict["weights"]

        self.ref_bes_state = state_dict["ref_bes_state"]
        self.weight_bes_state = state_dict["weight_bes_state"]
        self.ref_h2s_state = state_dict["ref_h2s_state"]
        self.weight_h2s_state = state_dict["weight_h2s_state"]
        self.ref_tes_state = state_dict["ref_tes_state"]
        self.weight_tes_state = state_dict["weight_tes_state"]

        combined_mat = state_dict["statespace"]

        out_dims = np.array([self.n, self.pco, self.pex, self.pze, self.pgt, self.pet])
        in_dims = np.array([self.n, self.mct, self.msp, self.oex])

        for i, row in enumerate(combined_mat):
            for j, mat in enumerate(row):
                if (out_dims[i] > 0) and (in_dims[j] > 0):
                    combined_mat[i][j] = np.array(combined_mat[i][j])
                else:
                    combined_mat[i][j] = np.zeros((out_dims[i], in_dims[j]))

        # combined_mat = [[np.array(mat) for mat in row] for row in combined_mat]

        self.A, self.Bct, self.Bsp, self.Eex = combined_mat[0]
        self.Cco, self.Dcoct, self.Dcosp, self.Fcoex = combined_mat[1]
        self.Cex, self.Dexct, self.Dexsp, self.Fexex = combined_mat[2]
        self.Cze, self.Dzect, self.Dzesp, self.Fzeex = combined_mat[3]
        self.Cgt, self.Dgtct, self.Dgtsp, self.Fgtex = combined_mat[4]
        self.Cet, self.Detct, self.Detsp, self.Fetex = combined_mat[5]

        self.M_dco_yco = np.array(state_dict["M_dco_yco"])
        self.yco_ub_ind = np.array(state_dict["yco_ub_ind"])
        self.yco_ub_node_ind = np.array(state_dict["yco_ub_node_ind"])

        self.cols_li = np.array(state_dict["cols_li"])
        self.cols_nl = np.array(state_dict["cols_nl"])
        self.rows_li = np.array(state_dict["rows_li"])
        self.rows_nl = np.array(state_dict["rows_nl"])

        self.block_ss = np.array(state_dict["block_ss"])

        []

    def plot_trajectory_generic(self, prob:ca.Opti, forecast):

        def get_sol_value(prob:ca.Opti, var):
            val = prob.value(var)
            val = np.reshape(val, var.shape)
            return val

        x_db = get_sol_value(prob, self.opt_vars["x"]) 
        uc_db = get_sol_value(prob, self.opt_vars["uct"])
        us_db = get_sol_value(prob, self.opt_vars["usp"])
        ys_db = get_sol_value(prob, self.opt_vars["yex"])
        yco_db = get_sol_value(prob, self.opt_vars["yco"])
        if self.grid_curtail_mod:
            gridcurtail = get_sol_value(prob, self.opt_vars["gridcurtail"])
            grid_db = np.where(gridcurtail >= 0, gridcurtail, 0)
            curtail_db = np.where(gridcurtail <= 0, -gridcurtail, 0)
        else:

            curtail_db = get_sol_value(prob, self.opt_vars["curtail"])
            grid_db = get_sol_value(prob, self.opt_vars["grid"])

        fig, ax = plt.subplots(
            np.max(
                [
                    uc_db.shape[0],
                    us_db.shape[0],
                    x_db.shape[0],
                    ys_db.shape[0],
                    yco_db.shape[0],
                ]
            ),
            5,
            sharex="all",
            layout="constrained",
        )

        to_plot = [x_db, uc_db, us_db, ys_db, yco_db]
        titles = [
            self.n_label,
            self.mct_label,
            self.msp_label,
            self.pex_label,
            self.pco_label,
        ]
        for i in range(len(to_plot)):
            # ax[0, i].set_title(titles[i])
            for j in range(len(to_plot[i])):
                ax[j, i].plot(to_plot[i][j, :])
                ax[j, i].set_title(titles[i][j])

                if (i == 0) or (i == 1):
                    if i == 0:
                        lb = self.bounds["x_lb"]
                        ub = self.bounds["x_ub"]
                    elif i == 1:
                        lb = self.bounds["u_lb"]
                        ub = self.bounds["u_ub"]

                    ylim = ax[j, i].get_ylim()
                    ax[j, i].axhline(lb[j], color="black", linewidth=0.75)
                    ax[j, i].axhline(ub[j], color="black", linewidth=0.75)
                    ax[j, i].set_ylim(ylim)

        ax[-1, 0].set_title("Forecast and curtail")
        ax[-1, 0].plot(forecast)
        ax[-1, 0].plot(forecast - curtail_db)
        ax[-1, 0].plot(forecast - curtail_db + grid_db)

    def plot_trajectory(self, step_index=None):

        idx = -1

        fig, ax = plt.subplots(2, 2, sharex="all", layout="constrained")

        ax[0, 0].plot(self.forecast_store[idx].T)
        ax[0, 0].fill_between(
            np.arange(0, self.horizon, 1),
            self.forecast_store[idx][0, :],
            self.forecast_store[idx][0, :] - self.curtail_store[idx][0, :],
        )

        gen_split_index = [
            i
            for i in range(self.msp)
            if self.msp_label[i].split(" ")[2] == "generation"
        ]
        start = np.zeros(self.horizon)
        time = np.arange(0, self.horizon, 1)
        for k in gen_split_index:
            stop = self.usp_store[idx][k, :]
            ax[0, 0].fill_between(
                time,
                start,
                stop,
                edgecolor=None,
                label=self.msp_label[gen_split_index[k]],
            )
            start += stop

        ax[0, 0].legend()

        pass

    def collect_system_matrices(self, traversal_order, G):
        # =============================================================================
        # ==                                                                         ==
        # ==                     Construct control model                             ==
        # ==                                                                         ==
        # =============================================================================
        dims = {
            "dims": {
                "n": [],  # number of states
                "mct": [],  # number of control inputs
                "msp": [],  # number of splitting inputs
                "m": [],  # total number of inputs
                "oex": [],  # number of external disturbances
                "oco": [],  # number of coupling disturbances
                "o": [],  # total number of disturbances
                "pex": [],  # number of external outputs
                "pco": [],  # number of coupling outputs
                "pze": [],  # number of zero output constraints (splitting)
                "pet": [],  # number of equal to zero output contraints (from cm)
                "pgt": [],  # number of greater than zero output constraints (from cm)
                "p": [],  # total number of outputs
                "pcons": [],  # total number of output constraints
            },
            "labels": {
                "n": [],
                "mct": [],
                "msp": [],
                "m": [],
                "oex": [],
                "oco": [],
                "o": [],
                "pex": [],
                "pco": [],
                "pze": [],
                "pet": [],
                "pgt": [],
                "p": [],
                "pcons": [],
            },
        }

        bounds = {
            "u_lb": [],
            "u_ub": [],
            "x_lb": [],
            "x_ub": [],
            "y_lb": [],
            "y_ub": [],
        }

        verbose_bounds = {}

        mats1 = {"A": [], "Bct": [], "Bsp": [], "Eco": [], "Eex": []}
        mats2 = {"Cex": [], "Dexct": [], "Dexsp": [], "Fexco": [], "Fexex": []}
        mats3 = {"Cco": [], "Dcoct": [], "Dcosp": [], "Fcoco": [], "Fcoex": []}
        mats4 = {"Cze": [], "Dzect": [], "Dzesp": [], "Fzeco": [], "Fzeex": []}
        mats5 = {"Cgt": [], "Dgtct": [], "Dgtsp": [], "Fgtco": [], "Fgtex": []}
        mats6 = {"Cet": [], "Detct": [], "Detsp": [], "Fetco": [], "Fetex": []}

        uct_order = {}
        usp_order = {}

        # TODO linear and nonlinear columns

        linear_cols = {"x": [], "uct": [], "usp": [], "dco": [], "dex": []}
        nonlinear_cols = {"x": [], "uct": [], "usp": [], "dco": [], "dex": []}
        linear_rows = {"x":[], "yex": [], "yco": [], "yze": [], "ygt": [], "yet":[]}
        nonlinear_rows = {"x":[], "yex": [], "yco": [], "yze": [], "ygt": [], "yet":[]}

        linear_vars = {}
        nonlinear_vars = {}

        for node in traversal_order:

            cm = G.nodes[node]["ionode"].model.control_model

            in_degree = G.nodes[node]["ionode"].in_degree
            out_degree = G.nodes[node]["ionode"].out_degree

            if out_degree > 1:
                usp_degree = out_degree
            else:
                usp_degree = 0

            # identify the component model dimensions
            n = cm.A.shape[0]

            # create state labels
            x_labels = []
            x_col_linear = []
            x_row_linear = []
            for i in range(n):

                if cm.x_linear[i]:
                    linear_str = "linear"
                else:
                    linear_str = "nonlinear"
                x_labels.append(f"x {i} {node} {linear_str}")

                x_col_linear.append(cm.x_linear[i] )
                x_row_linear.append(cm.x_linear[i] )

            mct = cm.B.shape[1]
            msp = usp_degree
            m = mct + msp

            # create controllable input label lists

            uct_indices = []

            uct_labels = []
            uct_col_linear = []
            for i in range(mct):
                uct_indices.append(int(len(uct_labels) + np.sum(dims["dims"]["mct"])))
                uct_labels.append(f"uct {i} {node}")

                uct_col_linear.append(cm.u_linear[i])

            if len(uct_indices) > 0:
                uct_order.update({node: uct_indices})

            usp_indices = []
            usp_labels = []
            usp_col_linear = []
            for i in range(usp_degree):
                usp_indices.append(int(len(usp_labels) + np.sum(dims["dims"]["msp"])))
                out_edges = list(G.out_edges(node))
                usp_labels.append(f"usp {i} {node} (to {out_edges[i][1]})")
                usp_col_linear.append(True)

            if len(usp_indices) > 0:
                usp_order.update({node: usp_indices})

            # create uncontrollable input label lists
            dex_labels = []
            dco_labels = []
            dex_col_linear = []
            dco_col_linear = []

            if G.nodes[node]["is_source"]:
                oex = cm.F.shape[1]
                assert in_degree == 1
                oco = 0

                for i in range(oex):
                    dex_labels.append(f"dex {i} {node}")
                    dex_col_linear.append(cm.d_linear[i])

            else:
                oex = 0
                oco = cm.F.shape[1] * in_degree

                for i in range(in_degree):
                    in_edges = list(G.in_edges(node))
                    dco_labels.append(f"dco {i} {node} (from {in_edges[i][0]})")
                    dco_col_linear.append(cm.d_linear[0])
                    # for j in range(cm.o):
                    #     dco_col_linear.append(cm.d_linear[j])

            o = oex + oco

            # create output label lists

            yex_labels = []
            yco_labels = []
            yze_labels = []
            yet_labels = []
            ygt_labels = []

            yex_row_linear = []
            yco_row_linear = []
            yze_row_linear = []
            yet_row_linear = []
            ygt_row_linear = []

            if G.nodes[node]["is_sink"]:
                pex = cm.C.shape[0]
                for i in range(pex):
                    yex_labels.append(f"yex {i} {node}")
                    yex_row_linear.append(cm.y_linear[i])
            else:
                pex = 0

            if usp_degree > 0:
                # Splitting node so yze constraints are needed
                pze = cm.C.shape[0]
                assert pex == 0, "sink node should not be splitting"
                pco = usp_degree

                for i in range(pze):
                    yze_labels.append(f"yze {i} {node}")
                    yze_row_linear.append(cm.y_linear[i])
            else:
                pze = 0
                pco = cm.C.shape[0] - pex

            if not G.nodes[node]["is_sink"]:
                for i in range(out_degree):
                    out_edges = list(G.out_edges(node))
                    yco_labels.append(f"yco {i} {node} (to {out_edges[i][1]})")
                    # Splitting modification means these output rows are linear
                    yco_row_linear.append(True)

            pet = cm.C_et.shape[0]
            pgt = cm.C_gt.shape[0]

            for i in range(pet):
                yet_labels.append(f"yet {i} {node}")
                # TODO double check if this is always true
                yet_row_linear.append(True)

            for i in range(pgt):
                ygt_labels.append(f"ygt {i} {node}")
                # TODO double check if this is always true
                ygt_row_linear.append(True)

            p = pex + pco
            pcons = pze + pet + pgt

            # Check the incoming edges for domain agreement
            in_edges = list(G.in_edges(node))
            disturbance_index = []
            for in_edge in in_edges:
                up_node = in_edge[0]
                up_cm = G.nodes[up_node]["ionode"].model.control_model
                up_node_output_domain = up_cm.output_domain
                disturbance_index.append(
                    np.where(
                        cm.disturbance_permutation
                        @ (cm.disturbance_domain * up_node_output_domain)
                        == 1
                    )[0]
                )
                # disturbance_index.append(
                #     np.where(cm.disturbance_domain @ up_node_output_domain == 1)[0]
                # )

            oco = len(disturbance_index)

            # assert oco == len(disturbance_index)

            dim_list = [n, mct, msp, m, oex, oco, o, pex, pco, pze, pet, pgt, p, pcons]
            labels_list = [
                x_labels,
                uct_labels,
                usp_labels,
                [],
                dex_labels,
                dco_labels,
                [],
                yex_labels,
                yco_labels,
                yze_labels,
                yet_labels,
                ygt_labels,
                [],
                [],
            ]

            for i, key in enumerate(dims["dims"].keys()):
                dims["dims"][key].append(dim_list[i])
                dims["labels"][key].append(labels_list[i])

            linear_col_list = [x_col_linear, uct_col_linear, usp_col_linear, dco_col_linear, dex_col_linear]
            linear_row_list = [x_row_linear, yex_row_linear, yco_row_linear, yze_row_linear, ygt_row_linear, yet_row_linear]

            for i, key in enumerate(linear_cols.keys()):
                linear_cols[key].append(linear_col_list[i])

            for i, key in enumerate(linear_rows.keys()):
                linear_rows[key].append(linear_row_list[i])

            # store bounds from the cm

            bounds["u_lb"].append(cm.u_lb)
            bounds["u_ub"].append(cm.u_ub)
            bounds["x_lb"].append(cm.x_lb)
            bounds["x_ub"].append(cm.x_ub)
            bounds["y_lb"].append(cm.y_lb)
            bounds["y_ub"].append(cm.y_ub)

            verbose_bounds.update(
                {
                    node: {
                        "u_lb": cm.u_lb,
                        "u_ub": cm.u_ub,
                        "x_lb": cm.x_lb,
                        "x_ub": cm.x_ub,
                        "y_lb": cm.y_lb,
                        "y_ub": cm.y_ub,
                    }
                }
            )

            # Collect the relevant matrices

            # state transition row
            A = cm.A
            Bct = cm.B
            Bsp = np.zeros((n, usp_degree))
            if G.nodes[node]["is_source"]:
                Eco = np.zeros((n, 0))
                Eex = cm.E
            else:
                Eco = np.concatenate(
                    [cm.E[:, di[0], None] for di in disturbance_index], axis=1
                )
                # Eco = np.tile(cm.E, in_degree)
                Eex = np.zeros((n, 0))

            m1 = [A, Bct, Bsp, Eco, Eex]
            for i, key in enumerate(mats1.keys()):
                mats1[key].append(m1[i])

            # external output row
            if G.nodes[node]["is_sink"]:
                Cex = cm.C
                Dexct = cm.D
                Dexsp = np.zeros((pex, msp))

                assert not G.nodes[node][
                    "is_source"
                ], "source should not be the same as sink"

                Fexco = np.concatenate(
                    [cm.F[:, di[0], None] for di in disturbance_index], axis=1
                )
                # Fexco = np.tile(cm.F, in_degree)
                Fexex = np.zeros((pex, oex))
            else:
                Cex = np.zeros((0, n))
                Dexct = np.zeros((0, mct))
                Dexsp = np.zeros((0, msp))
                Fexco = np.zeros((0, oco))
                Fexex = np.zeros((0, oex))

            m2 = [Cex, Dexct, Dexsp, Fexco, Fexex]
            for i, key in enumerate(mats2.keys()):
                mats2[key].append(m2[i])

            # coupling output row
            if G.nodes[node]["is_sink"]:
                # if it is the sink node then there should be no coupling outputs
                Cco = np.zeros((pco, n))
                Dcoct = np.zeros((pco, mct))
                Dcosp = np.zeros((pco, msp))
                Fcoco = np.zeros((pco, oco))
                Fcoex = np.zeros((pco, oex))

                # and if it is a sink node then there will be no splitting constraints

                # splitting zero constraint row
                Cze = np.zeros((pze, n))
                Dzect = np.zeros((pze, mct))
                Dzesp = np.zeros((pze, msp))
                Fzeco = np.zeros((pze, oco))
                Fzeex = np.zeros((pze, oex))

            else:

                # splitting zero constraint row
                if usp_degree > 1:
                    # not sink node but is splitting node

                    Cco = np.zeros((pco, n))
                    Dcoct = np.zeros((pco, mct))
                    Dcosp = np.eye(msp)
                    Fcoco = np.zeros((pco, oco))
                    Fcoex = np.zeros((pco, oex))

                    Cze = cm.C
                    Dzect = cm.D
                    Dzesp = -np.tile(
                        np.eye(cm.C.shape[0]), msp
                    )  # Dsp matrix is not in here but it should be okay because the splitting matrix will always be identity
                    if G.nodes[node]["is_source"]:
                        Fzeco = np.zeros((pze, oco))
                        Fzeex = cm.F
                    else:
                        Fzeco = np.tile(cm.F, in_degree)
                        Fzeex = np.zeros((pze, oex))

                else:
                    # not sink node and not splitting node

                    Cco = cm.C
                    Dcoct = cm.D
                    Dcosp = np.zeros((pco, msp))
                    if G.nodes[node]["is_source"]:
                        Fcoco = np.zeros((pco, 0))
                        Fcoex = cm.F
                    else:
                        Fcoco = np.concatenate(
                            [cm.F[:, di[0], None] for di in disturbance_index], axis=1
                        )
                        # Fcoco = np.tile(cm.F, in_degree)
                        Fcoex = np.zeros((pco, 0))

                    Cze = np.zeros((pze, n))
                    Dzect = np.zeros((pze, mct))
                    Dzesp = np.zeros((pze, msp))
                    Fzeco = np.zeros((pze, oco))
                    Fzeex = np.zeros((pze, oex))

            m3 = [Cco, Dcoct, Dcosp, Fcoco, Fcoex]
            for i, key in enumerate(mats3.keys()):
                mats3[key].append(m3[i])

            m4 = [Cze, Dzect, Dzesp, Fzeco, Fzeex]
            for i, key in enumerate(mats4.keys()):
                mats4[key].append(m4[i])

            # greater than zero constraint row
            Cgt = cm.C_gt
            Dgtct = cm.D_gt
            Dgtsp = np.zeros((pgt, msp))
            if G.nodes[node]["is_source"]:
                Fgtco = np.zeros((pgt, oco))
                Fgtex = cm.F_gt
            else:
                Fgtco = np.concatenate(
                    [cm.F_gt[:, di[0], None] for di in disturbance_index], axis=1
                )
                # Fgtco = np.tile(cm.F_gt, in_degree)
                Fgtex = np.zeros((pgt, oex))

            m5 = [Cgt, Dgtct, Dgtsp, Fgtco, Fgtex]
            for i, key in enumerate(mats5.keys()):
                mats5[key].append(m5[i])

            # equal to zero contraint row
            Cet = cm.C_et
            Detct = cm.D_et
            Detsp = np.zeros((pet, msp))
            if G.nodes[node]["is_source"]:
                Fetco = np.zeros((pet, oco))
                Fetex = cm.F_et
            else:
                Fetco = np.concatenate(
                    [cm.F_et[:, di[0], None] for di in disturbance_index], axis=1
                )
                # Fetco = np.tile(cm.F_et, in_degree)
                Fetex = np.zeros((pet, oex))

            m6 = [Cet, Detct, Detsp, Fetco, Fetex]
            for i, key in enumerate(mats6.keys()):
                mats6[key].append(m6[i])

            []

        A, Bct, Bsp, Eco, Eex = (
            scipy.linalg.block_diag(*mats1[key]) for key in mats1.keys()
        )
        Cex, Dexct, Dexsp, Fexco, Fexex = (
            scipy.linalg.block_diag(*mats2[key]) for key in mats2.keys()
        )
        Cco, Dcoct, Dcosp, Fcoco, Fcoex = (
            scipy.linalg.block_diag(*mats3[key]) for key in mats3.keys()
        )
        Cze, Dzect, Dzesp, Fzeco, Fzeex = (
            scipy.linalg.block_diag(*mats4[key]) for key in mats4.keys()
        )
        Cgt, Dgtct, Dgtsp, Fgtco, Fgtex = (
            scipy.linalg.block_diag(*mats5[key]) for key in mats5.keys()
        )
        Cet, Detct, Detsp, Fetco, Fetex = (
            scipy.linalg.block_diag(*mats6[key]) for key in mats6.keys()
        )

        ss_verbose = np.block(
            [
                [A, Bct, Bsp, Eco, Eex],
                [Cex, Dexct, Dexsp, Fexco, Fexex],
                [Cco, Dcoct, Dcosp, Fcoco, Fcoex],
                [Cze, Dzect, Dzesp, Fzeco, Fzeex],
                [Cgt, Dgtct, Dgtsp, Fgtco, Fgtex],
                [Cet, Detct, Detsp, Fetco, Fetex],
            ]
        )

        labels = dims["labels"]
        dims = dims["dims"]

        for key in dims.keys():
            setattr(self, key, np.sum(dims[key]))

        for key in labels.keys():
            labels[key] = [x for xs in labels[key] for x in xs]

        labels["m"] = labels["mct"] + labels["msp"]

        # this order comes from assumption baked into the node order
        labels["o"] = labels["oex"] + labels["oco"]
        labels["p"] = labels["pco"] + labels["pex"]
        labels["pcons"] = labels["pze"] + labels["pet"] + labels["pgt"]

        for key in labels.keys():
            setattr(self, f"{key}_label", labels[key])

        self.labels = labels
        self.dims = dims

        linear_cols_verbose = {}
        linear_rows_verbose = {}

        for key in linear_cols.keys():
            linear_cols_verbose.update({key: [boo for bool_list in linear_cols[key] for boo in bool_list]})

        for key in linear_rows.keys():
            linear_rows_verbose.update({key: [boo for bool_list in linear_rows[key] for boo in bool_list]})

        np.sum([len(linear_cols_verbose[key]) for key in linear_cols_verbose.keys()])
        np.sum([len(linear_rows_verbose[key]) for key in linear_rows_verbose.keys()])

        # Make indices and reduce the order of the verbose statespace

        # extended incidence matrix
        E_inc = np.concatenate(
            [
                np.array([[1] + [0] * (len(G.nodes) - 1)]).T,
                nx.incidence_matrix(
                    G, oriented=True, nodelist=self.node_order, edgelist=self.edge_order
                ).toarray(),
                np.array([[0] * (len(G.nodes) - 1) + [-1]]).T,
            ],
            axis=1,
        )
        E_inc_in = np.where(E_inc > 0, E_inc, 0)
        E_inc_out = np.where(E_inc < 0, -E_inc, 0)

        p_ins = []
        p_outs = []

        for i, node in enumerate(traversal_order):

            p_in = np.zeros((int(np.sum(E_inc_in[i, :])), E_inc.shape[1]))
            in_inds = np.where(E_inc_in[i, :] == 1)[0]
            for j in range(len(in_inds)):
                p_in[j, in_inds[j]] = 1
            p_ins.append(p_in)

            p_out = np.zeros((int(np.sum(E_inc_out[i, :])), E_inc.shape[1]))
            out_inds = np.where(E_inc_out[i, :] == 1)[0]
            for j in range(len(out_inds)):
                p_out[j, out_inds[j]] = 1
            p_outs.append(p_out)

        P_in = np.concatenate(p_ins, axis=0)
        P_out = np.concatenate(p_outs, axis=0)

        def get_index(label_list, substring):
            return np.array(
                [
                    [
                        i
                        for i in range(len(label_list))
                        if label_list[i].startswith(substring)
                    ]
                ]
            )

        # coupling outputs
        yco_index = get_index(labels["p"], "yco")

        # coupling disturbances
        dco_index = get_index(labels["o"], "dco")

        # coupling edges
        e_co = np.arange(1, len(G.edges) + 1, 1)[None, :]

        M_yco_dco = P_out[yco_index.T, e_co] @ np.linalg.inv(P_in[dco_index.T, e_co])
        # y_co  = M_yco_dco @ d_co
        self.M_yco_dco = M_yco_dco

        M_dco_yco = P_in[dco_index.T, e_co] @ np.linalg.inv(P_out[yco_index.T, e_co])
        # d_co = M_dco_yco @ yco
        self.M_dco_yco = M_dco_yco

        if False:
            fig, ax = plt.subplots(1, 2, layout="constrained")
            ax[0].imshow(Fcoco)
            ax[1].imshow(M_yco_dco)

        MFi = np.linalg.inv(M_yco_dco - Fcoco)

        uncoupled_mat = [
            [A, Bct, Bsp, Eex],
            [Cco, Dcoct, Dcosp, Fcoex],
            [Cex, Dexct, Dexsp, Fexex],
            [Cze, Dzect, Dzesp, Fzeex],
            [Cgt, Dgtct, Dgtsp, Fgtex],
            [Cet, Detct, Detsp, Fetex],
        ]

        coupling_mat = [
            [
                Eco @ MFi @ Cco,
                Eco @ MFi @ Dcoct,
                Eco @ MFi @ Dcosp,
                Eco @ MFi @ Fcoex,
            ],
            [
                Fcoco @ MFi @ Cco,
                Fcoco @ MFi @ Dcoct,
                Fcoco @ MFi @ Dcosp,
                Fcoco @ MFi @ Fcoex,
            ],
            [
                Fexco @ MFi @ Cco,
                Fexco @ MFi @ Dcoct,
                Fexco @ MFi @ Dcosp,
                Fexco @ MFi @ Fcoex,
            ],
            [
                Fzeco @ MFi @ Cco,
                Fzeco @ MFi @ Dcoct,
                Fzeco @ MFi @ Dcosp,
                Fzeco @ MFi @ Fcoex,
            ],
            [
                Fgtco @ MFi @ Cco,
                Fgtco @ MFi @ Dcoct,
                Fgtco @ MFi @ Dcosp,
                Fgtco @ MFi @ Fcoex,
            ],
            [
                Fetco @ MFi @ Cco,
                Fetco @ MFi @ Dcoct,
                Fetco @ MFi @ Dcosp,
                Fetco @ MFi @ Fcoex,
            ],
        ]

        linear_cols_coupled = {}

        # Find new nonlinear columns
        coupling_dict = {"x": MFi @ Cco, "uct": MFi @ Dcoct, "usp": MFi @ Dcosp, "dex": MFi @ Fcoex}
        for key in coupling_dict.keys():
            coupled_nl = coupling_dict[key].T @ np.invert(linear_cols_verbose["dco"])
            coupled_linear = np.invert(coupled_nl.astype(bool))

            linear_cols_coupled.update({key: np.invert(np.invert(linear_cols_verbose[key]) + np.invert(coupled_linear))})

            []

        linear_rows_coupled = linear_rows_verbose

        combined_mat = [
            [
                uncoupled_mat[i][j] + coupling_mat[i][j]
                for j in range(len(uncoupled_mat[i]))
            ]
            for i in range(len(uncoupled_mat))
        ]

        self.print_block_matrices(
            [combined_mat[i] for i in [0, 1, 2, 3, 4, 5]],
            in_labels=["x", "uct", "usp", "dex"],
            out_labels=["x+", "yco", "yex", "yze", "ygt", "yet"],
            save_description=True,
        )

        # self.print_block_matrices(
        #     combined_mat,
        #     in_labels=["x", "uct", "usp", "dex"],
        #     out_labels=["x+", "yco", "yex", "yze", "ygt", "yet"]
        # )

        self.A, self.Bct, self.Bsp, self.Eex = combined_mat[0]
        self.Cco, self.Dcoct, self.Dcosp, self.Fcoex = combined_mat[1]
        self.Cex, self.Dexct, self.Dexsp, self.Fexex = combined_mat[2]
        self.Cze, self.Dzect, self.Dzesp, self.Fzeex = combined_mat[3]
        self.Cgt, self.Dgtct, self.Dgtsp, self.Fgtex = combined_mat[4]
        self.Cet, self.Detct, self.Detsp, self.Fetex = combined_mat[5]

        self.block_ss = np.block(combined_mat)

        mat_names = [
            ["A", "Bct", "Bsp", "Eex"],
            ["Cco", "Dcoct", "Dcosp", "Fcoex"],
            ["Cex", "Dexct", "Dexsp", "Fexex"],
            ["Cze", "Dzect", "Dzesp", "Fzeex"],
            ["Cgt", "Dgtct", "Dgtsp", "Fgtex"],
            ["Cet", "Detct", "Detsp", "Fetex"],
        ]

        # TODO apply scaling here

        # self.calculate_minimal_inputs()

        self.E_inc = E_inc
        self.P_in = P_in
        self.P_out = P_out

        for key in bounds.keys():
            bounds[key] = np.concatenate(bounds[key])

        self.bounds = bounds
        self.bounds_verbose = verbose_bounds

        # Separate out the yco indices that need to be there and the ones that dont
        yco_ub = []

        self.yco_ub_node_ind = np.where(self.bounds["y_ub"] != np.inf)[0]
        for y_ind in np.where(self.bounds["y_ub"] != np.inf)[0]:
            node = self.node_order[y_ind]
            for j in range(self.pco):
                if (self.pco_label[j].split(" ")[2] == node):
                    yco_ub.append(j)
        self.yco_ub_ind = np.sort(yco_ub)

        self.uct_order = uct_order
        self.usp_order = usp_order

        self.linear_cols_dict = linear_cols_coupled
        self.linear_rows_dict = linear_rows_coupled

        cols_li = []
        cols_nl = []
        col_count = 0
        for key in ["x", "uct", "usp", "dex"]:
            for boo in linear_cols_coupled[key]:
                if boo:
                    cols_li.append(col_count)
                else:
                    cols_nl.append(col_count)
                col_count += 1
        self.cols_li = np.array(cols_li)[None, :]
        self.cols_nl = np.array(cols_nl)[None, :]

        rows_li = []
        rows_nl = []
        row_count = 0
        for key in ["x", "yex", "yco", "yze", "ygt", "yet"]:
            for boo in linear_rows_coupled[key]:
                if boo:
                    rows_li.append(row_count)
                else:
                    rows_nl.append(row_count)
                row_count += 1
        self.rows_li = np.array(rows_li)[:, None]
        self.rows_nl = np.array(rows_nl)[:, None]

        # self.solve_steady_reference()

        []

    def solve_steady_reference(self):

        # solving for Ax = b for steady state

        ref_yex = 35

        dex_ref = np.array([[142e3]])
        xy_vec = np.concatenate(
            [
                np.zeros((self.n, 1)),
                np.array([[ref_yex]]),
                np.zeros((self.pze, 1)),
                np.zeros((self.pgt, 1)),
            ]
        )
        b = xy_vec - np.concatenate(
            [
                self.Eex @ dex_ref,
                self.Fexex @ dex_ref,
                self.Fzeex @ dex_ref,
                self.Fgtex @ dex_ref,
            ]
        )

        A = np.block(
            [
                [self.A - np.eye(self.n), self.Bsp],
                [self.Cex, self.Dexsp],
                [self.Cze, self.Dzesp],
                [self.Cgt, self.Dgtsp],
            ]
        )
        # A = np.block([
        #     [self.A - np.eye(self.n), self.Bct, self.Bsp],
        #     [self.Cex, self.Dexct, self.Dexsp],
        #     [self.Cze, self.Dzect, self.Dzesp],
        #     [self.Cgt, self.Dgtct, self.Dgtsp]
        # ])

        []

    def plot_solution(self, uct, usp, x, ysp, forecast):

        # uc = sol.value(self.opt_vars["uct"])
        # us = sol.value(self.opt_vars["usp"])
        # x = sol.value(self.opt_vars["x"])
        # ys = sol.value(self.opt_vars["ysp"])[None, :]
        # e = sol.value(self.opt_vars["e"])

        uc = uct
        us = usp
        x = x
        ys = ysp

        fig, ax = plt.subplots(
            np.max([len(uc), len(us), len(x), len(ys)]),
            4,
            sharex="all",
            layout="constrained",
        )

        to_plot = [x, uc, us, ys]
        for i in range(len(to_plot)):
            for j in range(len(to_plot[i])):
                ax[j, i].plot(to_plot[i][j, :])

        fig, ax = plt.subplots(
            4, 2, figsize=(10, 10), sharex="all", layout="constrained"
        )

        ax[0, 0].fill_between(
            np.arange(0, len(forecast), 1),
            np.zeros(len(forecast)),
            forecast,
            alpha=0.25,
            edgecolor=None,
            color="yellow",
            label="forecast",
        )
        ax[0, 0].plot(us[0:2, :].T, label="generation")

        ax[1, 0].plot(uc[0, :], label="battery charge")
        # ax[1, 0].plot(-e[3, :], label="battery discharge")

        ax[2, 0].plot(forecast - uc[0, :], label="gen + bes")

        ax[2, 1].plot(us[3, :] - us[2, :] - uc[1, :], label="H2 to steel")
        ax[1, 1].plot(uc[1, :], label="H2S charge")
        # ax[1, 1].plot(-e[6, :], label="H2S discharge")

        ax[0, 1].fill_between(
            np.arange(0, uc.shape[1], 1),
            np.zeros(uc.shape[1]),
            np.sum(us[2:4, :], axis=0),
            alpha=0.25,
            edgecolor=None,
            color="blue",
            label="H2 gen",
        )

        ax[0, 1].plot(us[2:4, :].T, label="H2 gen")

        ax[3, 1].plot(ys[0, :], label="Steel")

        for i in range(ax.shape[0]):
            for j in range(ax.shape[1]):
                ax[i, j].legend()

    def print_block_matrices(
        self, mat, in_labels, out_labels, no_space=False, save_description=False
    ):

        try:
            np.block(mat)
        except:
            AssertionError("bad matrix")

        rounding_tol = -9
        rounded_flag = False

        block_mat = np.block(mat)

        col_widths = np.zeros(block_mat.shape[1], dtype=int)
        for i in range(block_mat.shape[1]):
            col_widths[i] = int(
                np.max(
                    [len(f"{block_mat[j,i]:.4g}") for j in range(block_mat.shape[0])]
                )
                + 2
            )

        # block_cols = block_mat.shape[1]
        # block_rows = block_mat.shape[0]

        out_label_width = int(np.max([len(label) for label in out_labels]))
        num_col_width = 10

        print_str = ""

        for row_num, row_mat in enumerate(mat):
            if not no_space:
                # print("")
                print_str += "\n"

            row_mat_lens = [matr.shape[1] for matr in row_mat]
            if row_num == 0:
                line = " " * (out_label_width + 5)
                # line2 = " " * (out_label_width + 5 + 4)
                line2 = " " * (out_label_width + 3)
                col_count = 0
                for coli, col_label in enumerate(in_labels):
                    label_pad = 0
                    for j in range(row_mat_lens[coli]):
                        # line2 += f"{j}".ljust(num_col_width)
                        line2 += f"{j}".rjust(col_widths[col_count])
                        label_pad += col_widths[col_count]
                        col_count += 1

                    line += f"{col_label}".ljust(label_pad + 2)
                    line2 += " " * 2

                # print(line)
                # print(line2)
                print_str += line + "\n"
                print_str += line2 + "\n"

            n_rows = row_mat[0].shape[0]
            for i in range(n_rows):
                line = f"{out_labels[row_num]}".ljust(out_label_width + 3)
                line += "[ "
                col_count = 0

                for col_mat in row_mat:
                    for j in range(col_mat.shape[1]):
                        if np.abs(col_mat[i, j]) < 10 ** (rounding_tol):
                            num = 0
                            rounded_flag = True
                        else:
                            num = np.round(col_mat[i, j], -rounding_tol)

                        # line += f"{col_mat[i,j] :.4g}, ".rjust(num_col_width)
                        # line += f"{num :.4g}, ".rjust(num_col_width)
                        line += f"{num :.4g}, ".rjust(col_widths[col_count])
                        col_count += 1

                    line = line[0:-2]
                    line += " ][ "
                line = line[0:-2]
                # print(line)
                print_str += line + "\n"

        if rounded_flag:
            print_str += (
                f"some values were lower than 1e{rounding_tol} so they were set to 0\n"
            )

        if save_description:
            self.state_space_string = print_str
        else:
            print(print_str)

        []

    def calculate_minimal_inputs(self):

        Cze = np.block([[self.Cze], [self.Cgt], [self.Cet]])
        Dzect = np.block([[self.Dzect], [self.Dgtct], [self.Detct]])
        Dzesp = np.block([[self.Dzesp], [self.Dgtsp], [self.Detsp]])
        Fzeex = np.block([[self.Fzeex], [self.Fgtex], [self.Fetex]])

        n_constraints = Cze.shape[0]
        n_variables = Dzect.shape[1] + Dzesp.shape[1]
        n_inds = n_variables - n_constraints
        n_deps = n_constraints

        uct_inds = np.array([1, 2, 5])
        usp_inds = np.array([1, 3, 5, 6])

        uct_deps = np.array(
            [i for i in range(self.mct) if i not in uct_inds], dtype=int
        )
        usp_deps = np.array(
            [i for i in range(self.msp) if i not in usp_inds], dtype=int
        )

        Dze = np.block([Dzect, Dzesp])
        inds_desired = np.concatenate([uct_inds, usp_inds + self.mct])
        deps_desired = np.concatenate([uct_deps, usp_deps + self.mct])

        # [(i, np.linalg.matrix_rank(Dze[:, np.delete(inds_desired, i)])) for i in range(len(inds_desired))]
        # [(i, np.linalg.matrix_rank(Dze[:, np.delete(deps_desired, i)])) for i in range(len(deps_desired))]

        inds = inds_desired
        # inds = np.array([0, 1, 2, 4, 6, 8, 9, 10, 11, 14, 16, 18])
        # inds = np.array([1,  3,  5,  6,  7, 8, 11, 14])
        deps = np.array(
            [i for i in range(self.mct + self.msp) if i not in inds], dtype=int
        )

        # inds_leftover = np.array(range(Dze.shape[1]))

        # for ind in inds_desired:
        #     _, index = sympy.Matrix(Dze[:, inds_leftover]).rref()
        #     if ind in index:
        #         inds_leftover = np.delete(inds_leftover, np.where(inds_leftover == ind)[0][0])

        # import sympy
        # _, index = sympy.Matrix(Dze).rref()
        # deps = index
        # inds = np.array([i for i in range(self.mct + self.msp) if i not in inds], dtype=int)

        Dze_inv = np.linalg.inv(Dze[:, deps])

        Dze_ind = Dze[:, inds]

        ind_ct = [i for i in inds if i < self.mct]
        ind_sp = [
            i - self.mct for i in inds if (i >= self.mct) and (i < self.mct + self.msp)
        ]

        dep_ct = [i for i in inds if i < self.mct]
        dep_sp = [
            i - self.mct for i in deps if (i >= self.mct) and (i < self.mct + self.msp)
        ]

        ct_labels = [self.mct_label[i] for i in ind_ct]
        sp_labels = [self.msp_label[i] for i in ind_sp]

        ct_labels_dep = [self.mct_label[i] for i in dep_ct]
        sp_labels_dep = [self.msp_label[i] for i in dep_sp]

        Anew = self.A - np.block([self.Bct, self.Bsp])[:, deps] @ Dze_inv @ Cze
        Bnew = (
            np.block([self.Bct, self.Bsp])[:, inds]
            - np.block([self.Bct, self.Bsp])[:, deps] @ Dze_inv @ Dze_ind
        )
        Enew = self.Eex - np.block([self.Bct, self.Bsp])[:, deps] @ Dze_inv @ Fzeex

        Cnew = self.Cex - np.block([self.Dexct, self.Dexsp])[:, deps] @ Dze_inv @ Cze
        Dnew = (
            np.block([self.Dexct, self.Dexsp])[:, inds]
            - np.block([self.Dexct, self.Dexsp])[:, deps] @ Dze_inv @ Dze_ind
        )
        Fnew = (
            self.Fexex - np.block([self.Dexct, self.Dexsp])[:, deps] @ Dze_inv @ Fzeex
        )

        Cdep = Dze_inv @ Cze
        Ddep = Dze_inv @ Dze_ind
        Fdep = Dze_inv @ Fzeex

        mat = [[Anew, Bnew, Enew], [Cnew, Dnew, Fnew], [Cdep, Ddep, Fdep]]

        self.print_block_matrices(
            mat,
            in_labels=["x", "u", "dex"],
            out_labels=["x+", "yex", "udep"],
            save_description=False,
        )
        pprint.pprint(list(zip(range(n_inds), ct_labels + sp_labels)))
        pprint.pprint(list(zip(range(n_deps), ct_labels_dep + sp_labels_dep)))

        []

    def plot_saved_trajectories(self):

        n_nodes = len(self.node_order)
        fig, ax = plt.subplots(
            n_nodes, 4, sharex="all", layout="constrained", figsize=(15, 10), dpi=100
        )

        ax[0, 0].set_title("Disturbance")
        ax[0, 1].set_title("Control input")
        ax[0, 2].set_title("State")
        ax[0, 3].set_title("Output")
        # ax[0, 4].set_title("Split")

        # for i, node in enumerate(list(RTS.G.nodes)):
        for i, node in enumerate(self.node_order):

            ax[i, 0].set_ylabel("\n".join(node.split("_")))

            # 0 - disturbance, 1 - control input, 2 - states, 3- outputs total, 4 - outputs split

            dex_inds = [
                i for i in range(len(self.oex_label)) if node in self.oex_label[i]
            ]
            dco_inds = [
                i
                for i in range(len(self.oco_label))
                if node in self.oco_label[i].split(" ")[2]
            ]

            uct_inds = [
                i for i in range(len(self.mct_label)) if node in self.mct_label[i]
            ]
            usp_inds = [
                i
                for i in range(len(self.msp_label))
                if node in self.msp_label[i].split(" ")[2]
            ]

            x_inds = [i for i in range(len(self.n_label)) if node in self.n_label[i]]
            y_inds = [
                i
                for i in range(len(self.p_label))
                if node in self.p_label[i].split(" ")[2]
            ]

            colors = ["blue", "orange", "red", "brown", "cyan"]

            def plot_one(ax, stored, inds):
                for j in range(len(self.step_index_store)):
                    t = np.arange(
                        self.step_index_store[j],
                        self.step_index_store[j] + self.horizon,
                    )[None, :]
                    for k in range(len(inds)):
                        if self.horizon == 1:
                            ax.scatter(t * np.ones(len(inds)), stored[j][inds, :], color=colors[k])
                        else:
                            ax.plot(t.T, stored[j][inds, :].T, color=colors[k])

            plot_one(ax[i, 0], self.dco_store, dco_inds)
            plot_one(ax[i, 1], self.uct_store, uct_inds)
            plot_one(
                ax[i, 2], [xst[:, 0 : self.horizon] for xst in self.x_store], x_inds
            )
            plot_one(
                ax[i, 3],
                [np.sum(ysp[y_inds, :], axis=0)[None, :] for ysp in self.ysp_store],
                [0],
            )
            # plot_one(ax[i, 4], self.ysp_store, y_inds)
            # ax[i, 4].set_ylim(ax[i, 3].get_ylim())

            if node == "generation":
                forecast_curtail_grid = [
                    np.concatenate([self.forecast_store[i], self.forecast_store[i] - self.curtail_store[i], self.forecast_store[i] + self.grid_store[i]])
                    # [self.forecast_store[i], np.array(self.forecast_store[i]) - np.array(self.curtail_store[i]), np.array(self.forecast_store[i]) + np.array(self.grid_store[i])]
                    for i in range(len(self.step_index_store))
                ]

                plot_one(ax[i, 0], forecast_curtail_grid, np.array([0, 1, 2]))

            fig.align_ylabels()

        pass


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout



if __name__ == "__main__":
    from pathlib import Path

    from greenheart.simulation.greenheart_simulation import GreenHeartSimulationConfig
    from greenheart.simulation.realtime_simulation import RealTimeSimulation
    from greenheart.simulation.technologies.dispatch.controllers.dispatch_mpc import    DispatchModelPredictiveController as mpc_ca
    from greenheart.simulation.technologies.dispatch.controllers.dispatch_mpc_pyomo import DispatchModelPredictiveController
    from hopp.simulation.technologies.sites.site_info import SiteInfo

    config_root = Path(
        "/Users/ztully/Documents/hybrids_code/GH_scripts/greenheart_scripts/minnesota_reference_design/01-minnesota-steel/greenHEART/input-files"
    )

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

    hopp_site = SiteInfo(**config.hopp_config["site"])


    class hopp_system:
        def __init__(self, site):
            self.site = site


    class hopp_interface:
        def __init__(self, site):
            self.system = hopp_system(site)


    hi = hopp_interface(hopp_site)
    simulator = RealTimeSimulation(config, hi)

    mpc_config = config.greenheart_config["realtime_simulation"]["dispatch"]["mpc"]

    # mpc_config["weights"]["output_tracking"] *= 1e-7

    horizon = 3
    mpc_config["horizon"] = horizon
    mpc = DispatchModelPredictiveController(
        config,
        simulator.G,
        node_order=simulator.node_order,
        edge_order=simulator.edge_order,
        mpc_config=mpc_config,
    )
    mpc.warm_start_with_previous_solution = False
    mpc.no_shortfall = False

    x0 = np.array([1800000.   , 3238578.913,  783000.   ])
    forecast = np.array([ 94841.828,  81526.197, 118532.144, 111062.737, 235747.435, 234696.2  ])

    # mpc.update_optimization_parameters(x0, forecast)
    uct, usp, curtail, grid, obj_values_uw = mpc.compute_trajectory(x0, forecast, ret_obj=True)



    []