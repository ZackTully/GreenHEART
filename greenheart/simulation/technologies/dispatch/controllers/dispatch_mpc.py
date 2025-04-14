import numpy as np
import scipy
import networkx as nx
import matplotlib.pyplot as plt
import casadi as ca
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
        self.warm_start_with_previous_solution = False
        self.grid_curtail_mod = True

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
                self.ref_bes_state = (
                    0.7
                    * simulation_graph.nodes["battery"]["ionode"].model.max_capacity_kWh
                )
                self.weight_bes_state = 1e-4 / self.ref_bes_state

            if "hydrogen_storage" in self.node_order:
                self.ref_h2s_state = (
                    0.7
                    * simulation_graph.nodes["hydrogen_storage"][
                        "ionode"
                    ].model.max_capacity_kg
                )
                self.weight_h2s_state = 1e-1 / self.ref_h2s_state

            if "thermal_energy_storage" in self.node_order:
                self.ref_tes_state = (
                    0.7
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

        # =============================================================================
        # ==                                                                         ==
        # ==                         Optimization setup                              ==
        # ==                                                                         ==
        # =============================================================================

        opti: ca.Opti = ca.Opti()
        opti.solver("ipopt", self.p_opts, self.s_opts)

        # Variables and bounds
        uct_var = opti.variable(self.mct, self.horizon)
        usp_var = opti.variable(self.msp, self.horizon)
        x_var = opti.variable(self.n, self.horizon + 1)
        yex_var = opti.variable(self.pex, self.horizon)
        yco_var = opti.variable(self.pco, self.horizon)
        # e_var = opti.variable(self.q + 2, self.horizon)

        # for k in range(self.horizon+1):
        for k in range(self.horizon + 1):
            opti.subject_to(x_var[:, k] >= self.bounds["x_lb"][:, None])
            opti.subject_to(x_var[:, k] <= self.bounds["x_ub"][:, None])

        for k in range(self.horizon):
            opti.subject_to(uct_var[:, k] >= self.bounds["u_lb"][:, None])
            opti.subject_to(uct_var[:, k] <= self.bounds["u_ub"][:, None])

            # opti.subject_to(yco_var[:, k] >= ca.MX(list(self.bounds["y_lb"][:, None])))
            # opti.subject_to(yco_var[:, k] <= ca.MX(list(self.bounds["y_ub"][:, None])))

            # opti.subject_to(yex_var[:,k] >= self.bounds["y_lb"][:, None])
            # opti.subject_to(yex_var[:,k] <= self.bounds["y_ub"][:, None])
            opti.subject_to(yex_var[:, k] >= 0)
            opti.subject_to(usp_var[:, k] >= np.zeros(self.msp))

            for node in self.node_order:

                node_idx = [i for i in range(self.pco) if self.pco_label[i].split(" ")[2] == node]

                if len(node_idx) > 0:
                    opti.subject_to(np.ones((1, len(node_idx))) @ yco_var[node_idx, k] <= self.bounds_verbose[node]["y_ub"])
                    opti.subject_to(np.ones((1, len(node_idx))) @ yco_var[node_idx, k] >= self.bounds_verbose[node]["y_lb"])

        dex_param = opti.parameter(self.oex, self.horizon)
        # e_src_param = opti.parameter(1, self.horizon)
        x0_param = opti.parameter(self.n, 1)

        if self.grid_curtail_mod:
            gridcurtail = opti.variable(self.oex, self.horizon)
            opti.subject_to(gridcurtail >= -dex_param)

        else:

            if self.allow_curtail_forecast:
                curtail = opti.variable(self.oex, self.horizon)
                opti.subject_to(curtail <= dex_param)
                opti.subject_to(curtail >= np.zeros(curtail.shape))
            else:
                curtail = opti.parameter(self.oex, self.horizon)

            if self.allow_grid_purchase:
                grid_purchase = opti.variable(self.oex, self.horizon)
                opti.subject_to(grid_purchase >= np.zeros(grid_purchase.shape))
            else:
                grid_purchase = opti.parameter(self.oex, self.horizon)

        # Parameters
        # Initial conditions and forecasted disturbance
        opti.subject_to(x_var[:, 0] == x0_param)
        # opti.subject_to(e_var[0, :] == e_src_param)

        objective = 0
        objective_terms = []

        # Set up indices for objective terms flexibly

        def find_index(label, index_list):
            indices = [i for i in range(len(index_list)) if label in index_list[i]]
            # print(indices)
            assert len(indices) == 1
            return indices[0]

        objective_var_inds = {}

        if "battery" in self.node_order:
            # uct_charge_bes = find_index("uct 0 battery", self.mct_label)
            # uct_discharge_bes = find_index("uct 1 battery", self.mct_label)
            # x_bes = find_index("x 0 battery", self.n_label)

            objective_var_inds.update(
                {
                    "uct_charge_bes": find_index("uct 0 battery", self.mct_label),
                    "uct_discharge_bes": find_index("uct 1 battery", self.mct_label),
                    "x_bes": find_index("x 0 battery", self.n_label),
                }
            )

        if "hydrogen_storage" in self.node_order:
            # uct_charge_h2s = find_index("uct 0 hydrogen_storage", self.mct_label)
            # uct_discharge_h2s = find_index("uct 1 hydrogen_storage", self.mct_label)
            # x_h2s = find_index("x 0 hydrogen_storage", self.n_label)

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
            # uct_charge_tes = find_index("uct 0 thermal_energy_storage", self.mct_label)
            # uct_discharge_tes = find_index( "uct 1 thermal_energy_storage", self.mct_label   )
            # x_tes = find_index("x 0 thermal_energy_storage", self.n_label)
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

        # Dynamics constraint
        for i in range(self.horizon):

            if self.grid_curtail_mod:
                grid_curtail = gridcurtail[:,i]
            else:
                grid_curtail = -curtail[:, i] + grid_purchase[:, i]

            xkp1, yexk, yco, yze, ygt, yet = self.step_control_model(x_var[:, i], uct_var[:, i], usp_var[:, i], dex_param[:, i], grid_curtail)


            # xkp1 = (
            #     self.A @ x_var[:, i]
            #     + self.Bct @ uct_var[:, i]
            #     + self.Bsp @ usp_var[:, i]
            #     + self.Eex @ (dex_param[:, i] + grid_curtail)
            # )
            # # external outputs
            # yexk = (
            #     self.Cex @ x_var[:, i]
            #     + self.Dexct @ uct_var[:, i]
            #     + self.Dexsp @ usp_var[:, i]
            #     + self.Fexex @ (dex_param[:, i] + grid_curtail)
            # )

            # # coupling outputs
            # yco = (
            #     self.Cco @ x_var[:, i]
            #     + self.Dcoct @ uct_var[:, i]
            #     + self.Dcosp @ usp_var[:, i]
            #     + self.Fcoex @ (dex_param[:, i] + grid_curtail)
            # )

            # # Splitting constraint zero outputs
            # yze = (
            #     self.Cze @ x_var[:, i]
            #     + self.Dzect @ uct_var[:, i]
            #     + self.Dzesp @ usp_var[:, i]
            #     + self.Fzeex @ (dex_param[:, i] + grid_curtail)
            # )

            # # greater than 0 constraint outputs
            # ygt = (
            #     self.Cgt @ x_var[:, i]
            #     + self.Dgtct @ uct_var[:, i]
            #     + self.Dgtsp @ usp_var[:, i]
            #     + self.Fgtex @ (dex_param[:, i] + grid_curtail)
            # )

            # # equal to 0 constraint outputs
            # yet = (
            #     self.Cet @ x_var[:, i]
            #     + self.Detct @ uct_var[:, i]
            #     + self.Detsp @ usp_var[:, i]
            #     + self.Fetex @ (dex_param[:, i] + grid_curtail)
            # )

            opti.subject_to(x_var[:, i + 1] == xkp1)
            opti.subject_to(yex_var[:, i] == yexk[0])
            opti.subject_to(yco_var[:, i] == yco)
            if self.pze > 0:
                opti.subject_to(yze == np.zeros((self.pze, 1)))
            if self.pgt > 0:
                opti.subject_to(ygt == np.zeros((self.pgt, 1)))
            if self.pet > 0:
                opti.subject_to(yet == np.zeros((self.pet, 1)))

            if self.grid_curtail_mod:
                step_obj, step_obj_terms = self.objective_step(
                    x_var[:, i],
                    uct_var[:, i],
                    usp_var[:, i],
                    yco,
                    yexk,
                    gridcurtail=gridcurtail[:,i],
                    var_inds=objective_var_inds,
                )
            else:

                step_obj, step_obj_terms = self.objective_step(
                    x_var[:, i],
                    uct_var[:, i],
                    usp_var[:, i],
                    yco,
                    yexk,
                    curtail=curtail[:, i],
                    grid=grid_purchase[:, i],
                    var_inds=objective_var_inds,
                )
            objective += step_obj
            objective_terms.append(step_obj_terms)

            # opti.subject_to(yco >= self.bounds["y_lb"])
            # opti.subject_to(yco <= self.bounds["y_ub"])

        # if self.allow_curtail_forecast:
        # opti.subject_to(curtail <= dex_param)
        # opti.subject_to(curtail >= np.zeros(curtail.shape))

        # Bounds
        # Add constraint
        # Objective
        # opti.minimize(self.objective(x_var, uct_var, usp_var, yex_var, curtail))
        opti.minimize(objective)

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

        # for j in range(len(objective_terms[0])):
        #     obj_term = 0
        #     # term_label = self.objective_labels[j]
        #     term_label = list(objective_terms[0].keys())[j]
        #     for i in range(len(objective_terms)):

        #         obj_term += objective_terms[i][j]

        #     self.objective_terms.update({term_label: obj_term})

        self.opti = opti
        self.opt_vars = {
            "uct": uct_var,
            "usp": usp_var,
            "x": x_var,
            "yex": yex_var,
            # "e": e_var,
            "yco": yco_var,
        }
        self.opt_params = {"dex": dex_param, "x0": x0_param}

        if self.grid_curtail_mod:

            self.opt_vars.update({"gridcurtail": gridcurtail})
        else:
            if self.allow_curtail_forecast:
                self.opt_vars.update({"curtail": curtail})
            else:
                self.opt_params.update({"curtail": curtail})

            if self.allow_grid_purchase:
                self.opt_vars.update({"grid": grid_purchase})
            else:
                self.opt_params.update({"grid": grid_purchase})

    def step_control_model(self, x_var, uct_var, usp_var, dex_param, grid_curtail):


            xkp1 = (
                self.A @ x_var
                + self.Bct @ uct_var
                + self.Bsp @ usp_var
                + self.Eex @ (dex_param + grid_curtail)
            )
            # external outputs
            yexk = (
                self.Cex @ x_var
                + self.Dexct @ uct_var
                + self.Dexsp @ usp_var
                + self.Fexex @ (dex_param + grid_curtail)
            )

            # coupling outputs
            yco = (
                self.Cco @ x_var
                + self.Dcoct @ uct_var
                + self.Dcosp @ usp_var
                + self.Fcoex @ (dex_param + grid_curtail)
            )

            # Splitting constraint zero outputs
            yze = (
                self.Cze @ x_var
                + self.Dzect @ uct_var
                + self.Dzesp @ usp_var
                + self.Fzeex @ (dex_param + grid_curtail)
            )

            # greater than 0 constraint outputs
            ygt = (
                self.Cgt @ x_var
                + self.Dgtct @ uct_var
                + self.Dgtsp @ usp_var
                + self.Fgtex @ (dex_param + grid_curtail)
            )

            # equal to 0 constraint outputs
            yet = (
                self.Cet @ x_var
                + self.Detct @ uct_var
                + self.Detsp @ usp_var
                + self.Fetex @ (dex_param + grid_curtail)
            )

            return self.step_control_model_NL(x_var, uct_var, usp_var, dex_param, grid_curtail)

            return xkp1, yexk, yco, yze, ygt, yet

    def step_control_model_NL(self, x_var, uct_var, usp_var, dex_param, grid_curtail):


        Y_block = self.block_ss @ ca.vertcat(x_var, uct_var, usp_var, dex_param+ grid_curtail)

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
            {"output_tracking": {"w": 1e9, "expr": (self.reference - yex) ** 2}}
        )

        if self.grid_curtail_mod:
            obj_terms.update({"gridcurtail": {"w": 1e-4, "expr": gridcurtail**2}})
        else:
            obj_terms.update({"curtail": {"w": 1e-4, "expr": curtail**2}})
            # grid_purchase = grid
            obj_terms.update({"grid_purchase": {"w": 1e-4, "expr": grid**2}})
            obj_terms.update({"gen_simultaneous": {"w": 1, "expr": curtail * grid}})

        # Sloppy terms for better tracking
        # h2_ref = (
        #     self.reference / self.G.nodes["steel"]["ionode"].model.control_model.F
        # )[0, 1]
        # P_ref = -self.G.nodes["steel"]["ionode"].model.control_model.F_gt[0, 1] * h2_ref
        # Q_ref = (
        #     -self.G.nodes["heat_exchanger"]["ionode"].model.control_model.F_gt[0, 0]
        #     * h2_ref
        # )

        # obj_terms.update(
        #     {"h2_ref": {"w": 1e0, "expr": ((usp[8] + uct[5]) - h2_ref) ** 2}}
        # )
        # term_keys.append("h2_ref")

        # obj_terms.update(
        #     {"P_ref": {"w": 1e0, "expr": ((usp[3] + usp[6]) - P_ref) ** 2}}
        # )
        # term_keys.append("P_ref")

        # obj_terms.update({"Q_ref": {"w": 1e0, "expr": (uct[3] - h2_ref) ** 2}})
        # term_keys.append("Q_ref")

        if "battery" in self.node_order:
            simu = uct[var_inds["uct_charge_bes"]] * uct[var_inds["uct_discharge_bes"]]
            obj_terms.update({"bes_simultaneous": {"w": 1e0, "expr": simu}})

            state = (x[var_inds["x_bes"]] - self.ref_bes_state) ** 2
            obj_terms.update({"bes_state": {"w": self.weight_bes_state, "expr": state}})
            # state = (x[var_inds["x_bes"]] - (2e6 - 4e5)) ** 2
            # obj_terms.update({"bes_state": {"w": 1e-8, "expr": state}})

            term_keys.append("bes_simultaneous")
            # term_keys.append("bes_state")

        if "hydrogen_storage" in self.node_order:
            simu = uct[var_inds["uct_charge_h2s"]] * uct[var_inds["uct_discharge_h2s"]]
            obj_terms.update({"h2s_simultaneous": {"w": 1e0, "expr": simu}})

            state = (x[var_inds["x_h2s"]] - self.ref_h2s_state) ** 2
            obj_terms.update({"h2s_state": {"w": self.weight_h2s_state, "expr": state}})
            # state = (x[var_inds["x_h2s"]] - (812209 / 2)) ** 2
            # obj_terms.update({"h2s_state": {"w": 1e-7, "expr": state}})

            term_keys.append("h2s_simultaneous")
            # term_keys.append("h2s_state")

        if "thermal_energy_storage" in self.node_order:
            simu = uct[var_inds["uct_charge_tes"]] * uct[var_inds["uct_discharge_tes"]]
            obj_terms.update({"tes_simultaneous": {"w": 1e0, "expr": simu}})

            state = (x[var_inds["x_tes"]] - self.ref_tes_state) ** 2
            obj_terms.update({"tes_state": {"w": self.weight_tes_state, "expr": state}})
            # state = (x[var_inds["x_tes"]] - 6.4e6 / 2) ** 2
            # obj_terms.update({"tes_state": {"w": 1e-9, "expr": state}})

            if self.weights["tes_simultaneous"] > 0:
                term_keys.append("tes_simultaneous")
            term_keys.append("tes_state")

        if self.use_config_weights:

            for key in self.weights.keys():
                if key in obj_terms:
                    obj_terms[key]["w"] = self.weights[key]

            # for term in term_keys:
            #     if term in self.weights.keys():
            #         obj_terms[term]["w"] = self.weights[term]

        if hasattr(self, "term_keys"):
            obj_term_keys = self.term_keys
        else:
            obj_term_keys = term_keys

        objective = 0
        for term in obj_term_keys:
            objective += obj_terms[term]["w"] * obj_terms[term]["expr"]

        obj_terms.update({"objective": {"w": 1, "expr": objective}})
        return objective, obj_terms

    # def objective_step(self, x, uct, usp, yco, yex, curtail, grid, var_inds= None):

    #     # =============================================================================
    #     # ==                                                                         ==
    #     # ==                                Objective                                ==
    #     # ==                                                                         ==
    #     # =============================================================================

    #     # ref_steel = 45.48e3
    #     ref_steel = 35
    #     self.reference = ref_steel
    #     output_tracking = (ref_steel - yex) ** 2
    #     # h2_tracking = 1e-3 *(2084.6 - (yco[5] + yco[6]))**2
    #     h2_tracking = (2084.6 - (yco[6] + yco[7])) ** 2

    #     grid_purchase = grid

    #     # BES_local_curtail = (yco[0] - uct[0]) ** 2
    #     BES_simultaneous = 1 * uct[0] * uct[1]
    #     BES_state = (x[0] - (2e6 - 4e5) / 2) ** 2

    #     # H2S_local_curtail =  (yco[4] - uct[2]) ** 2
    #     # H2S_local_curtail = (yco[5] - uct[2]) ** 2
    #     H2S_simultaneous = 1 * uct[4] * uct[5]
    #     # H2S_simultaneous = 1 * uct[2] * uct[3]
    #     H2s_state = (x[2] - (812209 / 2)) ** 2
    #     # H2s_state = (x[1] - (812209 / 2)) ** 2

    #     TES_simultaneous = uct[2] * uct[3]
    #     TES_state = (x[1] - 6.4e6 / 2) ** 2

    #     w_output = 1e5
    #     w_h2_tracking = 1e0
    #     w_bes_simultaneous = 1e0
    #     # w_bes_local_curtail = 1e0
    #     w_bes_state = 1e-8
    #     w_h2s_simultaneous = 1e0
    #     # w_h2s_local_curtail = 1e0
    #     w_tes_simultaneous = 1e0
    #     w_h2s_state = 1e-7
    #     w_tes_state = 1e-8
    #     w_curtail = 1e-4
    #     w_grid_purchase = 1e-3

    #     objective_weights = [
    #         1,  # no weight for the total optimization objective
    #         w_output,
    #         w_h2_tracking,
    #         w_bes_simultaneous,
    #         # w_bes_local_curtail,
    #         w_bes_state,
    #         w_h2s_simultaneous,
    #         # w_h2s_local_curtail,
    #         w_h2s_state,
    #         w_tes_state,
    #         w_tes_simultaneous,
    #         w_curtail,
    #         w_grid_purchase,
    #     ]

    #     obj_value = (
    #         w_output * output_tracking
    #         # + w_h2_tracking * h2_tracking
    #         + w_bes_simultaneous * BES_simultaneous
    #         # + w_bes_local_curtail * BES_local_curtail
    #         + w_bes_state * BES_state
    #         + w_h2s_simultaneous * H2S_simultaneous
    #         # + w_h2s_local_curtail * H2S_local_curtail
    #         + w_h2s_state * H2s_state
    #         + w_tes_state * TES_state
    #         + w_tes_simultaneous * TES_simultaneous
    #         + w_curtail * curtail**2
    #         + w_grid_purchase * grid_purchase**2
    #     )
    #     objective_terms = [
    #         obj_value,
    #         w_output * output_tracking,
    #         w_h2_tracking * h2_tracking,
    #         w_bes_simultaneous * BES_simultaneous,
    #         # w_bes_local_curtail * BES_local_curtail,
    #         w_bes_state * BES_state,
    #         w_h2s_simultaneous * H2S_simultaneous,
    #         # w_h2s_local_curtail * H2S_local_curtail,
    #         w_h2s_state * H2s_state,
    #         w_tes_state * TES_state,
    #         w_tes_simultaneous * TES_simultaneous,
    #         w_curtail * curtail**2,
    #         w_grid_purchase * grid_purchase,
    #     ]
    #     obj_term_labels = [
    #         "objective",
    #         "output",
    #         "h2_tracking",
    #         "bes_simultaneous",
    #         # "bes_local_curtail",
    #         "bes_state",
    #         "h2s_simultaneous",
    #         # "h2s_local_curtail",
    #         "h2s_state",
    #         "tes_state",
    #         "tes_simultaneous",
    #         "curtail",
    #         "grid_purchase",
    #     ]
    #     self.objective_labels = {
    #         i: obj_term_labels[i] for i in range(len(obj_term_labels))
    #     }
    #     self.objective_weights = {
    #         obj_term_labels[i]: objective_weights[i]
    #         for i in range(len(obj_term_labels))
    #     }
    #     return obj_value, objective_terms

    # def objective(self, x, uc, us, ys, curtail):

    #     ref_steel = 45.48e3
    #     # ref_steel = 37.92e3
    #     # ref_steel = 20e3
    #     self.reference = ref_steel

    #     bes_state_reference = 1200000
    #     h2s_state_reference = 320467

    #     objective_value = 0
    #     for i in range(self.horizon):

    #         # ysp = (
    #         #     self.Csp @ x[:, i] + self.Dspc @ uc[:, i] + self.Dsps @ us[:, i]
    #         # )  # + self.Fsp @ de

    #         tracking_term = (ref_steel - ys[0, i]) ** 2

    #         # h2s_sparsity = 1e-5 * (ysp[3] ** 2 + ysp[5] ** 2) ** 2
    #         # h2s_sparsity = 1e-5 * (ysp[3] * ysp[5] ) **2
    #         # h2s_sparsity = 1e3 * ((ysp[3] +  ysp[5]) + ca.fabs(uc[1,i]) )
    #         # h2s_sparsity = us[3,i] ** 2 - uc[1, i]**2
    #         # h2s_sparsity = 1e3 * ca.if_else(
    #         #     uc[1, i] >= 0, (uc[1, i] - us[3, i]) ** 2, 0
    #         # )
    #         # h2s_sparsity = (2 * us[3, i] - uc[1, i] - ca.fabs(uc[1, i]))
    #         # h2s_sparsity = 1e3 * (ysp[3] -  uc[1,i] ) **2
    #         # h2s_sparsity = 1e3 * (ysp[3] + ysp[5]) **2
    #         # bes_sparsity = 1e-5 * (ysp[0] ** 2 + ysp[2] ** 2) ** 2
    #         # bes_sparsity = 1e-5 * (ysp[0] * ysp[2]) ** 2

    #         # no_h2s_charge = 1e3 * uc[1, i] ** 2

    #         # bes_state = 1e-5 * (x[0, i] - bes_state_reference) ** 2
    #         # h2s_state = 1e-3 * (x[1, i] - h2s_state_reference) ** 2

    #         # curtail_penalty = 1e-3 * curtail[0, i] ** 2
    #         # storage_agreement = 1e-3 * (0.0218 * uc[0, i] - uc[1, i]) ** 2

    #         objective_value += (
    #             tracking_term
    #             # + bes_state
    #             # + h2s_state
    #             # + storage_agreement
    #             # + h2s_sparsity
    #             # + bes_sparsity
    #             # + curtail_penalty
    #         )

    #     return objective_value

    def update_optimization_parameters(self, x0, src_forecast):
        self.opti.set_value(self.opt_params["dex"], src_forecast)
        self.opti.set_value(self.opt_params["x0"], x0)
        if not self.allow_curtail_forecast:
            self.opti.set_value(
                self.opt_params["curtail"], np.zeros(self.opt_params["curtail"].shape)
            )

        if not self.allow_grid_purchase:
            self.opti.set_value(
                self.opt_params["grid"], np.zeros(self.opt_params["grid"].shape)
            )

    def update_optimization_constraints(self):
        pass

    def compute_trajectory(self, x0, forecast, step_index=0):
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
                    self.opti.set_initial(self.opt_vars["uct"], self.uc_init)
                    self.opti.set_initial(self.opt_vars["usp"], self.us_init)
                    self.opti.set_initial(self.opt_vars["x"], self.x_init)
                    self.opti.set_initial(self.opt_vars["yex"], self.ys_init)

            try:

                # t0 = time.time()

                sol = self.opti.solve()

                # t1 = time.time()
                # print(self.s_opts)
                # print(f"Solution took: {t1-t0:.4f} seconds")

            except:

                with Capturing() as output:
                    self.opti.debug.show_infeasibilities()

                # print()

                output2 = []

                violations = []

                i = 0
                while i < len(output):
                    if output[i].startswith(
                        "------- i = "
                    ):  # new constraint description
                        num_description = output[i + 1]
                        line_number = output[i + 2]
                        code_description = output[i + 3]
                        at_description = ""
                        # at_description = output[i + 4]

                        violation = float(
                            num_description.split("viol ")[1].split(")")[0]
                        )
                        violations.append(violation)

                        if violation >= 1e-3:
                            if "opti.subject" in code_description:
                                code_desc = code_description.split("opti.subject_to(")[
                                    1
                                ][:-1]
                            else:
                                code_desc = ""

                            print_line = (
                                str(num_description).ljust(45)
                                + code_desc.ljust(130)
                                + at_description
                            )
                            pprint.pprint(print_line, width=200)

                            # print(code_description)
                        i += 4
                    i += 1

                # if self.horizon == 1:
                #     x_db = self.opti.debug.value(self.opt_vars["x"])  # [None, :]
                #     uc_db = np.atleast_2d(self.opti.debug.value(self.opt_vars["uct"])).T  # [None, :]
                #     us_db = np.atleast_2d(self.opti.debug.value(self.opt_vars["usp"])).T
                #     ys_db = np.atleast_2d(self.opti.debug.value(self.opt_vars["yex"]))# [None, :]
                #     yco_db = np.atleast_2d(self.opti.debug.value(self.opt_vars["yco"])).T
                #     curtail_db = self.opti.debug.value(self.opt_vars["curtail"])
                #     grid_db = self.opti.debug.value(self.opt_vars["grid"])
                # else:
                #     x_db = self.opti.debug.value(self.opt_vars["x"])  # [None, :]
                #     uc_db = self.opti.debug.value(self.opt_vars["uct"])  # [None, :]
                #     us_db = self.opti.debug.value(self.opt_vars["usp"])
                #     ys_db = self.opti.debug.value(self.opt_vars["yex"])# [None, :]
                #     yco_db = self.opti.debug.value(self.opt_vars["yco"])
                #     curtail_db = self.opti.debug.value(self.opt_vars["curtail"])
                #     grid_db = self.opti.debug.value(self.opt_vars["grid"])

                # def get_sol_value(prob:ca.Opti, var):
                #     val = prob.value(var)
                #     val = np.reshape(val, var.shape)
                #     return val

                x_db = get_sol_value(self.opti.debug, self.opt_vars["x"]) 
                uc_db = get_sol_value(self.opti.debug, self.opt_vars["uct"])
                us_db = get_sol_value(self.opti.debug, self.opt_vars["usp"])
                ys_db = get_sol_value(self.opti.debug, self.opt_vars["yex"])
                yco_db = get_sol_value(self.opti.debug, self.opt_vars["yco"])
                if self.grid_curtail_mod:
                    gridcurtail = get_sol_value(self.opti.debug, self.opt_vars["gridcurtail"])
                    grid_db = np.where(gridcurtail >= 0, gridcurtail, 0)
                    curtail_db = np.where(gridcurtail <= 0, -gridcurtail, 0)
                else:

                    curtail_db = get_sol_value(self.opti.debug, self.opt_vars["curtail"])
                    grid_db = get_sol_value(self.opti.debug, self.opt_vars["grid"])

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

                np.set_printoptions(linewidth=200, suppress=True, precision=4)

                # if not self.debug_mode:
                #     self.save_state_for_debug(x0, forecast, step_index)

                assert np.max(np.abs(violations)) <= 1e-3

                self.bad_solve_count += 1
                self.bad_solve_step.append(step_index)
                self.bad_solve_violation.append(np.max(np.abs(violations)))
                sol = self.opti.debug

                plt.close()

            self.prev_sol = sol

            try:
                jac_uct = sol.value(ca.jacobian(self.opti.f, self.opt_vars["uct"]))
                jac_usp = sol.value(ca.jacobian(self.opti.f, self.opt_vars["usp"]))
                jac_x = sol.value(ca.jacobian(self.opti.f, self.opt_vars["x"]))
                jac_yex = sol.value(ca.jacobian(self.opti.f, self.opt_vars["yex"]))
                # jac_yco = sol.value(ca.jacobian(self.opti.f, self.opt_vars["uct"]))

                jac = sol.value(ca.jacobian(self.opti.f, self.opti.x)).toarray()[0]
                # jac = self.opti.debug.value(ca.jacobian(self.opti.debug.f, self.opti.debug.x)).toarray()[0]
                assert (np.abs(jac) < 1).any()
                # True
            except:
                np.set_printoptions(linewidth=200, suppress=True, precision=4)

                uc_slice = slice(0, self.mct * self.horizon)
                us_slice = slice(
                    self.mct * self.horizon, (self.mct + self.msp) * self.horizon
                )
                x_slice = slice(
                    (self.mct + self.msp) * self.horizon,
                    (self.mct + self.msp) * self.horizon + self.n * (self.horizon + 1),
                )
                ys_slice = slice(
                    (self.mct + self.msp) * self.horizon + self.n * (self.horizon + 1),
                    (self.mct + self.msp) * self.horizon
                    + self.n * (self.horizon + 1)
                    + self.pse * self.horizon,
                )

                jac_uc = np.reshape(jac[uc_slice], (self.horizon, self.mc))
                jac_us = np.reshape(jac[us_slice], (self.horizon, self.ms))
                jac_x = np.reshape(jac[x_slice], (self.horizon + 1, self.n))
                jac_ys = np.reshape(jac[ys_slice], (self.horizon, self.pse))

                self.print_block_matrices(
                    mat=[[jac_uc, jac_us, jac_x[0 : self.horizon, :], jac_ys]],
                    in_labels=["jac uc", "jac us", "jac x", "jac yex"],
                    out_labels=[f"step {i}" for i in range(self.horizon)],
                )

                []

            # self.opti.debug.value_parameters()
            # self.opti.debug.value_variables()
            # self.opti.debug.stats()
            # self.opti.debug.arg()
            # self.opti.debug.constraints()
            # self.opti.debug.show_infeasibilities()

            uct = get_sol_value(sol, self.opt_vars["uct"])
            usp = get_sol_value(sol, self.opt_vars["usp"])
            x = get_sol_value(sol, self.opt_vars["x"])
            yex = get_sol_value(sol, self.opt_vars["yex"])
            yco = get_sol_value(sol, self.opt_vars["yco"])
            dex = get_sol_value(sol, self.opt_params["dex"])   #[None, :]
            if self.grid_curtail_mod:
                gridcurtail = get_sol_value(sol, self.opt_vars["gridcurtail"])
                grid = np.where(gridcurtail >= 0, gridcurtail, 0)
                curtail = np.where(gridcurtail <= 0, -gridcurtail, 0)
            else:
                if self.allow_curtail_forecast:
                    curtail = get_sol_value(sol, self.opt_vars["curtail"])
                else:
                    curtail = get_sol_value(sol, self.opt_params["curtail"])
                if self.allow_grid_purchase:
                    grid = get_sol_value(sol, self.opt_vars["grid"])
                else:
                    grid = get_sol_value(sol, self.opt_params["grid"])

            # uct = sol.value(self.opt_vars["uct"])
            # usp = sol.value(self.opt_vars["usp"])
            # x = sol.value(self.opt_vars["x"])
            # yex = sol.value(self.opt_vars["yex"])
            # yco = sol.value(self.opt_vars["yco"])
            # # e = sol.value(self.opt_vars["e"])
            # dex = sol.value(self.opt_params["dex"])   #[None, :]
            # if self.allow_curtail_forecast:
            #     curtail = sol.value(self.opt_vars["curtail"])
            # else:
            #     curtail = sol.value(self.opt_params["curtail"])
            # if self.allow_grid_purchase:
            #     grid = sol.value(self.opt_vars["grid"])
            # else:
            #     grid = sol.value(self.opt_params["grid"])

            self.curtail_storage[step_index : step_index + self.horizon] = curtail

            # def dimension_check(arr:np.ndarray):
            #     if arr.ndim != 2:
            #         return arr[:, None]
            #     else:
            #         return arr

            self.uc_init = uct
            self.us_init = usp
            self.x_init = x
            self.ys_init = yex
            self.curtail_init = curtail

            ysp = (
                self.Cco @ x[:, :-1]
                # + self.Dcoct @ uct
                + self.Dcoct @ uct
                + self.Dcosp @ usp
                + self.Fcoex @ (dex - curtail)
            )
            # This is what worked with a horizon of 1. Try to make it flexible for all cases
            # ysp = (
            #     self.Cco @ np.atleast_2d(x)[:, :-1]
            #     # + self.Dcoct @ uct
            #     + self.Dcoct @ np.atleast_2d(uct).T
            #     + self.Dcosp @ np.atleast_2d(usp).T
            #     + self.Fcoex @ np.atleast_2d(dex - curtail)
            # )
            # coupling disturbances
            dco = self.M_dco_yco @ ysp

            ysp = np.concatenate([ysp, yex])
            # ysp = np.concatenate([ysp, np.atleast_2d(yex)])
            # ysp = np.concatenate([ysp, ys[None, :]])

            obj_values = {
                key: sol.value(self.obj_terms[key]) for key in self.obj_terms.keys()
            }
            obj_values_uw = {
                key: sol.value(self.obj_terms_uw[key])
                for key in self.obj_terms_uw.keys()
            }

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

            if False:

                mat_co = (
                    self.Csp @ x[:, 0 : self.horizon]
                    + self.Dspc @ uc
                    + self.Dsps @ us
                    + self.Fsp @ de
                )
                mat_ex = (
                    self.Cs @ x[:, 0 : self.horizon]
                    + self.Dsc @ uc
                    + self.Dss @ us
                    + self.Fs @ de
                )
                self.print_block_matrices(
                    [[mat_co[None, i, :]] for i in range(mat_co.shape[0])]
                    + [list(mat_ex[None, :])],
                    in_labels=[""],
                    out_labels=self.ys_list,
                    no_space=True,
                )

            # if self.mct == 1:
            #     u_ctrl = uct[None, 0]
            # else:
            #     u_ctrl = uct[:, 0]

            # if curtail.ndim < 2:
            #     curtail = curtail[None, :]

            # self.plot_solution(sol, forecast)
            # self.plot_solution(uct, usp, x, ysp, forecast)

            # self.plot_trajectory(step_index)

            # u_split = usp[:, 0]
            # if not self.debug_mode:
            #     self.save_state_for_debug(x0, forecast, step_index)

            return uct, usp, curtail, grid

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

        []

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
            for i in range(n):
                x_labels.append(f"x {i} {node}")

            mct = cm.B.shape[1]
            msp = usp_degree
            m = mct + msp

            # create controllable input label lists

            uct_indices = []

            uct_labels = []
            for i in range(mct):
                uct_indices.append(int(len(uct_labels) + np.sum(dims["dims"]["mct"])))
                uct_labels.append(f"uct {i} {node}")

            if len(uct_indices) > 0:
                uct_order.update({node: uct_indices})

            usp_indices = []
            usp_labels = []
            for i in range(usp_degree):
                usp_indices.append(int(len(usp_labels) + np.sum(dims["dims"]["msp"])))
                out_edges = list(G.out_edges(node))
                usp_labels.append(f"usp {i} {node} (to {out_edges[i][1]})")

            if len(usp_indices) > 0:
                usp_order.update({node: usp_indices})

            # create uncontrollable input label lists
            dex_labels = []
            dco_labels = []

            if G.nodes[node]["is_source"]:
                oex = cm.F.shape[1]
                assert in_degree == 1
                oco = 0

                for i in range(oex):
                    dex_labels.append(f"dex {i} {node}")

            else:
                oex = 0
                oco = cm.F.shape[1] * in_degree

                for i in range(in_degree):
                    in_edges = list(G.in_edges(node))
                    dco_labels.append(f"dco {i} {node} (from {in_edges[i][0]})")

            o = oex + oco

            # create output label lists

            yex_labels = []
            yco_labels = []
            yze_labels = []
            yet_labels = []
            ygt_labels = []

            if G.nodes[node]["is_sink"]:
                pex = cm.C.shape[0]
                for i in range(pex):
                    yex_labels.append(f"yex {i} {node}")
            else:
                pex = 0

            if usp_degree > 0:
                pze = cm.C.shape[0]
                assert pex == 0, "sink node should not be splitting"
                pco = usp_degree

                for i in range(pze):
                    yze_labels.append(f"yze {i} {node}")

            else:
                pze = 0
                pco = cm.C.shape[0] - pex

            if not G.nodes[node]["is_sink"]:
                for i in range(out_degree):
                    out_edges = list(G.out_edges(node))
                    yco_labels.append(f"yco {i} {node} (to {out_edges[i][1]})")

            pet = cm.C_et.shape[0]
            pgt = cm.C_gt.shape[0]

            for i in range(pet):
                yet_labels.append(f"yet {i} {node}")

            for i in range(pgt):
                ygt_labels.append(f"ygt {i} {node}")

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

        np.block(
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
            [Eco @ MFi @ Cco, Eco @ MFi @ Dcoct, Eco @ MFi @ Dcosp, Eco @ MFi @ Fcoex],
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
        #     out_labels=["x+", "yco", "yex", "yze", "ygt", "yet"],
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
        self.uct_order = uct_order
        self.usp_order = usp_order

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
            n_nodes, 5, sharex="all", layout="constrained", figsize=(15, 10), dpi=100
        )

        ax[0, 0].set_title("Disturbance")
        ax[0, 1].set_title("Control input")
        ax[0, 2].set_title("State")
        ax[0, 3].set_title("Output")
        ax[0, 4].set_title("Split")

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
            plot_one(ax[i, 4], self.ysp_store, y_inds)
            ax[i, 4].set_ylim(ax[i, 3].get_ylim())

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
