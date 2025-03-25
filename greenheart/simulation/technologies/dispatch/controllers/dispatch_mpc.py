import numpy as np
import scipy
import networkx as nx
import matplotlib.pyplot as plt
import casadi as ca
import pprint
import sys
from io import StringIO
import pickle

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
        node_order=None,
        edge_order=None,
        mpc_config=None,
    ):

        system_graph = load_yaml(
            config.greenheart_config["realtime_simulation"]["system"][
                "system_graph_config"
            ]
        )

        nodes = system_graph["traversal_order"]
        traversal_order = system_graph["traversal_order"]

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

        if "battery" in self.node_order:
            self.ref_bes_state = (
                0.7 * simulation_graph.nodes["battery"]["ionode"].model.max_capacity_kWh
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

        self.traversal_order = traversal_order

        self.allow_curtail_forecast = True
        self.allow_grid_purchase = True
        self.include_edges = True
        self.use_sparsity_constraint = False

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

        self.curtail_storage = np.zeros(8760 + self.horizon)

        self.bad_solve_count = 0
        self.bad_solve_step = []
        self.bad_solve_violation = []
        self.prev_sol = None

        # self.build_control_model(traversal_order, simulation_graph)
        self.collect_system_matrices(traversal_order, simulation_graph)
        self.setup_optimization()
        self.setup_solution_storage()

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
        p_opts = {"print_time": False, "verbose": False}
        s_opts = {
            "print_level": 0,
            # "linear_solver": "ma27",
            # "max_iter": 2000,
            # "tol": 1e-3
            # "jac_c_constant": "yes",
            # "jac_d_constant": "yes",
        }
        opti.solver(
            "ipopt",
            p_opts,
            s_opts,
        )

        # Variables
        uct_var = opti.variable(self.mct, self.horizon)
        usp_var = opti.variable(self.msp, self.horizon)
        x_var = opti.variable(self.n, self.horizon + 1)
        yex_var = opti.variable(self.pex, self.horizon)
        # e_var = opti.variable(self.q + 2, self.horizon)
        if self.allow_curtail_forecast:
            curtail = opti.variable(self.oex, self.horizon)
        else:
            curtail = opti.parameter(self.oex, self.horizon)

        if self.allow_grid_purchase:
            grid_purchase = opti.variable(self.oex, self.horizon)
            opti.subject_to(grid_purchase >= np.zeros(grid_purchase.shape))
        else:
            grid_purchase = opti.parameter(self.oex, self.horizon)

        # Parameters
        dex_param = opti.parameter(self.oex, self.horizon)
        # e_src_param = opti.parameter(1, self.horizon)
        x0_param = opti.parameter(self.n, 1)

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

            xkp1 = (
                self.A @ x_var[:, i]
                + self.Bct @ uct_var[:, i]
                + self.Bsp @ usp_var[:, i]
                + self.Eex @ (dex_param[:, i] - curtail[:, i] + grid_purchase[:, i])
            )
            # external outputs
            yexk = (
                self.Cex @ x_var[:, i]
                + self.Dexct @ uct_var[:, i]
                + self.Dexsp @ usp_var[:, i]
                + self.Fexex @ (dex_param[:, i] - curtail[:, i] + grid_purchase[:, i])
            )

            # coupling outputs
            yco = (
                self.Cco @ x_var[:, i]
                + self.Dcoct @ uct_var[:, i]
                + self.Dcosp @ usp_var[:, i]
                + self.Fcoex @ (dex_param[:, i] - curtail[:, i] + grid_purchase[:, i])
            )

            # Splitting constraint zero outputs
            yze = (
                self.Cze @ x_var[:, i]
                + self.Dzect @ uct_var[:, i]
                + self.Dzesp @ usp_var[:, i]
                + self.Fzeex @ (dex_param[:, i] - curtail[:, i] + grid_purchase[:, i])
            )

            # greater than 0 constraint outputs
            ygt = (
                self.Cgt @ x_var[:, i]
                + self.Dgtct @ uct_var[:, i]
                + self.Dgtsp @ usp_var[:, i]
                + self.Fgtex @ (dex_param[:, i] - curtail[:, i] + grid_purchase[:, i])
            )

            # equal to 0 constraint outputs
            yet = (
                self.Cet @ x_var[:, i]
                + self.Detct @ uct_var[:, i]
                + self.Detsp @ usp_var[:, i]
                + self.Fetex @ (dex_param[:, i] - curtail[:, i] + grid_purchase[:, i])
            )

            opti.subject_to(x_var[:, i + 1] == xkp1)
            opti.subject_to(yex_var[:, i] == yexk[0])
            if self.pze > 0:
                opti.subject_to(yze == np.zeros((self.pze, 1)))
            if self.pgt > 0:
                opti.subject_to(ygt == np.zeros((self.pgt, 1)))
            if self.pet > 0:
                opti.subject_to(yet == np.zeros((self.pet, 1)))

            if self.use_sparsity_constraint:
                opti.subject_to(
                    usp_var[2, i] == ca.if_else(uct_var[1, i] >= 0, uct_var[1, i], 0)
                )
                opti.subject_to(
                    usp_var[0, i] == ca.if_else(uct_var[0, i] >= 0, uct_var[0, i], 0)
                )

            step_obj, step_obj_terms = self.objective_step(
                x_var[:, i],
                uct_var[:, i],
                usp_var[:, i],
                yco,
                yexk,
                curtail[:, i],
                grid_purchase[:, i],
                objective_var_inds,
            )
            objective += step_obj
            objective_terms.append(step_obj_terms)

            # opti.subject_to(us_var[2, i] <= ca.fabs(uc_var[1,i]))

            opti.subject_to(usp_var[:, i] >= np.zeros(self.msp))
            # opti.subject_to(e_var[:, i] >= np.zeros(self.q + 2))

            opti.subject_to(uct_var[:, i] >= self.bounds["u_lb"])
            opti.subject_to(uct_var[:, i] <= self.bounds["u_ub"])

            opti.subject_to(x_var[:, i] >= self.bounds["x_lb"])
            opti.subject_to(x_var[:, i] <= self.bounds["x_ub"])
            # opti.subject_to(uct_var[:, i] >= self.u_lb_list)
            # opti.subject_to(uct_var[:, i] <= self.u_ub_list)

            # opti.subject_to(x_var[:, i] >= self.x_lb_list)
            # opti.subject_to(x_var[:, i] <= self.x_ub_list)

            # opti.subject_to((self.Ds_flat @ ys_var[:, i]) >= self.y_lb_list)
            # opti.subject_to((self.Ds_flat @ ys_var[:, i]) <= self.y_ub_list)

            # trickier constraints - maybe wont work

            # opti.subject_to(yco[0] == uct_var[0, i])
            # opti.subject_to(yco[4] == uct_var[2, i])

            # opti.subject_to(uct_var[0,i] <= yco[0])
            # opti.subject_to(uct_var[2,i] <= yco[4])

        if self.allow_curtail_forecast:
            opti.subject_to(curtail <= dex_param)
            opti.subject_to(curtail >= np.zeros(curtail.shape))

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
        }
        self.opt_params = {"dex": dex_param, "x0": x0_param}

        if self.allow_curtail_forecast:
            self.opt_vars.update({"curtail": curtail})
        else:
            self.opt_params.update({"curtail": curtail})

        if self.allow_grid_purchase:
            self.opt_vars.update({"grid": grid_purchase})
        else:
            self.opt_params.update({"grid": grid_purchase})

    def objective_step(self, x, uct, usp, yco, yex, curtail, grid, var_inds=None):

        # =============================================================================
        # ==                                                                         ==
        # ==                                Objective                                ==
        # ==                                                                         ==
        # =============================================================================
        obj_terms = {}

        term_keys = [
            "output_tracking",
            # "curtail",
            "grid_purchase",
        ]

        
        # output_tracking = (ref_steel - yex) ** 2
        obj_terms.update(
            {"output_tracking": {"w": 1e9, "expr": (self.reference - yex) ** 2}}
        )

        # h2_tracking = 1e-3 *(2084.6 - (yco[5] + yco[6]))**2
        # h2_tracking = (2084.6 - (yco[6] + yco[7])) ** 2

        # obj_terms.update({"curtail": {"w": 1e-4, "expr": curtail**2}})
        # grid_purchase = grid
        obj_terms.update({"grid_purchase": {"w": 1e-4, "expr": grid**2}})

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

            term_keys.append("tes_simultaneous")
            # term_keys.append("tes_state")

        if self.use_config_weights:
            for term in term_keys:
                obj_terms[term]["w"] = self.weights[term]

        objective = 0
        for term in term_keys:

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

    def objective(self, x, uc, us, ys, curtail):

        ref_steel = 45.48e3
        # ref_steel = 37.92e3
        # ref_steel = 20e3
        self.reference = ref_steel

        bes_state_reference = 1200000
        h2s_state_reference = 320467

        objective_value = 0
        for i in range(self.horizon):

            # ysp = (
            #     self.Csp @ x[:, i] + self.Dspc @ uc[:, i] + self.Dsps @ us[:, i]
            # )  # + self.Fsp @ de

            tracking_term = (ref_steel - ys[0, i]) ** 2

            # h2s_sparsity = 1e-5 * (ysp[3] ** 2 + ysp[5] ** 2) ** 2
            # h2s_sparsity = 1e-5 * (ysp[3] * ysp[5] ) **2
            # h2s_sparsity = 1e3 * ((ysp[3] +  ysp[5]) + ca.fabs(uc[1,i]) )
            # h2s_sparsity = us[3,i] ** 2 - uc[1, i]**2
            # h2s_sparsity = 1e3 * ca.if_else(
            #     uc[1, i] >= 0, (uc[1, i] - us[3, i]) ** 2, 0
            # )
            # h2s_sparsity = (2 * us[3, i] - uc[1, i] - ca.fabs(uc[1, i]))
            # h2s_sparsity = 1e3 * (ysp[3] -  uc[1,i] ) **2
            # h2s_sparsity = 1e3 * (ysp[3] + ysp[5]) **2
            # bes_sparsity = 1e-5 * (ysp[0] ** 2 + ysp[2] ** 2) ** 2
            # bes_sparsity = 1e-5 * (ysp[0] * ysp[2]) ** 2

            # no_h2s_charge = 1e3 * uc[1, i] ** 2

            # bes_state = 1e-5 * (x[0, i] - bes_state_reference) ** 2
            # h2s_state = 1e-3 * (x[1, i] - h2s_state_reference) ** 2

            # curtail_penalty = 1e-3 * curtail[0, i] ** 2
            # storage_agreement = 1e-3 * (0.0218 * uc[0, i] - uc[1, i]) ** 2

            objective_value += (
                tracking_term
                # + bes_state
                # + h2s_state
                # + storage_agreement
                # + h2s_sparsity
                # + bes_sparsity
                # + curtail_penalty
            )

        return objective_value

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

            self.update_optimization_parameters(x0, forecast)
            if hasattr(self, "x_init"):  # then try to update initial guess
                self.opti.set_initial(self.opt_vars["uct"], self.uc_init)
                self.opti.set_initial(self.opt_vars["usp"], self.us_init)
                self.opti.set_initial(self.opt_vars["x"], self.x_init)
                self.opti.set_initial(self.opt_vars["yex"], self.ys_init)

            try:

                sol = self.opti.solve()

            except:

                with Capturing() as output:
                    self.opti.debug.show_infeasibilities()

                print()

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

                        if violation >= 1e-6:
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

                x_db = self.opti.debug.value(self.opt_vars["x"])  # [None, :]
                uc_db = self.opti.debug.value(self.opt_vars["uct"])  # [None, :]
                us_db = self.opti.debug.value(self.opt_vars["usp"])
                ys_db = self.opti.debug.value(self.opt_vars["yex"])[None, :]
                curtail_db = self.opti.debug.value(self.opt_vars["curtail"])
                grid_db = self.opti.debug.value(self.opt_vars["grid"])

                fig, ax = plt.subplots(
                    np.max(
                        [uc_db.shape[0], us_db.shape[0], x_db.shape[0], ys_db.shape[0]]
                    ),
                    4,
                    sharex="all",
                    layout="constrained",
                )

                to_plot = [x_db, uc_db, us_db, ys_db]
                titles = [self.n_label, self.mct_label, self.msp_label, self.pex_label]
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

                # assert False, self.opti.debug

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

            uct = sol.value(self.opt_vars["uct"])
            usp = sol.value(self.opt_vars["usp"])
            x = sol.value(self.opt_vars["x"])
            yex = sol.value(self.opt_vars["yex"])
            # e = sol.value(self.opt_vars["e"])
            dex = sol.value(self.opt_params["dex"])[None, :]
            if self.allow_curtail_forecast:
                curtail = sol.value(self.opt_vars["curtail"])
            else:
                curtail = sol.value(self.opt_params["curtail"])
            if self.allow_grid_purchase:
                grid = sol.value(self.opt_vars["grid"])
            else:
                grid = sol.value(self.opt_params["grid"])

            self.curtail_storage[step_index : step_index + self.horizon] = curtail

            self.uc_init = uct
            self.us_init = usp
            self.x_init = x
            self.ys_init = yex
            self.curtail_init = curtail

            ysp = (
                self.Cco @ np.atleast_2d(x)[:, :-1]
                + self.Dcoct @ np.atleast_2d(uct)
                + self.Dcosp @ usp
                + self.Fcoex @ (dex - curtail)
            )
            # coupling disturbances
            dco = self.M_dco_yco @ ysp

            ysp = np.concatenate([ysp, np.atleast_2d(yex)])
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

            if self.mct == 1:
                u_ctrl = uct[None, 0]
            else:
                u_ctrl = uct[:, 0]

            if curtail.ndim < 2:
                curtail = curtail[None, :]

            # self.plot_solution(sol, forecast)
            # self.plot_solution(uct, usp, x, ysp, forecast)

            # self.plot_trajectory(step_index)

            u_split = usp[:, 0]

            return uct, usp, curtail, grid

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

        # self.print_block_matrices(
        #     [combined_mat[i] for i in [0, 2, 3, 4, 5]],
        #     in_labels=["x", "uct", "usp", "dex"],
        #     out_labels=["x+", "yex", "yze", "ygt", "yet"],
        # )
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

        self.E_inc = E_inc
        self.P_in = P_in
        self.P_out = P_out

        for key in bounds.keys():
            bounds[key] = np.concatenate(bounds[key])

        self.bounds = bounds
        self.uct_order = uct_order
        self.usp_order = usp_order

        self.solve_steady_reference()

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

    def print_block_matrices(self, mat, in_labels, out_labels, no_space=False):

        try:
            np.block(mat)
        except:
            AssertionError("bad matrix")

        block_mat = np.block(mat)
        # block_cols = block_mat.shape[1]
        # block_rows = block_mat.shape[0]

        out_label_width = int(np.max([len(label) for label in out_labels]))
        num_col_width = 10

        for row_num, row_mat in enumerate(mat):
            if not no_space:
                print("")

            row_mat_lens = [matr.shape[1] for matr in row_mat]
            if row_num == 0:
                line = " " * (out_label_width + 5)
                line2 = " " * (out_label_width + 5 + 4)
                for coli, col_label in enumerate(in_labels):
                    line += f"{col_label}".ljust(row_mat_lens[coli] * num_col_width + 3)
                    for j in range(row_mat_lens[coli]):
                        line2 += f"{j}".ljust(num_col_width)
                    line2 += " " * 3

                print(line)
                print(line2)
            n_rows = row_mat[0].shape[0]
            for i in range(n_rows):
                line = f"{out_labels[row_num]}".ljust(out_label_width + 3)
                line += "["
                for col_mat in row_mat:
                    for j in range(col_mat.shape[1]):
                        line += f"{col_mat[i,j] :.4g}, ".rjust(num_col_width)

                    line = line[0:-2]
                    line += " ]  ["
                line = line[0:-2]
                print(line)

        []


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout
