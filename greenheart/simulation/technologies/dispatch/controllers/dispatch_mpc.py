import numpy as np
import scipy
import networkx as nx
import matplotlib.pyplot as plt
import casadi as ca
import pprint
import sys
from io import StringIO
import pickle
import logging
from logging import handlers

import time

from hopp.utilities import load_yaml

from greenheart.simulation.technologies.dispatch.controllers.controller_tools.control_model_builder import ControlModelBuilder
from greenheart.simulation.technologies.dispatch.controllers.controller_tools.gradient_helper import GradientHelper
from greenheart.simulation.technologies.dispatch.controllers.controller_tools.plotter_helper import MPCPlotter
from greenheart.simulation.technologies.dispatch.controllers.controller_tools.objective_helper import Objective


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
        p_opts={"print_time": False, "verbose": False, "record_time":True},
        s_opts={"print_level": 0, "compl_inf_tol": 1e-3, "max_iter":1e5},
        # s_opts={"print_level": 0, "compl_inf_tol": 1e-3, "max_iter":2e5},
        debug_mode=False,
    ):

        self.debug_mode = debug_mode
        self.verbose = False


        self.mpc_config = mpc_config

        if "logging" in self.mpc_config:
            self.setup_logging(self.mpc_config.pop("logging"))


        self.use_objective_class = True

        if mpc_config is not None:
            options = mpc_config["options"]

            # Option flags
            self.use_config_weights = options["use_config_weights"]
            self.warm_start_with_previous_solution = options["warm_start"]
            self.use_NL_electrolzyer = options["use_NL_electrolyzer"]
            self.NL_EL_order = options["NL_order"]
            self.only_bounded_yco = options["only_bounded_yco"]
            self.no_shortfall = options["no_shortfall"]
            self.terminal_cost = options["terminal_cost"]
            self.terminal_constraint = options["terminal_constraint"]

            # self.constrain_output_tracking = options["constrain_output_tracking"]
            self.horizon = mpc_config["horizon"]

        else:

            # Option flags
            self.use_config_weights = True
            self.warm_start_with_previous_solution = True
            self.use_NL_electrolzyer = False
            self.NL_EL_order = 1
            self.only_bounded_yco = True
            self.no_shortfall = True
            self.terminal_cost = True
            self.terminal_constraint = False

            # self.constrain_output_tracking = False

            self.horizon = 5

        # if self.no_shortfall:
        #     print(f"{self.no_shortfall = }")

        # if self.use_NL_electrolzyer:
        #     print(f"{self.use_NL_electrolzyer = }, {self.NL_EL_order = }")

        # print(f"{self.terminal_cost = }")

        self.p_opts = p_opts
        self.s_opts = s_opts

        self.setup_solution_storage()
        self.curtail_storage = np.zeros(8760 + self.horizon)

        if self.debug_mode:
            self.load_state_for_debug(saved_state)
            self.use_saved_solution = False

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

            self.G = simulation_graph

            self.control_model = ControlModelBuilder(mpc=self)
            self.control_model.build_control_model(traversal_order, simulation_graph)

            # self.collect_system_matrices(traversal_order, simulation_graph)

            

            self.use_saved_solution = (
                "use_saved_solution"
                in config.greenheart_config["realtime_simulation"]["dispatch"]["mpc"]
            )

            if "battery" in self.node_order:
                self.x_bes_max = simulation_graph.nodes["battery"]["ionode"].model.max_capacity_kWh
                self.x_bes_min = simulation_graph.nodes["battery"]["ionode"].model.min_capacity_kWh
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
                self.x_h2s_max = simulation_graph.nodes["hydrogen_storage"][
                        "ionode"
                    ].model.max_capacity_kg
                self.x_h2s_min = simulation_graph.nodes["hydrogen_storage"]["ionode"].model.min_capacity_kg
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
                self.x_tes_max = simulation_graph.nodes["thermal_energy_storage"][
                        "ionode"
                    ].model.H_capacity_kWh
                self.x_tes_min = 0
                # self.x_tes_min = simulation_graph.nodes["thermal_energy_storage"]["ionode"].model.H_buffer_max_kWh
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
        if "reference" in mpc_config:
            self.reference = mpc_config["reference"]
        # else:
        #     ref_steel = 165
        #     self.reference = ref_steel

        if "weights" in mpc_config:
            self.use_config_weights = True
            self.weights = mpc_config["weights"]
        if "terms" in mpc_config:
            self.term_keys = mpc_config["terms"]

        self.objective_manager = Objective(
            horizon=self.horizon,
            active_terms=self.term_keys,
            weights=self.weights,
            references=dict(steel=self.reference, x_bes=self.ref_bes_state, x_tes=self.ref_tes_state, x_h2s=self.ref_h2s_state, soc_bes=self.mpc_config["references"]["bes"], soc_tes=self.mpc_config["references"]["tes"], soc_h2s=self.mpc_config["references"]["h2s"]),
            capacities=dict(x_bes_max=self.x_bes_max, x_bes_min=self.x_bes_min, x_tes_max= self.x_tes_max, x_tes_min=self.x_tes_min, x_h2s_max=self.x_h2s_max, x_h2s_min=self.x_h2s_min),
            var_inds=self.get_objective_var_inds()
        )

        if self.use_saved_solution:
            self.load_stored_values(
                config.greenheart_config["realtime_simulation"]["dispatch"]["mpc"][
                    "use_saved_solution"
                ]
            )

        self.setup_optimization()

        if self.horizon == 1:
            self.warm_start_with_previous_solution = False

        self.bad_solve_count = 0
        self.bad_solve_step = []
        self.bad_solve_violation = []
        self.prev_sol = None

    def set_no_shortfall_bool(self, no_shortfall: bool):
        self.no_shortfall = no_shortfall
        self.setup_optimization()

    def set_use_NL_electrolyzer(self, use_NL: bool):
        self.use_NL_electrolzyer = use_NL
        self.setup_optimization()

    def set_terminal_cost_bool(self, terminal_bool):
        self.terminal_cost = terminal_bool
        self.setup_optimization()



    # @property
    # def labels(self):
    #     return self.labels
    
    # @labels.setter
    # def labels(self, labels):
    #     self.labels = labels





    def setup_logging(self, log_config):



        self.logger = logging.getLogger(f"MPC {log_config['case_description']}")
        self.logger.setLevel(logging.DEBUG)
    
        
        queue_handler = handlers.QueueHandler(log_config["queue"])
        queue_handler.setLevel(logging.DEBUG)
        self.logger.addHandler(queue_handler)

        self.logger.info("Logger initialized")



      

    def setup_solution_storage(self):
        self.step_index_store = []
        self.uct_store = []
        self.usp_store = []
        self.x_store = []
        self.yex_store = []
        self.ysp_store = []
        self.forecast_store = []
        self.x0_store = []
        self.curtail_store = []
        self.grid_store = []
        self.de_store = []
        self.dco_store = []
        self.objective_store = []
        self.objective_uw_store = []

        self.solstats_step_index_store = []
        self.iter_count_store = []
        self.t_wall_total_store = []
        # self.t_wall_func = []
        # self.t_wall_grad = []
        self.t_proc_total_store = []

    def store_solve_stats(self, stats, step_index):
        self.solstats_step_index_store.append(step_index)
        self.iter_count_store.append(stats["iter_count"])
        self.t_wall_total_store.append(stats["t_wall_total"])
        self.t_proc_total_store.append(stats["t_proc_total"])

    def store_solution(
        self,
        step_index,
        uc,
        us,
        x,
        yex,
        ysp,
        forecast,
        x0,
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
        self.x0_store.append(np.atleast_2d(x0))
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
        # TODO: remove, this method probably doesn't work anymore
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

        # Define variables
        uct_var_sym = ca.MX.sym("uct_var", self.mct, self.horizon)
        uct_var = opti.variable(uct_var_sym)
        opti.set_domain(uct_var, "real")

        usp_var_sym = ca.MX.sym("usp_var", self.msp, self.horizon)
        usp_var = opti.variable(usp_var_sym)
        opti.set_domain(usp_var, "real")

        gridcurtail_sym = ca.MX.sym("curtail_var", self.oex, self.horizon)
        gridcurtail = opti.variable(gridcurtail_sym)
        opti.set_domain(gridcurtail, "real")

        x_var_sym = ca.MX.sym("x_var", self.n, self.horizon+1)
        x_var = opti.variable(x_var_sym)
        opti.set_domain(x_var, "real")

        yex_var_sym = ca.MX.sym("yex_var", self.pex, self.horizon)
        yex_var = opti.variable(yex_var_sym)
        opti.set_domain(yex_var, "real")

        if self.only_bounded_yco:
            yco_var_sym = ca.MX.sym("yco_var", len(self.yco_ub_ind), self.horizon)
            # yco_var = opti.variable(len(self.yco_ub_ind), self.horizon)
        else:
            yco_var_sym = ca.MX.sym("yco_var", self.pco, self.horizon)
            # yco_var = opti.variable(self.pco, self.horizon)

        yco_var = opti.variable(yco_var_sym)
        opti.set_domain(yco_var, "real")

        # Define parameters
        dex_param = opti.parameter(self.oex, self.horizon)
        x0_param = opti.parameter(self.n, 1)

        # Apply bounds to variables
        for k in range(self.horizon + 1):
            if k == 0:
                continue

            # x_lb_expr = x_var[:, k] >= self.bounds["x_lb"][:, None]
            # x_lb_expr.name = f"x LB h{k}"
            # opti.subject_to(x_lb_expr)

            opti.subject_to( x_var[:, k] >= self.bounds["x_lb"][:, None])
            opti.subject_to(x_var[:, k] <= self.bounds["x_ub"][:, None])

        for k in range(self.horizon):
            opti.subject_to(uct_var[:, k] >= self.bounds["u_lb"][:, None])
            opti.subject_to(uct_var[:, k] <= self.bounds["u_ub"][:, None])

            opti.subject_to(yex_var[:, k] >= 0)
            opti.subject_to(yex_var[:, k] <= self.reference)
            opti.subject_to(usp_var[:, k] >= np.zeros(self.msp))

            if self.only_bounded_yco:

                assert self.yco_ub_node_ind.shape[0] == 1
                for node_idx in self.yco_ub_node_ind:
                    opti.subject_to(
                        np.ones((1, len(self.yco_ub_ind))) @ yco_var[:, k]
                        <= self.bounds_verbose[self.node_order[node_idx]]["y_ub"]
                    )
                    opti.subject_to(
                        np.ones((1, len(self.yco_ub_ind))) @ yco_var[:, k]
                        >= self.bounds_verbose[self.node_order[node_idx]]["y_lb"]
                    )
            else:

                for node in self.node_order:
                    node_idx = [
                        i
                        for i in range(self.pco)
                        if self.pco_label[i].split(" ")[2] == node
                    ]
                    if len(node_idx) > 0:
                        opti.subject_to(
                            np.ones((1, len(node_idx))) @ yco_var[node_idx, k]
                            <= self.bounds_verbose[node]["y_ub"]
                        )
                        opti.subject_to(
                            np.ones((1, len(node_idx))) @ yco_var[node_idx, k]
                            >= self.bounds_verbose[node]["y_lb"]
                        )

        opti.subject_to(gridcurtail >= -dex_param)
        if self.no_shortfall:
            opti.subject_to(gridcurtail <= 0)
        else:
            opti.subject_to(gridcurtail <= 2e6)

        # Initial conditions
        opti.subject_to(x_var[:, 0] == x0_param)

        if not self.use_objective_class:
            objective = 0
            objective_terms = []

        objective_var_inds = self.get_objective_var_inds()

        # Loop through time steps in the horizon, apply dynamics constraint and calculate objective at each step
        for k in range(self.horizon):

            grid_curtail = gridcurtail[:, k]

            xkp1, yexk, yco, yze, ygt, yet = self.control_model.step_control_model(
                x_var[:, k], uct_var[:, k], usp_var[:, k], dex_param[:, k], grid_curtail
            )
            # xkp1, yexk, yco, yze, ygt, yet = self.step_control_model(
            #     x_var[:, k], uct_var[:, k], usp_var[:, k], dex_param[:, k], grid_curtail
            # )

            opti.subject_to(x_var[:, k + 1] == xkp1)
            opti.subject_to(yex_var[:, k] == yexk[0])
            if self.only_bounded_yco:
                opti.subject_to(yco_var[:, k] == yco[self.yco_ub_ind])
            else:
                opti.subject_to(yco_var[:, k] == yco)
            if self.pze > 0:
                opti.subject_to(yze == np.zeros((self.pze, 1)))
            if self.pgt > 0:
                opti.subject_to(ygt == np.zeros((self.pgt, 1)))
            if self.pet > 0:
                opti.subject_to(yet == np.zeros((self.pet, 1)))


            if not self.use_objective_class:
                step_obj, step_obj_terms = self.objective_step(
                    x_var[:, k],
                    uct_var[:, k],
                    usp_var[:, k],
                    yco_var[:,k],
                    yex_var[:, k],
                    gridcurtail=gridcurtail[:, k],
                    var_inds=objective_var_inds,
                )
                # step_obj, step_obj_terms = self.objective_step(
                #     x_var[:, k],
                #     uct_var[:, k],
                #     usp_var[:, k],
                #     yco,
                #     yexk,
                #     gridcurtail=gridcurtail[:, k],
                #     var_inds=objective_var_inds,
                # )

                objective += step_obj
                objective_terms.append(step_obj_terms)


        if self.use_objective_class:
            objective = self.objective_manager.construct_objective(uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail)
            self.obj_terms = self.objective_manager.obj_terms_w
            self.obj_terms_uw = self.objective_manager.obj_terms_uw
        else:


            terminal_obj, terminal_terms = self.terminal_objective(xkp1)

            if self.terminal_cost:
                objective += terminal_obj

            self.obj_terms = {}
            self.obj_terms_uw = {}

            for term in objective_terms[0].keys():
                obj_term = 0
                obj_term_uw = 0
                for i in range(self.horizon):
                    weight_i = objective_terms[i][term]["w"]
                    expr_i = objective_terms[i][term]["expr"]

                    obj_term += weight_i * expr_i
                    obj_term_uw += expr_i

                self.obj_terms.update({term: obj_term})
                self.obj_terms_uw.update({term: obj_term_uw})

            for term in terminal_terms.keys():
                expr = terminal_terms[term]["expr"]
                w = terminal_terms[term]["w"]

                self.obj_terms.update({term: w * expr})
                self.obj_terms_uw.update({term: expr})


        # Set objective to objective expression
        opti.minimize(objective)
        self.opti = opti
        self.opt_vars = {
            "uct": uct_var,
            "usp": usp_var,
            "x": x_var,
            "yex": yex_var,
            "yco": yco_var,
        }
        self.opt_params = {"dex": dex_param, "x0": x0_param}
        self.opt_vars.update({"gridcurtail": gridcurtail})

        # self.compare_objective_implementations(
        #     obj1=objective,
        #     obj_terms1=self.obj_terms,
        #     obj_terms_uw1=self.obj_terms_uw,
        #     obj2=objective_man,
        #     obj_terms2=self.obj_terms_man,
        #     obj_terms_uw2=self.obj_terms_uw_man,
        # )

    def compare_objective_implementations(self, obj1, obj_terms1, obj_terms_uw1, obj2, obj_terms2, obj_terms_uw2 ):

        ov = {key: np.random.rand(val.shape[0], val.shape[1]) for key,val in self.opt_vars.items()}

        F_obj1 = ca.Function("obj1", list(self.opt_vars.values()), [obj1])
        F_obj2 = ca.Function("obj2", list(self.opt_vars.values()), [obj2])

        print(f"Objectives are probably the same: {np.float64(F_obj1(*list(ov.values()))) == np.float64(F_obj2(*list(ov.values())))}")

        obj_diff = np.float64(F_obj1(*list(ov.values()))) - np.float64(F_obj2(*list(ov.values())))


        for term in obj_terms1.keys():
            F1 = ca.Function(f"f1", list(self.opt_vars.values()), [obj_terms1[term]])
            F2 = ca.Function(f"f2", list(self.opt_vars.values()), [obj_terms2[term]])

            F1_uw = ca.Function(f"f1_uw", list(self.opt_vars.values()), [obj_terms_uw1[term]])
            F2_uw = ca.Function(f"f2_uw", list(self.opt_vars.values()), [obj_terms_uw1[term]])


            f_equal = np.float64(F1(*list(ov.values()))) == np.float64(F2(*list(ov.values())))
            f_diff = np.float64(F1(*list(ov.values()))) - np.float64(F2(*list(ov.values())))

            f_uw_equal = np.float64(F1_uw(*list(ov.values()))) == np.float64(F2_uw(*list(ov.values())))
            f_uw_diff = np.float64(F1_uw(*list(ov.values()))) - np.float64(F2_uw(*list(ov.values())))


            print(f"Term {term} are equal UW: {f_uw_equal}, W: {f_equal}")

            []



        pass

    def get_objective_var_inds(self):
        # Set up indices for objective terms flexibly
        def getid(label, index_list):
            indices = [i for i in range(len(index_list)) if label in index_list[i]]
            assert len(indices) == 1
            return indices[0]

        objective_var_inds = {}

        if "battery" in self.node_order:
            bes_var_inds = dict(
                uct_charge_bes=getid("uct 0 battery", self.mct_label),
                uct_discharge_bes=getid("uct 1 battery", self.mct_label),
                x_bes=getid("x 0 battery", self.n_label),
            )
            objective_var_inds.update(bes_var_inds)

        if "hydrogen_storage" in self.node_order:
            h2s_var_inds = dict(
                uct_charge_h2s=getid("uct 0 hydrogen_storage", self.mct_label),
                uct_discharge_h2s=getid("uct 1 hydrogen_storage", self.mct_label),
                x_h2s=getid("x 0 hydrogen_storage", self.n_label),
            )
            objective_var_inds.update(h2s_var_inds)

        if "thermal_energy_storage" in self.node_order:
            tes_var_inds = dict(
                uct_charge_tes=getid("uct 0 thermal_energy_storage", self.mct_label),
                uct_discharge_tes=getid("uct 1 thermal_energy_storage", self.mct_label),
                x_tes=getid("x 0 thermal_energy_storage", self.n_label),
            )
            objective_var_inds.update(tes_var_inds)
        return objective_var_inds

    def objective_step(
        self,
        x,
        uct,
        usp,
        yco,
        yex,
        gridcurtail=None,
        var_inds=None,
    ):

        # =============================================================================
        # ==                                                                         ==
        # ==                                Objective                                ==
        # ==                                                                         ==
        # =============================================================================
        obj_terms = {}

        term_keys = ["output_tracking", "gridcurtail"]

        obj_terms.update(
            {"output_tracking": {"w": 1e9, "expr": (self.reference - yex) ** 2}}
        )

        obj_terms.update({"gridcurtail": {"w": 1e-4, "expr": gridcurtail**2}})

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

    def terminal_objective(self, x_h):

        obj_terms = {}

        term_keys = []

        if "battery" in self.node_order:
            if "bes_terminal" in self.weights.keys():
                w_bes = self.weights["bes_terminal"]
            else:
                w_bes = 1
            ex_bes = (self.ref_bes_state - x_h[0]) ** 2
            obj_terms.update({"bes_terminal": {"w": w_bes, "expr": ex_bes}})
            term_keys.append("bes_terminal")

        if "hydrogen_storage" in self.node_order:
            if "h2s_terminal" in self.weights.keys():
                w_h2s = self.weights["h2s_terminal"]
            else:
                w_h2s = 1
            ex_h2s = (self.ref_h2s_state - x_h[2]) ** 2
            obj_terms.update({"h2s_terminal": {"w": w_h2s, "expr": ex_h2s}})
            term_keys.append("h2s_terminal")

        if "thermal_energy_storage" in self.node_order:
            if "tes_terminal" in self.weights.keys():
                w_tes = self.weights["tes_terminal"]
            else:
                w_tes = 1
            ex_tes = (self.ref_tes_state - x_h[1]) ** 2
            obj_terms.update({"tes_terminal": {"w": w_tes, "expr": ex_tes}})
            term_keys.append("tes_terminal")

        objective = 0

        # TODO make the terminal terms be relient on the term keys too

        for term in term_keys:
            objective += obj_terms[term]["w"] * obj_terms[term]["expr"]

        return objective, obj_terms

    def update_optimization_parameters(self, x0, src_forecast):
        self.opti.set_value(self.opt_params["dex"], src_forecast)
        self.opti.set_value(self.opt_params["x0"], x0)

    def compute_trajectory(self, x0, forecast, step_index=0, ret_obj=False):
        # =============================================================================
        # ==                                                                         ==
        # ==                            Compute Trajectory                           ==
        # ==                                                                         ==
        # =============================================================================

        def get_sol_value(prob: ca.Opti, var):
            val = prob.value(var)
            val = np.reshape(val, var.shape)
            return val

        if self.use_saved_solution:

            uct, usp, curtail, grid = self.step_saved_solution(
                step_index=step_index, forecast=forecast
            )
            return uct, usp, curtail, grid

        else:

            if len(self.x_store) > 0:
                # Error between where the MPC planned for the state to be and where the measure state is
                state_error = (
                    x0 - self.x_store[-1][:, step_index - self.step_index_store[-1]]
                )

            self.update_optimization_parameters(x0, forecast)
            if self.warm_start_with_previous_solution:
                if hasattr(self, "x_init") and self.prev_success:
                    # Then the optimization has been run at least once and there should
                    # be initial values from the previous solution to borrow

                    overlap = self.horizon - (step_index - self.step_index_store[-1])
                    self.opti.set_initial(
                        self.opt_vars["uct"][:, :overlap], self.uc_init[:, -overlap:]
                    )
                    self.opti.set_initial(
                        self.opt_vars["usp"][:, :overlap], self.us_init[:, -overlap:]
                    )
                    self.opti.set_initial(
                        self.opt_vars["x"][:, :overlap], self.x_init[:, -overlap:]
                    )
                    self.opti.set_initial(
                        self.opt_vars["yex"][:, :overlap], self.ys_init[:, -overlap:]
                    )

            try:
                sol = self.opti.solve()
                sol_stats = sol.stats()
                self.store_solve_stats(sol_stats, step_index)
                successful_optimization = True
                []
            except:
                # If the optimization does not solve, dig into the issues
                self.unpack_bad_solution(step_index=step_index, forecast=forecast, x0=x0)
                sol = self.opti.debug
                successful_optimization = False

            self.prev_sol = sol

            self.prev_success = successful_optimization

            # self.check_gradients(sol)

            uct = get_sol_value(sol, self.opt_vars["uct"])
            usp = get_sol_value(sol, self.opt_vars["usp"])
            x = get_sol_value(sol, self.opt_vars["x"])
            yex = get_sol_value(sol, self.opt_vars["yex"])
            yco = get_sol_value(sol, self.opt_vars["yco"])
            dex = get_sol_value(sol, self.opt_params["dex"])  # [None, :]

            gridcurtail = get_sol_value(sol, self.opt_vars["gridcurtail"])
            grid = np.where(gridcurtail >= 0, gridcurtail, 0)
            curtail = np.where(gridcurtail <= 0, -gridcurtail, 0)

            self.curtail_storage[step_index : step_index + self.horizon] = curtail

            # Save solution values for next warm start
            self.uc_init = uct
            self.us_init = usp
            self.x_init = x
            self.ys_init = yex
            self.curtail_init = curtail

            ysp = (
                self.control_model.Cco @ x[:, :-1]
                + self.control_model.Dcoct @ uct
                + self.control_model.Dcosp @ usp
                + self.control_model.Fcoex @ (dex - curtail)
            )
            dco = self.control_model.M_dco_yco @ ysp

            ysp = np.concatenate([ysp, yex])

            obj_values = {
                key: sol.value(self.obj_terms[key]) for key in self.obj_terms.keys()
            }
            obj_values_uw = {
                key: sol.value(self.obj_terms_uw[key])
                for key in self.obj_terms_uw.keys()
            }

            # self.check_gradients(sol, print_jacs=True)

            self.store_solution(
                step_index=step_index,
                uc=uct,
                us=usp,
                x=x,
                yex=yex,
                ysp=ysp,
                forecast=forecast,
                x0=x0,
                curtail=curtail,
                grid_purchase=grid,
                dex=dex,
                dco=dco,
                objective=obj_values,
                objective_uw=obj_values_uw,
            )

            if not self.debug_mode:
                # self.save_state_for_debug(x0, forecast, step_index)
                pass

            if ret_obj:
                return uct, usp, curtail, grid, obj_values_uw
            else:
                return uct, usp, curtail, grid

    def check_gradients(self, sol, print_jacs=False):

        def cast_numpy(arr):
            if isinstance(arr, scipy.sparse.spmatrix):
                arr = arr.toarray()
            return arr

        def jac_f(sol, var):
            jac_var = cast_numpy(sol.value(ca.jacobian(self.opti.f, var))).reshape(var.T.shape).T
            return jac_var

        def jac_g(sol, var):
            jac_var = cast_numpy(sol.value(ca.jacobian(self.opti.g, var))).reshape((self.opti.g.shape[0], var.shape[1], var.shape[0]))
            return jac_var

        def jac_obj(sol, var, f):
            jac_var = cast_numpy(sol.value(ca.jacobian(f, var))).reshape(var.T.shape).T
            return jac_var

        def print_with_labels(jac, labels):

            print("")
            np.set_printoptions(linewidth=200, suppress=True, precision=4)
            max_label = np.max([len(lab) for lab in labels])

            jac_rows = jac.__str__().split("\n")

            for i, lab, in enumerate(labels):
                jac_rows[i] = labels[i].ljust(max_label+2, " ") + jac_rows[i]

            print("\n".join(jac_rows))

        def print_constraints(sol, var):
            jac_g_var = jac_g(sol, var)

            for i in range(sol.opti.g.shape[0]):
                out_str = "" 
                out_str += str(sol.value(sol.opti.lbg[i])).ljust(20, " ")

                expr = str(sol.opti.g[i])

                out_str += "<=     " +  expr + "     <="

                out_str += str(sol.value(sol.opti.ubg[i])).rjust(20, " ")

                print(out_str)

            pass

        sol_curtail = sol.value(self.opt_vars["gridcurtail"])
        sol_uct = sol.value(self.opt_vars["uct"])
        sol_usp = sol.value(self.opt_vars["usp"])
        sol_x = sol.value(self.opt_vars["x"])
        sol_yex = sol.value(self.opt_vars["yex"])

        jac_curtail = jac_f(sol, self.opt_vars["gridcurtail"])            
        jac_uct = jac_f(sol, self.opt_vars["uct"])
        jac_usp = jac_f(sol, self.opt_vars["usp"])
        jac_x = jac_f(sol, self.opt_vars["x"])
        jac_yex = jac_f(sol, self.opt_vars["yex"])

        jac_g_uct = jac_g(sol, self.opt_vars["uct"])
        jac_g_usp = jac_g(sol, self.opt_vars["usp"])
        jac_g_x = jac_g(sol, self.opt_vars["x"])
        jac_g_yex = jac_g(sol, self.opt_vars["yex"])

        if print_jacs:

            print_with_labels(sol_curtail, ["curtail"])
            print_with_labels(sol_uct, self.mct_label)
            print_with_labels(sol_usp, self.msp_label)
            print_with_labels(sol_x, self.n_label)
            print_with_labels(sol_yex, self.pex_label)

            print_with_labels(jac_uct, self.mct_label)
            print_with_labels(jac_usp, self.msp_label)
            print_with_labels(jac_x, self.n_label)
            print_with_labels(jac_yex, self.pex_label)

            for obj_key in self.obj_terms_uw.keys():
                print("")
                print("===============================================================")
                print(obj_key)
                print("===============================================================")

                print_with_labels(jac_obj(sol, self.opt_vars["gridcurtail"], self.obj_terms_uw[obj_key]), ["curtail"])
                print_with_labels(jac_obj(sol, self.opt_vars["uct"], self.obj_terms_uw[obj_key]), self.mct_label)
                print_with_labels(jac_obj(sol, self.opt_vars["usp"], self.obj_terms_uw[obj_key]), self.msp_label)
                print_with_labels(jac_obj(sol, self.opt_vars["x"], self.obj_terms_uw[obj_key]), self.n_label)
                print_with_labels(jac_obj(sol, self.opt_vars["yex"], self.obj_terms_uw[obj_key]), self.pex_label)

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

    def unpack_bad_solution(self, step_index, forecast, x0):
        with Capturing() as output:
            self.opti.debug.show_infeasibilities()

        violations = []

        i = 0
        while i < len(output):
            if output[i].startswith("------- i = "):
                # new constraint description
                num_desc = output[i + 1]
                line_number = output[i + 2]
                code_desc = output[i + 3]
                at_desc = ""
                # at_description = output[i + 4]

                violation = float(num_desc.split("viol ")[1].split(")")[0])
                violations.append(violation)

                if violation >= 1e-3:
                    if "opti.subject" in code_desc:
                        code_desc = code_desc.split("opti.subject_to(")[1][:-1]
                    else:
                        code_desc = ""

                    print_line = (
                        str(num_desc).ljust(45) + code_desc.ljust(130) + at_desc
                    )
                    if self.verbose:
                        pprint.pprint(print_line, width=200)
                i += 4
            i += 1

        self.plot_trajectory_generic(self.opti.debug, forecast)

        np.set_printoptions(linewidth=200, suppress=True, precision=4)

        # if not self.debug_mode:
        #     self.save_state_for_debug(x0, forecast, step_index)

        # if not (np.max(np.abs(violations)) <= 1e-3):
        #     self.check_gradients(self.opti.debug)
        #     []

        # assert np.max(np.abs(violations)) <= 1e3, f"violation too large at step index {step_index}"


        if np.max(np.abs(violations)) > 1e3:
            self.check_gradients(self.opti.debug, print_jacs=self.verbose)
            if not self.debug_mode:
                self.save_state_for_debug(x0, forecast, step_index)

                raise AssertionError(f"violation too large at step index {step_index}")



        self.bad_solve_count += 1
        self.bad_solve_step.append(step_index)
        self.bad_solve_violation.append(np.max(np.abs(violations)))

        plt.close()

    def step_saved_solution(self, step_index, forecast):
        # find the right index

        save_index = [
            i
            for i in range(len(self.step_index_saved))
            if step_index == self.step_index_saved[i]
        ]
        assert len(save_index) == 1
        save_index = save_index[0]

        # Values set in load_stored_values

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

    def save_state_for_debug(self, x0, forecast, step_index):

        assert not self.debug_mode

        import datetime
        from pathlib import Path
        import json

        datetime_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S-%s")
        dir_path = "/Users/ztully/Documents/hybrids_code/GH_scripts/greenheart_scripts/minnesota_reference_design/01-minnesota-steel/saved_data/mpc_saved_states"
        dir = f"{dir_path}/mpcstate_{datetime_string}_step{step_index}"
        # Path(dir).mkdir(parents=True, exist_ok=True)
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        # fpath = f"{dir}/mpc_data.json"
        fpath = f"{dir}.json"

        js_kw = dict(ensure_ascii=True)
        # js_kw = dict(ensure_ascii=True, indent=4)

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
                bounds_verbose={
                    node: {
                        key: self.bounds_verbose[node][key].tolist()
                        for key in self.bounds_verbose[node].keys()
                    }
                    for node in self.bounds_verbose.keys()
                },
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
                yco_ub_ind=self.yco_ub_ind.tolist(),
                yco_ub_node_ind=self.yco_ub_node_ind.tolist(),
                cols_li=self.cols_li.tolist(),
                cols_nl=self.cols_nl.tolist(),
                rows_li=self.rows_li.tolist(),
                rows_nl=self.rows_nl.tolist(),
                block_ss=self.block_ss.tolist(),
                x_bes_max=self.G.nodes["battery"]["ionode"].model.max_capacity_kWh,
                x_tes_max = self.G.nodes["thermal_energy_storage"]["ionode"].model.H_capacity_kWh,
                x_h2s_max =self.G.nodes["hydrogen_storage"]["ionode"].model.max_capacity_kg,
                repr = self.opti.__repr__(),
                return_status = self.opti.return_status(),
                casadi_stats = self.opti.stats(),      
                opti_params = str(self.opti.value_parameters()),      
                opti_variables = str(self.opti.value_variables()),  
                s_opts = self.s_opts,
                p_opts = self.p_opts,    
            )

            if hasattr(self, "x_init"):  # then try to update initial guess
                save_dict.update(
                    # dict(
                    #     uct_init=self.uc_init.tolist(),
                    #     usp_init=self.us_init.tolist(),
                    #     x_init=self.x_init.tolist(),
                    #     yex_init=self.ys_init.tolist(),
                    # )
                    dict(
                        step_index_init = self.step_index_store[-1],
                        uc_init = self.uct_store[-1].tolist(),
                        us_init = self.usp_store[-1].tolist(),
                        x_init = self.x_store[-1].tolist(),
                        ys_init = self.yex_store[-1].tolist(),
                        curtail_init = self.curtail_store[-1].tolist()
                    )
                )

                # self.opti.set_initial(self.opt_vars["uct"], self.uc_init)
                # self.opti.set_initial(self.opt_vars["usp"], self.us_init)
                # self.opti.set_initial(self.opt_vars["x"], self.x_init)
                # self.opti.set_initial(self.opt_vars["yex"], self.ys_init)

            save_dict.update({"mpc_config": self.mpc_config})

            json.dump(save_dict, f, **js_kw)

        pass

    def load_state_for_debug(self, state_dict):
        self.horizon = state_dict["horizon"]

        self.bounds = {
            key: np.array(state_dict["bounds"][key], dtype=float)
            for key in state_dict["bounds"].keys()
        }
        self.bounds_verbose = {
            node: {
                key: np.array(state_dict["bounds_verbose"][node][key], dtype=float)
                for key in state_dict["bounds_verbose"][node].keys()
            }
            for node in state_dict["bounds_verbose"].keys()
        }

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
                    combined_mat[i][j] = np.array(combined_mat[i][j], dtype=float)
                else:
                    combined_mat[i][j] = np.zeros((out_dims[i], in_dims[j]), dtype=float)

        # combined_mat = [[np.array(mat) for mat in row] for row in combined_mat]

        self.A, self.Bct, self.Bsp, self.Eex = combined_mat[0]
        self.Cco, self.Dcoct, self.Dcosp, self.Fcoex = combined_mat[1]
        self.Cex, self.Dexct, self.Dexsp, self.Fexex = combined_mat[2]
        self.Cze, self.Dzect, self.Dzesp, self.Fzeex = combined_mat[3]
        self.Cgt, self.Dgtct, self.Dgtsp, self.Fgtex = combined_mat[4]
        self.Cet, self.Detct, self.Detsp, self.Fetex = combined_mat[5]

        self.M_dco_yco = np.array(state_dict["M_dco_yco"], dtype=float)
        self.yco_ub_ind = np.array(state_dict["yco_ub_ind"], dtype=int)
        self.yco_ub_node_ind = np.array(state_dict["yco_ub_node_ind"], dtype=int)

        self.cols_li = np.array(state_dict["cols_li"])
        self.cols_nl = np.array(state_dict["cols_nl"])
        self.rows_li = np.array(state_dict["rows_li"])
        self.rows_nl = np.array(state_dict["rows_nl"])

        self.block_ss = np.array(state_dict["block_ss"], dtype=float)


        if "x_init" in state_dict:

            self.step_index_store.append(state_dict["step_index_init"])

            self.uc_init = np.array(state_dict["uc_init"], dtype=float)
            self.us_init = np.array(state_dict["us_init"], dtype=float)
            self.x_init = np.array(state_dict["x_init"], dtype=float)
            self.ys_init = np.array(state_dict["ys_init"], dtype=float)
            self.curtail_init = np.array(state_dict["curtail_init"], dtype=float)

            self.prev_success = True

        self.x_bes_max = float(state_dict["x_bes_max"])
        self.x_tes_max = float(state_dict["x_tes_max"])
        self.x_h2s_max = float(state_dict["x_h2s_max"])

        []

    def plot_trajectory_generic(self, prob: ca.Opti, forecast):

        def get_sol_value(prob: ca.Opti, var):
            val = prob.value(var)
            val = np.reshape(val, var.shape)
            return val

        x_db = get_sol_value(prob, self.opt_vars["x"])
        uc_db = get_sol_value(prob, self.opt_vars["uct"])
        us_db = get_sol_value(prob, self.opt_vars["usp"])
        ys_db = get_sol_value(prob, self.opt_vars["yex"])
        yco_db = get_sol_value(prob, self.opt_vars["yco"])

        gridcurtail = get_sol_value(prob, self.opt_vars["gridcurtail"])
        grid_db = np.where(gridcurtail >= 0, gridcurtail, 0)
        curtail_db = np.where(gridcurtail <= 0, -gridcurtail, 0)

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

    # def plot_trajectory(self, step_index=None):

    #     idx = -1

    #     fig, ax = plt.subplots(2, 2, sharex="all", layout="constrained")

    #     ax[0, 0].plot(self.forecast_store[idx].T)
    #     ax[0, 0].fill_between(
    #         np.arange(0, self.horizon, 1),
    #         self.forecast_store[idx][0, :],
    #         self.forecast_store[idx][0, :] - self.curtail_store[idx][0, :],
    #     )

    #     gen_split_index = [
    #         i
    #         for i in range(self.msp)
    #         if self.msp_label[i].split(" ")[2] == "generation"
    #     ]
    #     start = np.zeros(self.horizon)
    #     time = np.arange(0, self.horizon, 1)
    #     for k in gen_split_index:
    #         stop = self.usp_store[idx][k, :]
    #         ax[0, 0].fill_between(
    #             time,
    #             start,
    #             stop,
    #             edgecolor=None,
    #             label=self.msp_label[gen_split_index[k]],
    #         )
    #         start += stop

    #     ax[0, 0].legend()

    #     pass


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

            # dex_inds = [
            #     i for i in range(len(self.oex_label)) if node in self.oex_label[i]
            # ]
            dco_inds = [
                i
                for i in range(len(self.oco_label))
                if node in self.oco_label[i].split(" ")[2]
            ]

            uct_inds = [
                i for i in range(len(self.mct_label)) if node in self.mct_label[i]
            ]
            # usp_inds = [
            #     i
            #     for i in range(len(self.msp_label))
            #     if node in self.msp_label[i].split(" ")[2]
            # ]

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
                            ax.scatter(
                                t * np.ones(len(inds)),
                                stored[j][inds, :],
                                color=colors[k],
                            )
                        else:
                            ax.plot(t.T, stored[j][inds[k], :].T, color=colors[k])

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
                    np.concatenate(
                        [
                            self.forecast_store[i],
                            self.forecast_store[i] - self.curtail_store[i],
                            self.forecast_store[i] + self.grid_store[i],
                        ]
                    )
                    # [self.forecast_store[i], np.array(self.forecast_store[i]) - np.array(self.curtail_store[i]), np.array(self.forecast_store[i]) + np.array(self.grid_store[i])]
                    for i in range(len(self.step_index_store))
                ]

                plot_one(ax[i, 0], forecast_curtail_grid, np.array([0, 1, 2]))

            fig.align_ylabels()

        pass


# class Objective:
#     def __init__(self, horizon:int, active_terms:list[str], weights:list[float], references:dict, capacities:dict, var_inds:dict):

#         self.horizon = horizon

#         self.active_terms = active_terms
        
#         self.weights = weights

#         self.steel_ref = references["steel"]
        
#         self.x_bes_ref = references["x_bes"]
#         self.x_tes_ref = references["x_tes"]
#         self.x_h2s_ref = references["x_h2s"]

#         self.soc_bes_ref = references["soc_bes"]
#         self.soc_tes_ref = references["soc_tes"]
#         self.soc_h2s_ref = references["soc_h2s"]

#         self.x_bes_max = capacities["x_bes_max"]
#         self.x_tes_max = capacities["x_tes_max"]
#         self.x_h2s_max = capacities["x_h2s_max"]
#         self.x_bes_min = capacities["x_bes_min"]
#         self.x_tes_min = capacities["x_tes_min"]
#         self.x_h2s_min = capacities["x_h2s_min"]

#         self.var_inds = var_inds

#         self.all_terms = [
#             "output_tracking",
#             "gridcurtail",
#             "bes_simultaneous",
#             "tes_simultaneous",
#             "h2s_simultaneous",
#             "bes_state", 
#             "tes_state",
#             "h2s_state",
#             "bes_soc_state",
#             "tes_soc_state",
#             "h2s_soc_state",
#             "bes_terminal",
#             "tes_terminal", 
#             "h2s_terminal", 
#             # "storage_state_LQ>'
#         ]

#         self.inactive_terms = [t for t in self.all_terms if t not in self.active_terms]

#         assert all([(t in self.weights) for t in self.active_terms]), f"These objective terms were activated but not given any weights: {[t for t in self.active_terms if (t not in self.weights)]}"

#         for term in self.all_terms:
#             if term not in self.weights:
#                 self.weights[term] = 1

#         self.step_term_list = []
#         self.non_step_term_list = []


#         self.term_map = dict(
#             output_tracking = self.term_step_output_tracking,
#             gridcurtail = self.term_step_grid_curtail,
#             bes_simultaneous=self.term_step_bes_simultaneous,
#             tes_simultaneous=self.term_step_tes_simultaneous,
#             h2s_simultaneous=self.term_step_h2s_simultaneous,
#             bes_state=self.term_step_bes_state,
#             tes_state=self.term_step_tes_state,
#             h2s_state=self.term_step_h2s_state,
#             bes_terminal = self.term_bes_terminal,
#             tes_terminal = self.term_tes_terminal,
#             h2s_terminal = self.term_h2s_terminal,
#             bes_soc_state = self.term_step_bes_state_soc_quadratic,
#             tes_soc_state = self.term_step_tes_state_soc_quadratic,
#             h2s_soc_state = self.term_step_h2s_state_soc_quadratic,
#         )



#     def construct_objective(self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail):
#         kwargs = dict(uct_var = uct_var,
#                       usp_var= usp_var, 
#                       x_var = x_var, 
#                       yex_var = yex_var, 
#                       yco_var = yco_var,
#                       gridcurtail = gridcurtail)

#         obj_uw = 0
#         obj_w = 0

#         obj_terms_uw = {}
#         obj_terms_w = {}


#         # for term in self.active_terms:
#         for term in self.all_terms:
#             obj_term = self.term_map[term](**kwargs)

#             obj_terms_uw.update({term:obj_term})
#             obj_terms_w.update({term:self.weights[term] * obj_term})
        

#             if term in self.active_terms:
#                 obj_w += self.weights[term] * obj_term

#         obj_terms_uw.update({"objective":obj_w})
#         obj_terms_w.update({"objective":obj_w})


#         self.obj_terms_w = obj_terms_w
#         self.obj_terms_uw = obj_terms_uw
   



#         return obj_w


#     def term_step_output_tracking(self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail):
#         obj = 0

#         for k in range(self.horizon):
#             obj_k = (self.steel_ref - yex_var[:, k])**2
#             obj += obj_k

#         return obj



#     def term_step_grid_curtail(self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail):
#         obj = 0

#         for k in range(self.horizon):
#             obj_k = (gridcurtail[:, k])**2
#             obj+=obj_k

#         return obj

#     def term_step_bes_simultaneous(self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail):
#         obj = 0

#         for k in range(self.horizon):
#             obj_k = uct_var[self.var_inds["uct_charge_bes"], k] * uct_var[self.var_inds["uct_discharge_bes"], k]
#             obj += obj_k

#         return obj

#     def term_step_tes_simultaneous(self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail):
#         obj = 0
#         for k in range(self.horizon):
#             obj_k = uct_var[self.var_inds["uct_charge_tes"], k] * uct_var[self.var_inds["uct_discharge_tes"], k]
#             obj += obj_k

#         return obj

#     def term_step_h2s_simultaneous(self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail):
#         obj = 0

#         for k in range(self.horizon):
#             obj_k = uct_var[self.var_inds["uct_charge_h2s"], k] * uct_var[self.var_inds["uct_discharge_h2s"], k]
#             obj+= obj_k

#         return obj

#     def term_step_bes_state(self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail):
#         obj = 0

#         for k in range(self.horizon):
#             obj_k = (x_var[self.var_inds["x_bes"], k] - self.x_bes_ref)**2
#             obj += obj_k

#         return obj
    
#     def term_step_tes_state(self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail):
#         obj = 0

#         for k in range(self.horizon):
#             obj_k = (x_var[self.var_inds["x_tes"], k] - self.x_tes_ref)**2
#             obj += obj_k

#         return obj
    
#     def term_step_h2s_state(self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail):
#         obj = 0

#         for k in range(self.horizon):
#             obj_k = (x_var[self.var_inds["x_h2s"], k] - self.x_h2s_ref)**2
#             obj += obj_k

#         return obj

#     def term_bes_terminal(self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail):

#         # Absolute state reference terminal
#         # obj = (x_var[self.var_inds["x_bes"], self.horizon] - self.x_bes_ref)**2
        
#         # Relative or SOC reference
#         obj = (x_var[self.var_inds["x_bes"], self.horizon]/self.x_bes_max - self.soc_bes_ref)**2

        
#         return obj
    
#     def term_tes_terminal(self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail):

#         # Absolute reference
#         # obj = (x_var[self.var_inds["x_tes"], self.horizon] - self.x_tes_ref)**2

#         # Relative reference
#         obj = (x_var[self.var_inds["x_tes"], self.horizon]/self.x_tes_max - self.soc_tes_ref)**2
        
#         return obj
    
#     def term_h2s_terminal(self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail):

#         # Absolute reference
#         # obj = (x_var[self.var_inds["x_h2s"], self.horizon] - self.x_h2s_ref)**2

#         # Relative SOC reference
#         obj = (x_var[self.var_inds["x_h2s"], self.horizon]/self.x_h2s_max - self.soc_h2s_ref)**2
        
#         return obj

#     def term_step_storage_state_linear_quadratic(self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail):
#         x_bar = np.array([[self.x_bes_max, self.x_tes_max, self.x_h2s_max]])
        
#         Q_quad = 3 * np.eye(3) - np.ones((3, 3))
#         Q_lin = -np.ones((1, 3))

#         x_soc = x_var / x_bar

#         term_value = x_soc.T @ Q_quad @ x_soc + Q_lin @ x_soc

#         return term_value


#     def term_step_bes_state_linear(self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail):
#         pass


    
#     def term_step_bes_state_soc_quadratic(self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail):
#         w_bes = 1 / (3 * (self.x_bes_max - self.x_bes_min)**2)
#         term_obj = 0
#         for k in range(self.horizon):
#             term_obj += w_bes * (self.x_bes_max - x_var[self.var_inds["x_bes"], k])**2
#         return term_obj
    
#     def term_step_tes_state_soc_quadratic(self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail):
#         w_tes = 1 / (3 * (self.x_tes_max - self.x_tes_min)**2)
#         term_obj = 0
#         for k in range(self.horizon):
#             term_obj += w_tes * (self.x_tes_max - x_var[self.var_inds["x_tes"], k])**2
#         return term_obj

#     def term_step_h2s_state_soc_quadratic(self, uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail):
#         w_h2s = 1 / (3 * (self.x_h2s_max - self.x_h2s_min)**2)
#         term_obj = 0
#         for k in range(self.horizon):
#             term_obj += w_h2s * (self.x_h2s_max - x_var[self.var_inds["x_h2s"], k])**2
#         return term_obj


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
    from hopp.simulation.technologies.sites.site_info import SiteInfo
    from greenheart.simulation.realtime_simulation import RealTimeSimulation

    class HOPPSystem:
        def __init__(self, site):
            self.site = site

    class HOPPInterface:
        def __init__(self, site):
            self.system = HOPPSystem(site)

    # config_root = Path(__file__).parents[0] / "dispatch_inputs"
    config_root = Path(__file__).parents[5]/ "tests" / "greenheart" / "test_dispatch" / "dispatch_inputs"

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

    []