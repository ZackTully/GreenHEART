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
import traceback
import contextlib
import io

import time

from hopp.utilities import load_yaml

from greenheart.simulation.technologies.dispatch.controllers.controller_tools.control_model_builder import ControlModelBuilder
from greenheart.simulation.technologies.dispatch.controllers.controller_tools.gradient_helper import GradientHelper
from greenheart.simulation.technologies.dispatch.controllers.controller_tools.plotter_helper import MPCPlotter
from greenheart.simulation.technologies.dispatch.controllers.controller_tools.objective_helper import Objective
from greenheart.simulation.technologies.dispatch.controllers.controller_tools.debug_helper import DebugHelper


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
        self.G = simulation_graph

        self.node_order = node_order
        self.edge_order = edge_order

        self.mpc_config = mpc_config

        self.debug_mode = debug_mode
        self.verbose = False

        if "logging" in self.mpc_config:
            self.logging_config = self.mpc_config.pop("logging")
            self.setup_logging(self.logging_config)

        self.plotter = MPCPlotter(mpc=self)
        self.debug_helper = DebugHelper(mpc=self)
        self.control_model = ControlModelBuilder(mpc=self)
        self.gradient_helper = GradientHelper(mpc=self)

        self.use_objective_class = True
        self.horizon = mpc_config["horizon"]

        self.reference = mpc_config["reference"]
        self.weights = mpc_config["weights"]
        self.term_keys = mpc_config["terms"]

        self.set_option_flags(mpc_config)

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
            self.debug_helper.load_state_for_debug(saved_state)
        else:
            self.config = config
            system_graph = load_yaml(
                config.greenheart_config["realtime_simulation"]["system"][
                    "system_graph_config"
                ]
            )
            self.traversal_order = system_graph["traversal_order"]

            self.control_model.build_control_model(self.traversal_order, self.G)

            if "battery" in self.node_order:
                self.get_battery_graph_info()

            if "hydrogen_storage" in self.node_order:
                self.get_hydrogen_storage_graph_info()

            if "thermal_energy_storage" in self.node_order:
                self.get_thermal_energy_storage_graph_info()

        self.objective_manager = Objective(
            mpc=self,
            horizon=self.horizon,
            active_terms=self.term_keys,
            weights=self.weights,
            references=dict(
                steel=self.reference,
                x_bes=self.ref_bes_state,
                x_tes=self.ref_tes_state,
                x_h2s=self.ref_h2s_state,
                soc_bes=self.mpc_config["references"]["bes"],
                soc_tes=self.mpc_config["references"]["tes"],
                soc_h2s=self.mpc_config["references"]["h2s"],
            ),
            capacities=dict(
                x_bes_max=self.x_bes_max,
                x_bes_min=self.x_bes_min,
                x_tes_max=self.x_tes_max,
                x_tes_min=self.x_tes_min,
                x_h2s_max=self.x_h2s_max,
                x_h2s_min=self.x_h2s_min,
            ),
            # var_inds=self.objective_manager.get_objective_var_inds()
        )

        self.setup_optimization()
        self.gradient_helper.get_mpc_attrs()

        if self.horizon == 1:
            self.warm_start_with_previous_solution = False

        self.prev_sol = None

    def set_option_flags(self, mpc_config):
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

    def get_battery_graph_info(self):
        self.x_bes_max = self.G.nodes["battery"]["ionode"].model.max_capacity_kWh
        self.x_bes_min = self.G.nodes["battery"]["ionode"].model.min_capacity_kWh

        bes_soc_ref = self.mpc_config["references"]["bes"]
        self.ref_bes_state = bes_soc_ref * self.x_bes_max

        self.weight_bes_state = 1e-4 / self.ref_bes_state

    def get_hydrogen_storage_graph_info(self):
        graph_h2s = self.G.nodes["hydrogen_storage"]["ionode"].model

        self.x_h2s_max = graph_h2s.max_capacity_kg
        self.x_h2s_min = graph_h2s.min_capacity_kg

        h2s_soc_ref = self.mpc_config["references"]["h2s"]
        self.ref_h2s_state = h2s_soc_ref * self.x_h2s_max
        self.weight_h2s_state = 1e-1 / self.ref_h2s_state

    def get_thermal_energy_storage_graph_info(self):
        graph_tes = self.G.nodes["thermal_energy_storage"]["ionode"].model

        self.x_tes_max = graph_tes.H_capacity_kWh
        self.x_tes_min = 0
        # self.x_tes_min = simulation_graph.nodes["thermal_energy_storage"]["ionode"].model.H_buffer_max_kWh

        tes_soc_ref = self.mpc_config["references"]["tes"]
        self.ref_tes_state = tes_soc_ref * self.x_tes_max
        self.weight_tes_state = 1e-4 / self.ref_tes_state

    # def set_no_shortfall_bool(self, no_shortfall: bool):
    #     self.no_shortfall = no_shortfall
    #     self.setup_optimization()

    # def set_use_NL_electrolyzer(self, use_NL: bool):
    #     self.use_NL_electrolzyer = use_NL
    #     self.setup_optimization()

    # def set_terminal_cost_bool(self, terminal_bool):
    #     self.terminal_cost = terminal_bool
    #     self.setup_optimization()

    def setup_logging(self, log_config):
        if log_config["queue"] is None:
            self.logger = log_config["logger"]

        else:
            self.logger = logging.getLogger(f"MPC {log_config['case_description']}")
            self.logger.setLevel(logging.DEBUG)

            queue_handler = handlers.QueueHandler(log_config["queue"])
            queue_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(queue_handler)

        self.logger.info("MPC logger initialized")

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
        self.t_proc_total_store = []

        self.bad_solutions_saved = 0

        self.bad_solve_count = 0
        self.bad_solve_step = []
        self.bad_solve_violation = []

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

        # if not self.use_objective_class:
        #     objective = 0
        #     objective_terms = []

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

            # if not self.use_objective_class:
            #     step_obj, step_obj_terms = self.objective_step(
            #         x_var[:, k],
            #         uct_var[:, k],
            #         usp_var[:, k],
            #         yco_var[:,k],
            #         yex_var[:, k],
            #         gridcurtail=gridcurtail[:, k],
            #         var_inds=self.objective_manager.var_inds,
            #     )
            #     # step_obj, step_obj_terms = self.objective_step(
            #     #     x_var[:, k],
            #     #     uct_var[:, k],
            #     #     usp_var[:, k],
            #     #     yco,
            #     #     yexk,
            #     #     gridcurtail=gridcurtail[:, k],
            #     #     var_inds=objective_var_inds,
            #     # )

            #     objective += step_obj
            #     objective_terms.append(step_obj_terms)

        if self.use_objective_class:
            objective = self.objective_manager.construct_objective(uct_var, usp_var, x_var, yex_var, yco_var, gridcurtail)
            self.obj_terms = self.objective_manager.obj_terms_w
            self.obj_terms_uw = self.objective_manager.obj_terms_uw
        else:
            pass

            # terminal_obj, terminal_terms = self.terminal_objective(xkp1)

            # if self.terminal_cost:
            #     objective += terminal_obj

            # self.obj_terms = {}
            # self.obj_terms_uw = {}

            # for term in objective_terms[0].keys():
            #     obj_term = 0
            #     obj_term_uw = 0
            #     for i in range(self.horizon):
            #         weight_i = objective_terms[i][term]["w"]
            #         expr_i = objective_terms[i][term]["expr"]

            #         obj_term += weight_i * expr_i
            #         obj_term_uw += expr_i

            #     self.obj_terms.update({term: obj_term})
            #     self.obj_terms_uw.update({term: obj_term_uw})

            # for term in terminal_terms.keys():
            #     expr = terminal_terms[term]["expr"]
            #     w = terminal_terms[term]["w"]

            #     self.obj_terms.update({term: w * expr})
            #     self.obj_terms_uw.update({term: expr})

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

                self.opti.set_initial(self.opti.x, np.zeros(self.opti.x.shape))

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


                uct_feas, usp_feas, x_feas, yex_feas, yco_feas, ucur_feas = self.control_model.compute_feasible_initial_values(x0, forecast, self.opti, start_index = overlap)
                self.opti.set_initial(self.opt_vars["uct"], uct_feas)
                self.opti.set_initial(self.opt_vars["usp"], usp_feas)
                self.opti.set_initial(self.opt_vars["x"], x_feas)
                self.opti.set_initial(self.opt_vars["yex"], yex_feas)
                self.opti.set_initial(self.opt_vars["yco"], yco_feas)
                self.opti.set_initial(self.opt_vars["gridcurtail"], ucur_feas)


                # self.gradient_helper.check_initial_values(self.opti)

                # g_init = self.opti.value(self.opti.g, self.opti.initial())
                # lbg_init = self.opti.value(self.opti.lbg, self.opti.initial())
                # ubg_init = self.opti.value(self.opti.ubg, self.opti.initial())




                []

            else:
                uct_feas, usp_feas, x_feas, yex_feas, yco_feas, ucur_feas = self.control_model.compute_feasible_initial_values(x0, forecast)
                self.opti.set_initial(self.opt_vars["uct"], uct_feas)
                self.opti.set_initial(self.opt_vars["usp"], usp_feas)
                self.opti.set_initial(self.opt_vars["x"], x_feas)
                self.opti.set_initial(self.opt_vars["yex"], yex_feas)
                self.opti.set_initial(self.opt_vars["yco"], yco_feas)
                self.opti.set_initial(self.opt_vars["gridcurtail"], ucur_feas)

                # self.gradient_helper.check_initial_values(self.opti)

        try:
            stderr_buffer = io.StringIO()
            # Use this workaround to capture casadi NaN detected errors
            with contextlib.redirect_stderr(stderr_buffer):
                sol = self.opti.solve()

            stderr_msg = stderr_buffer.getvalue()
            if len(stderr_msg) > 0:
                log_msg = self.debug_helper.process_stderr(stderr_msg)
                if log_msg is not None: 
                    self.logger.warning(f"{step_index = }:\nlog_msg")

            sol_stats = sol.stats()
            self.store_solve_stats(sol_stats, step_index)
            successful_optimization = True
            []
        except Exception as e:
            self.logger.debug(e)
            # self.logger.debug(traceback.format_exc())

            # If the optimization does not solve, dig into the issues
            violation_desc = self.unpack_bad_solution(step_index=step_index, forecast=forecast, x0=x0)
            self.logger.debug(violation_desc)


            # Think about adding more debug information to the logger here
            # Gradients

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
            # self.debug_helper.save_state_for_debug(x0, forecast, step_index)
            pass


        active_obj_uw = {k:v for k, v in self.objective_manager.obj_terms_uw_traj.items() if k in self.term_keys}

        if np.any([sol.value(v) > 1.0001 for vals in active_obj_uw.values() for v in vals] ):
            self.gradient_helper.print_objective_trajectory_values(sol, terms=self.term_keys, weighted=False)
            pass
        

        # self.gradient_helper.print_objective_values(sol, terms=self.term_keys)
        if (yex < 0.99 * self.reference).any():
            # self.gradient_helper.print_objective_trajectory_values(sol, terms=self.term_keys, weighted=True)
            # self.gradient_helper.print_objective_trajectory_values(sol, terms=self.term_keys, weighted=False)
            # self.gradient_helper.print_objective_jacobian(sol, terms=self.term_keys)

            # self.gradient_helper.check_gradients(sol)
            # self.plotter.plot_saved_trajectories()
            pass

        # if step_index > 20:
        #     self.plotter.plot_saved_trajectories()

        # self.gradient_helper.check_gradients(sol)

        if ret_obj:
            return uct, usp, curtail, grid, obj_values_uw
        else:
            return uct, usp, curtail, grid


    def unpack_bad_solution(self, step_index, forecast, x0):
        with Capturing() as output:
            self.opti.debug.show_infeasibilities()

        violations = []

        violation_summary = ""

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
                    violation_summary += print_line + "\n"
                    if self.verbose:
                        pprint.pprint(print_line, width=200)
                i += 4
            i += 1


        # np.set_printoptions(linewidth=200, suppress=True, precision=4)


        # if True:
        if np.max(np.abs(violations)) > 1e-3:
            self.gradient_helper.check_gradients(self.opti.debug)



            # if not self.debug_mode:
            #     self.debug_helper.save_state_for_debug(x0, forecast, step_index)

            #     raise AssertionError(f"violation too large at step index {step_index}")

        self.bad_solve_count += 1
        self.bad_solve_step.append(step_index)
        self.bad_solve_violation.append(np.max(np.abs(violations)))

        if self.bad_solutions_saved < 20:
            self.debug_helper.save_state_for_debug(x0, forecast, step_index)
            self.bad_solutions_saved += 1

        return violation_summary 

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


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout


# if __name__ == "__main__":

#     from pathlib import Path
#     from greenheart.simulation.greenheart_simulation import GreenHeartSimulationConfig
#     from hopp.simulation.technologies.sites.site_info import SiteInfo
#     from greenheart.simulation.realtime_simulation import RealTimeSimulation

#     class HOPPSystem:
#         def __init__(self, site):
#             self.site = site

#     class HOPPInterface:
#         def __init__(self, site):
#             self.system = HOPPSystem(site)

#     # config_root = Path(__file__).parents[0] / "dispatch_inputs"
#     config_root = Path(__file__).parents[5]/ "tests" / "greenheart" / "test_dispatch" / "dispatch_inputs"

#     fname_hopp_config = str(config_root / "plant/hopp_config_mn.yaml")
#     fname_greenheart_config = str(config_root / "plant/greenheart_config_onshore_mn.yaml")
#     fname_turbine_config = str(
#         config_root / "turbines/ATB2024_6MW_170RD_floris_turbine.yaml"
#     )
#     fname_floris_config = str(config_root / "floris/floris_input_lbw_6MW.yaml")


#     config = GreenHeartSimulationConfig(
#         fname_hopp_config,
#         fname_greenheart_config,
#         fname_turbine_config,
#         fname_floris_config,
#         verbose=False,
#         show_plots=False,
#         save_plots=False,
#         use_profast=True,
#         post_processing=True,
#         incentive_option=1,
#         plant_design_scenario=1,
#         output_level=8,
#     )

#     config.realtime_simulation = True

#     hopp_site = SiteInfo(**config.hopp_config["site"])
#     hi = HOPPInterface(hopp_site)
#     simulator = RealTimeSimulation(config, hi)


#     mpc_config = config.greenheart_config["realtime_simulation"]["dispatch"]["mpc"]
    
#     # Minimal required attributes for instantiation
#     ctrl = DispatchModelPredictiveController(
#         config=config,
#         simulation_graph=simulator.G,
#         node_order=simulator.node_order,
#         edge_order=simulator.edge_order,
#         mpc_config=mpc_config,
#     )

#     []
