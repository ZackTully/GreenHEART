import numpy as np
import networkx as nx
import pprint

from greenheart.simulation.technologies.dispatch.controllers.dispatch_mpc import (
    DispatchModelPredictiveController,
)

# from greenheart.simulation.technologies.dispatch.controllers.dispatch_mpc_pyomo import DispatchModelPredictiveController
from greenheart.simulation.technologies.dispatch.controllers.dispatch_heuristic import (
    DispatchHeuristicController,
    SimpleSystemController,
)


class GreenheartDispatch:
    controller: DispatchHeuristicController | DispatchModelPredictiveController

    def __init__(self, GHconfig, simulator, dispatch_config):

        self.use_MPC = "mpc" in dispatch_config

        self.update_period = dispatch_config["update_period"]

        if self.use_MPC:
            self.setup_MPC(GHconfig, simulator, dispatch_config)
        else:
            self.use_heuristic = True
            self.setup_heuristic(GHconfig, simulator, dispatch_config)

        self.validation = False
        if self.validation:
            self.setup_validation()

    def setup_MPC(self, GHconfig, simulator, dispatch_config):
        self.uc_mpc = None
        self.us_mpc = None
        self.previous_update = 0

        mpc_config = dispatch_config["mpc"]

        self.controller = DispatchModelPredictiveController(
            GHconfig,
            simulator.G,
            node_order=simulator.node_order,
            edge_order=simulator.edge_order,
            mpc_config=mpc_config,
        )

    def setup_heuristic(self, GHconfig, simulator, dispatch_config):

        if "simple_system" in dispatch_config:
            self.controller =SimpleSystemController(
                GHconfig,
                simulator.G,
                node_order=simulator.node_order,
                edge_order=simulator.edge_order,
                dispatch_config=dispatch_config,
            )
            self.heuristic_ctrl = "simple_system"
        else:
            self.controller = DispatchHeuristicController(
                config=GHconfig,
                simulation_graph=simulator.G,
                node_order=simulator.node_order,
                edge_order=simulator.edge_order,
            )

            self.heuristic_ctrl = "default"

    def step(
        self,
        G,
        available_power,
        forecast=None,
        x_measured=None,
        feedback_error=None,
        step_index=None,
    ):

        if self.use_MPC:
            G = self.step_MPC(G, available_power, forecast, x_measured, step_index)
        elif self.validation:
            G = self.step_validation(G, available_power, step_index)
        elif self.use_heuristic:
            if self.heuristic_ctrl == "simple_system":
                G = self.step_simple_system(
                    G, available_power, forecast, x_measured, step_index
                )
            else:
                G = self.step_heuristic(
                    G, available_power, forecast, x_measured, step_index
                )

        return G

    def step_simple_system(self, G, available_power, forecast, x_measured, step_index):
        G = self.controller.step(G, available_power, forecast, x_measured, step_index)
        return G

    def step_heuristic(self, G, available_power, forecast, x_measured, step_index):
        G = self.controller.step(G, available_power, forecast, x_measured, step_index)
        return G

    def step_MPC(self, G_dispatch, available_power, forecast, x_measured, step_index):

        G = G_dispatch
        error_flag = False
        if (step_index > 0) and (len(self.controller.step_index_store) > 0):
            mpc_state = self.controller.x_store[-1][
                :, step_index - self.controller.step_index_store[-1]
            ]
            frac_error = np.abs(x_measured - mpc_state) / (
                0.5 * (x_measured + mpc_state)
            )

            if np.any(frac_error > 0.1):
                self.uc_mpc_traj[[0, 1], 0 : (step_index - self.previous_update)]
                G.nodes["battery"]["ionode"].model.store_charge_power[
                    self.previous_update : step_index
                ]
                step_index - self.previous_update
                error_flag = True

        x0 = x_measured

        if not (step_index % self.update_period) or (step_index == 0) or error_flag:
            uc_mpc_traj, us_mpc_traj, curtail_mpc_traj, grid_mpc_traj = (
                self.controller.compute_trajectory(x0, forecast, step_index)
            )

            self.uc_mpc_traj = uc_mpc_traj
            self.us_mpc_traj = us_mpc_traj

            self.curtail_mpc_traj = np.atleast_2d(curtail_mpc_traj)
            self.grid_mpc_traj = np.atleast_2d(grid_mpc_traj)
            self.previous_update = step_index

        uc_mpc = self.uc_mpc_traj[:, step_index - self.previous_update]
        us_mpc = self.us_mpc_traj[:, step_index - self.previous_update]
        curtail_mpc = self.curtail_mpc_traj[:, step_index - self.previous_update]
        grid_mpc = self.grid_mpc_traj[:, step_index - self.previous_update]

        for node in list(G.nodes):
            # G.nodes[node].update({"dispatch_split": np.array([1])})
            G.nodes[node].update({"dispatch_split": np.array([[1]])})
            G.nodes[node].update({"dispatch_ctrl": np.array([[0]])})

        # for edge in list(G_dispatch.edges):
        #     G_dispatch.edges[edge].update({"dispatch": 0})

        G.nodes["generation"].update({"dispatch_ctrl": [curtail_mpc]})
        G.nodes["generation"].update({"grid_purchase": [grid_mpc]})

        for node in self.controller.uct_order.keys():
            if len(self.controller.uct_order[node]) > 0:
                G.nodes[node]["dispatch_ctrl"] = uc_mpc[self.controller.uct_order[node]]

        for node in self.controller.usp_order.keys():
            if len(self.controller.usp_order[node]) >= 1:
                G.nodes[node]["dispatch_split"] = us_mpc[
                    self.controller.usp_order[node]
                ]

        return G

    def setup_validation(self):
        validation_data = np.load(
            "/Users/ztully/Documents/hybrids_code/GH_scripts/dispatch/comparison/data/seq_data.npz"
        )
        self.validation_data = validation_data

    def step_validation(self, G, available_power, step_index):

        # self.validation_data["generation"][step_index] - available_power

        for node in list(G.nodes):
            G.nodes[node].update({"dispatch_split": np.array([1])})
            G.nodes[node].update({"dispatch_ctrl": np.array([0])})

        G.nodes["generation"].update(
            {
                "dispatch_split": np.nan_to_num(
                    np.array(
                        [
                            self.validation_data["gen2bes"][step_index],
                            self.validation_data["gen2el"][step_index],
                        ]
                    )
                    / np.sum(
                        [
                            self.validation_data["gen2bes"][step_index],
                            self.validation_data["gen2el"][step_index],
                        ]
                    )
                )
            }
        )

        G.nodes["electrolyzer"].update(
            {
                "dispatch_split": np.nan_to_num(
                    np.array(
                        [
                            self.validation_data["h2s_charge"][step_index],
                            self.validation_data["EL_h2_gen"][step_index]
                            - self.validation_data["h2s_charge"][step_index],
                        ]
                    )
                    / np.sum(
                        [
                            self.validation_data["h2s_charge"][step_index],
                            self.validation_data["EL_h2_gen"][step_index]
                            - self.validation_data["h2s_charge"][step_index],
                        ]
                    )
                )
            }
        )

        G.nodes["battery"].update(
            {
                "dispatch_ctrl": np.array(
                    [
                        self.validation_data["bes_charge"][step_index]
                        - self.validation_data["bes_discharge"][step_index]
                    ]
                )
            }
        )

        G.nodes["hydrogen_storage"].update(
            {
                "dispatch_ctrl": np.array(
                    [
                        self.validation_data["h2s_charge"][step_index]
                        - self.validation_data["h2s_discharge"][step_index]
                    ]
                )
            }
        )

        for edge in list(G.edges):
            G.edges[edge].update({"dispatch": 0})

        if False:
            nx.get_node_attributes(G, "dispatch_ctrl")
            nx.get_node_attributes(G, "dispatch_split")

        return G

  