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
            self.controller = DispatchHeuristicController()
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

    # def step_heuristic_case3(self, G, available_power):

    #     mean_generation = 150000  # kWh / hr
    #     # mean_generation = 142000  # kWh / hr

    #     for edge in list(G.edges):
    #         G.edges[edge].update({"dispatch": 0})

    #     for node in list(G.nodes):

    #         # Split control
    #         if G.out_degree[node] > 1:
    #             G.nodes[node].update(
    #                 {
    #                     "dispatch_split": np.nan_to_num(
    #                         (
    #                             (available_power / G.out_degree[node])
    #                             * np.ones(G.out_degree[node])
    #                         )
    #                         / available_power
    #                     )
    #                 }
    #             )
    #         else:
    #             G.nodes[node].update({"dispatch_split": []})

    #         # Control control
    #         # if node in ["battery", "thermal_energy_storage", "hydrogen_storage"]:
    #         #     G.nodes[node].update(
    #         #         {"dispatch_ctrl": [1 / 5 * (available_power - mean_generation)]}
    #         #     )
    #         # else:
    #         #     G.nodes[node].update({"dispatch_ctrl": []})

    #         G.nodes[node].update({"dispatch_ctrl": np.array([0])})

    #     G.nodes["battery"].update(
    #         {"dispatch_ctrl": np.array([available_power - mean_generation])}
    #     )
    #     G.nodes["hydrogen_storage"].update(
    #         {"dispatch_ctrl": np.array([1 / 55 * (available_power - mean_generation)])}
    #     )

    #     return G

    # def splitting_controller(self, G, available_power, feedback_error=None):

    #     mean_generation = 150000  # kWh / hr
    #     frac_electrolysis = 0.944
    #     frac_heating = 0.056
    #     mean_to_hydrogen = frac_electrolysis * mean_generation
    #     mean_to_heating = frac_heating * mean_generation

    #     power_to_hydrogen = frac_electrolysis * available_power

    #     gen_to_EL = np.min([mean_to_hydrogen, power_to_hydrogen])
    #     gen_to_BES = np.max([0, power_to_hydrogen - mean_to_hydrogen])
    #     BES_to_EL = np.max([0, mean_to_hydrogen - power_to_hydrogen])
    #     EL_to_HX = 1 / 55 * (gen_to_EL + BES_to_EL)

    #     # Heat path

    #     power_to_heating = frac_heating * available_power

    #     gen_to_HX = np.min([mean_to_heating, power_to_heating])
    #     gen_to_TES = np.max([0, power_to_heating - mean_to_heating])
    #     TES_to_HX = np.max([0, mean_to_heating - power_to_heating])

    #     HX_to_output = mean_to_hydrogen / 55

    #     gen_split = np.array([gen_to_EL, gen_to_BES, gen_to_HX, gen_to_TES])
    #     gen_split_normalized = np.nan_to_num(gen_split / np.sum(gen_split))

    #     # BES_charge = np.sign(gen_to_BES)
    #     # if gen_to_BES == 0:
    #     #     BES_charge = 0

    #     # TES_charge = np.sign(gen_to_TES)
    #     # if gen_to_TES == 0:
    #     #     TES_charge = 0

    #     if gen_to_BES > 0:
    #         BES_charge = gen_to_BES
    #     elif BES_to_EL > 0:
    #         BES_charge = -BES_to_EL
    #     else:
    #         BES_charge = 0

    #     if gen_to_TES > 0:
    #         TES_charge = gen_to_TES
    #     elif TES_to_HX > 0:
    #         TES_charge = -TES_to_HX
    #     else:
    #         TES_charge = 0

    #     for node in list(G.nodes):
    #         G.nodes[node].update({"dispatch_split": np.array([1])})
    #         G.nodes[node].update({"dispatch_ctrl": None})

    #     G.nodes["generation"].update({"dispatch_split": gen_split_normalized})
    #     G.nodes["battery"].update({"dispatch_ctrl": BES_charge})
    #     G.nodes["thermal_energy_storage"].update({"dispatch_ctrl": TES_charge})

    #     dispatch_IO = {
    #         ("generation", "electrolyzer"): {"dispatch": [gen_to_EL, 0, 0, 0]},
    #         ("generation", "battery"): {"dispatch": [gen_to_BES, 0, 0, 0]},
    #         ("battery", "electrolyzer"): {"dispatch": [BES_to_EL, 0, 0, 0]},
    #         ("electrolyzer", "heat_exchanger"): {"dispatch": [0, 0, EL_to_HX, 80]},
    #         ("generation", "thermal_energy_storage"): {
    #             "dispatch": [gen_to_TES, 0, 0, 0]
    #         },
    #         ("generation", "heat_exchanger"): {"dispatch": [gen_to_HX, 0, 0, 0]},
    #         ("thermal_energy_storage", "heat_exchanger"): {
    #             "dispatch": [0, TES_to_HX, 0, 0]
    #         },
    #         ("heat_exchanger", "output"): {"dispatch": [0, 0, HX_to_output, 900]},
    #     }

    #     nx.set_edge_attributes(G, dispatch_IO)

    #     return G

    # def example_control_BES_TES_heat_exchanger(
    #     self, G, available_power, feedback_error=None
    # ):

    #     # Plan for CF = 0.3, 150 MW average generation

    #     mean_generation = 150000  # kWh / hr

    #     frac_electrolysis = 0.944
    #     frac_heating = 0.056

    #     # Hydrogen path
    #     mean_to_hydrogen = frac_electrolysis * mean_generation

    #     power_to_hydrogen = frac_electrolysis * available_power

    #     gen_to_EL = np.min([mean_to_hydrogen, power_to_hydrogen])
    #     gen_to_BES = np.max([0, power_to_hydrogen - mean_to_hydrogen])
    #     BES_to_EL = np.max([0, mean_to_hydrogen - power_to_hydrogen])
    #     EL_to_HX = 1 / 55 * (gen_to_EL + BES_to_EL)

    #     # Heat path
    #     mean_to_heating = frac_heating * mean_generation

    #     power_to_heating = frac_heating * available_power

    #     gen_to_HX = np.min([mean_to_heating, power_to_heating])
    #     gen_to_TES = np.max([0, power_to_heating - mean_to_heating])
    #     TES_to_HX = np.max([0, mean_to_heating - power_to_heating])

    #     HX_to_output = mean_to_hydrogen / 55

    #     dispatch_IO = {
    #         ("generation", "electrolyzer"): {"dispatch": [gen_to_EL, 0, 0, 0]},
    #         ("generation", "battery"): {"dispatch": [gen_to_BES, 0, 0, 0]},
    #         ("battery", "electrolyzer"): {"dispatch": [BES_to_EL, 0, 0, 0]},
    #         ("electrolyzer", "heat_exchanger"): {"dispatch": [0, 0, EL_to_HX, 80]},
    #         ("generation", "thermal_energy_storage"): {
    #             "dispatch": [gen_to_TES, 0, 0, 0]
    #         },
    #         ("generation", "heat_exchanger"): {"dispatch": [gen_to_HX, 0, 0, 0]},
    #         ("thermal_energy_storage", "heat_exchanger"): {
    #             "dispatch": [0, TES_to_HX, 0, 0]
    #         },
    #         ("heat_exchanger", "output"): {"dispatch": [0, 0, HX_to_output, 900]},
    #     }

    #     # TODO Update to get the error signal from G not from feedback_error
    #     if feedback_error is not None:
    #         error_IO = nx.get_edge_attributes(feedback_error, "error")

    #         for key in dispatch_IO.keys():
    #             self.dispatch_IO[key]["value"] = (
    #                 dispatch_IO[key]["value"] - error_IO[key]
    #             )
    #             dispatch_IO = self.dispatch_IO
    #     else:
    #         self.dispatch_IO = dispatch_IO

    #     nx.set_edge_attributes(G, dispatch_IO)

    #     # pprint.pprint(nx.get_edge_attributes(G_dispatch, "value"))

    #     return G

    # def example_control_elec_heat_exchanger(self, G_dispatch, available_power):

    #     gen_to_curtail = [np.max([available_power - 50000, 0]), 0, 0, 0]
    #     gen_to_el = [50000 * 0.95, 0, 0, 0]
    #     el_to_hx = [0, 0, (1 / 55) * gen_to_el[0], 80]
    #     # el_to_hx = [0, 0, 500, 80]
    #     # gen_to_hx = [available_power / 2, 0, 0, 0]
    #     gen_to_hx = [50000 * 0.05, 0, 0, 0]

    #     nx.set_edge_attributes(
    #         G_dispatch,
    #         {
    #             ("generation", "curtail"): {"value": gen_to_curtail},
    #             ("generation", "electrolyzer"): {"value": gen_to_el},
    #             ("electrolyzer", "heat_exchanger"): {"value": el_to_hx},
    #             ("generation", "heat_exchanger"): {"value": gen_to_hx},
    #         },
    #     )

    #     return G_dispatch

    # def example_control_hydrogen_heat(self, G_dispatch, available_power):

    #     for edge in list(G_dispatch.edges):
    #         G_dispatch.edges[edge].update({"value": available_power / 3})

    #     return G_dispatch

    # def example_control_elec_storage_steel(self, G_dispatch, available_power):
    #     electrolyzer_eta = 1 / 55  # kg/kWh
    #     average_generation = (
    #         0.3 * (10 * 6000) + 0.3 * 100000
    #     )  # wind + solar rated * 0.3 capacity factor

    #     # average_generation = 50000

    #     # Always send all of the power through the electrolyzer
    #     # If available is greater than average, charge the storage by the difference
    #     # If available is less than average, discharge the storage by the difference

    #     available_more_than_average_difference = np.max(
    #         [0, available_power - average_generation]
    #     )
    #     available_less_than_average_difference = np.max(
    #         [0, average_generation - available_power]
    #     )

    #     generation_to_curtail = 0
    #     generation_to_electrolyzer = available_power
    #     electrolyzer_to_storage = (
    #         electrolyzer_eta * available_more_than_average_difference
    #     )
    #     # electrolyzer_to_storage = electrolyzer_eta * (available_power - average_generation)

    #     electrolyzer_to_steel = electrolyzer_eta * average_generation
    #     storage_to_steel = electrolyzer_eta * available_less_than_average_difference

    #     assert not np.isnan(storage_to_steel)

    #     dispatch_edges = list(G_dispatch.edges)
    #     dispatch_IO = {
    #         ("generation", "curtail"): {"value": generation_to_curtail},
    #         ("generation", "electrolyzer"): {"value": generation_to_electrolyzer},
    #         ("electrolyzer", "hydrogen_storage"): {"value": electrolyzer_to_storage},
    #         ("electrolyzer", "steel"): {"value": electrolyzer_to_steel},
    #         ("hydrogen_storage", "steel"): {"value": storage_to_steel},
    #     }

    #     nx.set_edge_attributes(G_dispatch, dispatch_IO)

    #     if False:
    #         layout = nx.random_layout(G_dispatch)
    #         nx.draw_networkx(G_dispatch, pos=layout, with_labels=True)
    #         nx.draw_networkx_edge_labels(
    #             G_dispatch, pos=layout, edge_labels=dispatch_IO
    #         )

    #     return G_dispatch
