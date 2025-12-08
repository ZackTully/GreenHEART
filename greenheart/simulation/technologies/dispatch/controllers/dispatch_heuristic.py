import numpy as np
import pandas as pd
import networkx as nx


from greenheart.simulation.technologies.steel.steel import Feedstocks
from greenheart.simulation.technologies.dispatch.controllers.controller_tools.control_model_builder import (
    ControlModelBuilder,
)
from hopp.utilities import load_yaml

np.set_printoptions(linewidth=200)


class SimpleSystemController:
    def __init__(
        self, config, simulation_graph, node_order, edge_order, dispatch_config
    ):
        self.config = config

        self.node_order = node_order
        self.edge_order = edge_order

        system_graph = load_yaml(
            config.greenheart_config["realtime_simulation"]["system"][
                "system_graph_config"
            ]
        )
        self.traversal_order = system_graph["traversal_order"]
        self.G = simulation_graph

        self.control_model = ControlModelBuilder(mpc=self)
        self.control_model.build_control_model(self.traversal_order, self.G)
        pass

    def step(self, G, available_power, forecast, x_measured, step_index):

        for node in list(G.nodes):
            G.nodes[node].update({"dispatch_split": np.array([[1]])})
            G.nodes[node].update({"dispatch_ctrl": np.array([[0]])})

        mean_power = 500e3

        bes_charge = max(available_power - mean_power, 0)
        bes_discharge = max(mean_power - available_power, 0)

        # Battery charging and discharging
        uct = np.array([bes_charge, bes_discharge])

        # Generation splitting
        usp = np.array([bes_charge, available_power - bes_charge])

        G.nodes["generation"].update({"dispatch_ctrl": np.array([[0]])})
        G.nodes["generation"].update({"grid_purchase": np.array([[0]])})

        for node in self.control_model.uct_order.keys():
            if len(self.control_model.uct_order[node]) > 0:
                G.nodes[node]["dispatch_ctrl"] = uct

        for node in self.control_model.usp_order.keys():
            if len(self.control_model.usp_order[node]) >= 1:
                G.nodes[node]["dispatch_split"] = usp

        return G


class DispatchHeuristicController:

    def __init__(
        self,
        config=None,
        simulation_graph: nx.Graph = None,
        node_order: list = None,
        edge_order: list = None,
    ):

        self.use_stored = False

        self.config = config

        self.ctrl_config = config.greenheart_config["realtime_simulation"]["dispatch"][
            "heuristic"
        ]
        self.reference = self.ctrl_config["reference"]

        self.G = simulation_graph

        self.node_order = node_order
        self.edge_order = edge_order

        self.control_model = ControlModelBuilder(mpc=self)
        system_graph = load_yaml(
            config.greenheart_config["realtime_simulation"]["system"][
                "system_graph_config"
            ]
        )
        self.traversal_order = system_graph["traversal_order"]

        self.control_model.build_control_model(self.traversal_order, self.G)

        self.calc_no_storage_efficiency()
        self.calc_storage_discharge()

        feedstocks = Feedstocks(None)

        power_kwhptls = feedstocks.electricity_consumption * 1e3
        h2_kgptls = feedstocks.hydrogen_consumption * 1e3

        self.electrolyzer_efficiency_kwhpkg = 53
        power_h2_kwhptls = h2_kgptls * self.electrolyzer_efficiency_kwhpkg

        h2_heating_kwhpkg = 3.67
        power_heating_kwhptls = h2_kgptls * h2_heating_kwhpkg

        power_total = power_kwhptls + power_h2_kwhptls + power_heating_kwhptls

        self.ratio_to_hydrogen = power_h2_kwhptls / power_total
        self.ratio_to_heating = power_heating_kwhptls / power_total
        self.ratio_to_steel = power_kwhptls / power_total

        self.steel_reference = 170.3413  # tonne per hour
        self.power_total_setpoint = power_total * self.steel_reference

        self.power_h2_setpoint = power_h2_kwhptls * self.steel_reference
        self.power_heat_setpoint = power_heating_kwhptls * self.steel_reference
        self.power_steel_setpoint = power_kwhptls * self.steel_reference

        self.h2_setpoint = h2_kgptls * self.steel_reference

        # Load saved data
        # edges_path = "/Users/ztully/Documents/hybrids_code/GH_scripts/greenheart_scripts/minnesota_reference_design/01-minnesota-steel/saved_data/sequential_edges.csv"
        # df_edges = pd.read_csv(edges_path, header=[0, 1], index_col=0)
        # self.edges = list(df_edges.columns)
        # self.df_edges = df_edges

        # ctrl_path = "/Users/ztully/Documents/hybrids_code/GH_scripts/greenheart_scripts/minnesota_reference_design/01-minnesota-steel/saved_data/sequential_ctrl.csv"
        # df_ctrl = pd.read_csv(ctrl_path, header=0, index_col=0)
        # self.ctrl = list(df_ctrl.columns)
        # self.df_ctrl = df_ctrl

        self.horizon = 1

    def calc_no_storage_efficiency(self):

        uct_nonzero_bool = []
        for key in self.control_model.mct_label:
            if "thermal_energy_storage" in key:
                uct_nonzero_bool.append(True)
            else:
                uct_nonzero_bool.append(False)

        usp_nonzero_bool = []
        for key in self.control_model.msp_label:
            if "battery" in key:
                usp_nonzero_bool.append(False)
            elif "hydrogen_storage" in key:
                usp_nonzero_bool.append(False)
            else:
                usp_nonzero_bool.append(True)

        x_nonzero_bool = [False] * self.control_model.n
        d_nonzero_bool = [False] * self.control_model.oex

        # Get columns of block_ss
        column_bool = (
            x_nonzero_bool + uct_nonzero_bool + usp_nonzero_bool + d_nonzero_bool
        )
        all_col_labels = (
            self.control_model.n_label
            + self.control_model.mct_label
            + self.control_model.msp_label
            + self.control_model.oex_label
        )
        nonzero_col_label = [
            cl for i, cl in enumerate(all_col_labels) if column_bool[i]
        ]

        block_ss_narrow = self.control_model.block_ss[:, column_bool]

        A = self.control_model.A[:, x_nonzero_bool]
        Cco = self.control_model.Cco[:, x_nonzero_bool]
        Cex = self.control_model.Cex[:, x_nonzero_bool]
        Cze = self.control_model.Cze[:, x_nonzero_bool]
        Cgt = self.control_model.Cgt[:, x_nonzero_bool]

        Bct = self.control_model.Bct[:, uct_nonzero_bool]
        Dcoct = self.control_model.Dcoct[:, uct_nonzero_bool]
        Dexct = self.control_model.Dexct[:, uct_nonzero_bool]
        Dzect = self.control_model.Dzect[:, uct_nonzero_bool]
        Dgtct = self.control_model.Dgtct[:, uct_nonzero_bool]

        Bsp = self.control_model.Bsp[:, usp_nonzero_bool]
        Dcosp = self.control_model.Dcosp[:, usp_nonzero_bool]
        Dexsp = self.control_model.Dexsp[:, usp_nonzero_bool]
        Dzesp = self.control_model.Dzesp[:, usp_nonzero_bool]
        Dgtsp = self.control_model.Dgtsp[:, usp_nonzero_bool]

        Eex = self.control_model.Eex[:, d_nonzero_bool]
        Fcoex = self.control_model.Fcoex[:, d_nonzero_bool]
        Fexex = self.control_model.Fexex[:, d_nonzero_bool]
        Fzeex = self.control_model.Fzeex[:, d_nonzero_bool]
        Fgtex = self.control_model.Fgtex[:, d_nonzero_bool]

        block = [
            [A, Bct, Bsp, Eex],
            [Cco, Dcoct, Dcosp, Fcoex],
            [Cex, Dexct, Dexsp, Fexex],
            [Cze, Dzect, Dzesp, Fzeex],
            [Cgt, Dgtct, Dgtsp, Fgtex],
        ]

        block_row = [np.block(row) for row in block]

        row_nonzero_bools = [
            np.where(np.sum(np.abs(br), axis=1) != 0, True, False) for br in block_row
        ]

        all_row_labels = (
            self.control_model.n_label
            + self.control_model.pco_label
            + self.control_model.pex_label
            + self.control_model.pze_label
            + self.control_model.pgt_label
        )
        row_bool = np.concatenate(row_nonzero_bools)
        nonzero_row_label = [rl for i, rl in enumerate(all_row_labels) if row_bool[i]]

        blocks_short = [
            [m[row_nonzero_bools[i], :] for m in br] for i, br in enumerate(block)
        ]

        ex_row = np.block(blocks_short[2])
        con_rows = np.block(blocks_short[3:])

        out_cons_mat = np.concatenate([ex_row, con_rows], axis=0)
        cols = np.linalg.solve(
            out_cons_mat,
            np.concatenate([[1], [-4300], np.zeros(con_rows.shape[0] - 1)]),
        )
        cols[0] = cols[1]
        cols[2] = cols[1]

        efficiency = cols[1] + cols[3] + cols[4]

        self.efficiency = efficiency

        # Reconstruct columns for straight pass through no storage
        col_vals = {nonzero_col_label[i]: cols[i] for i in range(len(cols))}

        self.cols_through = {
            k: (col_vals[k] if (k in col_vals) else 0) for k in all_col_labels
        }

        self.steel_from_H2 = self.control_model.Dexsp[
            np.where(np.array(self.pex_label) == "yex 0 steel")[0][0],
            np.where(
                np.array(self.msp_label) == "usp 1 electrolyzer (to heat_exchanger)"
            )[0][0],
        ]

        self.H2_from_P = self.control_model.Dzesp[
            np.where(np.array(self.pze_label) == "yze 0 electrolyzer")[0][0],
            np.where(np.array(self.msp_label) == "usp 2 generation (to electrolyzer)")[
                0
            ][0],
        ]
        self.Q_to_H2 = self.control_model.Dgtsp[
            np.where(np.array(self.pgt_label) == "ygt 0 heat_exchanger")[0][0],
            np.where(
                np.array(self.msp_label) == "usp 1 electrolyzer (to heat_exchanger)"
            )[0][0],
        ]
        self.P_to_steel_per_H2 = self.control_model.Dgtsp[
            np.where(np.array(self.pgt_label) == "ygt 0 steel")[0][0],
            np.where(
                np.array(self.msp_label) == "usp 1 electrolyzer (to heat_exchanger)"
            )[0][0],
        ]

        self.P_to_steel = np.abs(self.P_to_steel_per_H2 / self.steel_from_H2)
        self.Q_to_steel = np.abs(self.Q_to_H2 / self.steel_from_H2)
        self.P_to_H2_to_steel = np.abs(1 / self.H2_from_P / self.steel_from_H2)

        # ex_row_zero = np.where(ex_row == 0, True, False)[0, :]
        # con_row_zero_rows = np.where(con_rows[:, np.bitwise_not(ex_row_zero)] == 0, True, False)[:, 0]

        # np.bitwise_not(ex_row_zero)

        # ex_col_vals = ex_row.T * 1
        # zero_ex_col_locs = np.where(ex_col_vals==0, True, False)[:, 0]

        # cons_defined = con_rows[:, np.bitwise_not(zero_ex_col_locs)]
        # cons_undefined = con_rows[:, zero_ex_col_locs]

        # cons_X = cons_defined @ ex_col_vals[np.bitwise_not(ex_row_zero)]
        # cons_A = cons_undefined

        # nonzero_cons_X = np.where(cons_X != 0)[0]
        # nonzero_cols_A = np.where(np.sum(np.abs(cons_A[nonzero_cons_X, :]), axis=0) !=0)[0]

        # cons_A_nonzero = cons_A[nonzero_cons_X, :][:, nonzero_cols_A]
        # cons_X_nonzero = cons_X[nonzero_cons_X]

        # cons_sol = np.linalg.solve(cons_A_nonzero, cons_X_nonzero)

        # ex_col_vals[np.where(zero_ex_col_locs)[0][nonzero_cols_A] ] = cons_sol

        pass

    def calc_storage_discharge(self):

        uct_nonzero_bool = []
        for key in self.control_model.mct_label:
            if "uct 1" in key:
                uct_nonzero_bool.append(True)
            else:
                uct_nonzero_bool.append(False)

        usp_nonzero_bool = []
        for key in self.control_model.msp_label:
            if "battery (" in key:
                usp_nonzero_bool.append(True)
            else:
                usp_nonzero_bool.append(False)

        x_nonzero_bool = [False] * self.control_model.n
        d_nonzero_bool = [False] * self.control_model.oex

        # Get columns of block_ss
        column_bool = (
            x_nonzero_bool + uct_nonzero_bool + usp_nonzero_bool + d_nonzero_bool
        )
        all_col_labels = (
            self.control_model.n_label
            + self.control_model.mct_label
            + self.control_model.msp_label
            + self.control_model.oex_label
        )
        nonzero_col_label = [
            cl for i, cl in enumerate(all_col_labels) if column_bool[i]
        ]

        block_ss_narrow = self.control_model.block_ss[:, column_bool]

        A = self.control_model.A[:, x_nonzero_bool]
        Cco = self.control_model.Cco[:, x_nonzero_bool]
        Cex = self.control_model.Cex[:, x_nonzero_bool]
        Cze = self.control_model.Cze[:, x_nonzero_bool]
        Cgt = self.control_model.Cgt[:, x_nonzero_bool]

        Bct = self.control_model.Bct[:, uct_nonzero_bool]
        Dcoct = self.control_model.Dcoct[:, uct_nonzero_bool]
        Dexct = self.control_model.Dexct[:, uct_nonzero_bool]
        Dzect = self.control_model.Dzect[:, uct_nonzero_bool]
        Dgtct = self.control_model.Dgtct[:, uct_nonzero_bool]

        Bsp = self.control_model.Bsp[:, usp_nonzero_bool]
        Dcosp = self.control_model.Dcosp[:, usp_nonzero_bool]
        Dexsp = self.control_model.Dexsp[:, usp_nonzero_bool]
        Dzesp = self.control_model.Dzesp[:, usp_nonzero_bool]
        Dgtsp = self.control_model.Dgtsp[:, usp_nonzero_bool]

        Eex = self.control_model.Eex[:, d_nonzero_bool]
        Fcoex = self.control_model.Fcoex[:, d_nonzero_bool]
        Fexex = self.control_model.Fexex[:, d_nonzero_bool]
        Fzeex = self.control_model.Fzeex[:, d_nonzero_bool]
        Fgtex = self.control_model.Fgtex[:, d_nonzero_bool]

        block = [
            [A, Bct, Bsp, Eex],
            [Cco, Dcoct, Dcosp, Fcoex],
            [Cex, Dexct, Dexsp, Fexex],
            [Cze, Dzect, Dzesp, Fzeex],
            [Cgt, Dgtct, Dgtsp, Fgtex],
        ]

        block_row = [np.block(row) for row in block]

        row_nonzero_bools = [
            np.where(np.sum(np.abs(br), axis=1) != 0, True, False) for br in block_row
        ]

        all_row_labels = (
            self.control_model.n_label
            + self.control_model.pco_label
            + self.control_model.pex_label
            + self.control_model.pze_label
            + self.control_model.pgt_label
        )
        row_bool = np.concatenate(row_nonzero_bools)
        nonzero_row_label = [rl for i, rl in enumerate(all_row_labels) if row_bool[i]]

        blocks_short = [
            [m[row_nonzero_bools[i], :] for m in br] for i, br in enumerate(block)
        ]

        ex_row = np.block(blocks_short[2])
        con_rows = np.block(blocks_short[3:])

        out_cons_mat = np.concatenate([ex_row, con_rows], axis=0)
        cols = np.linalg.solve(
            out_cons_mat, np.concatenate([[1], [0], np.zeros(con_rows.shape[0] - 1)])
        )

        col_vals = {nonzero_col_label[i]: cols[i] for i in range(len(cols))}

        self.cols_discharge = {
            k: (col_vals[k] if (k in col_vals) else 0) for k in all_col_labels
        }

        pass

    def step(self, G, available_power, forecast, x_measured, step_index):

        # Should return
        # uc_mpc_traj, us_mpc_traj, curtail_mpc_traj, grid_mpc_traj

        if self.use_stored:
            return self.step_from_stored(G, step_index)
        else:
            return self.step_heuristic(
                G, available_power, forecast, x_measured, step_index
            )

    def step_heuristic(self, G, available_power, forecast, x_measured, step_index):
        # Should return
        # uc_mpc_traj, us_mpc_traj, curtail_mpc_traj, grid_mpc_traj

        bes_duration, tes_duration, h2s_duration = self.calc_storage_duration_available(
            x_measured
        )

        # Initial energy balance
        if available_power > self.efficiency * self.reference:
            # excess power, plan on charging

            gen_output = self.reference
            charge_energy = available_power - self.efficiency * self.reference

            cols_through_action = {
                k: v * gen_output for k, v in self.cols_through.items()
            }

            cols_charge_action, curtail = self.calc_storage_charge(
                charge_energy, x_measured
            )
            cols_action = {
                k: cols_through_action[k] + cols_charge_action[k]
                for k in self.cols_through.keys()
            }

            cols_action.update({"curtail": np.array([curtail])})
            pass
        else:
            # insufficient power, plan on discharging

            self.efficiency * self.reference - available_power

            gen_output = available_power / self.efficiency
            discharge_output = self.reference - available_power / self.efficiency

            cols_through_action = {
                k: v * gen_output for k, v in self.cols_through.items()
            }

            if bes_duration < 1 or tes_duration < 1 or h2s_duration < 1:

                max_discharge_output = np.min(
                    [
                        bes_duration* self.reference,
                        tes_duration* self.reference,
                        h2s_duration* self.reference,
                    ]
                )
                cols_discharge_action = {
                    k: v * max_discharge_output for k, v in self.cols_discharge.items()
                }
            else:
                cols_discharge_action = {
                    k: v * discharge_output for k, v in self.cols_discharge.items()
                }

            cols_action = {
                k: cols_through_action[k] + cols_discharge_action[k]
                for k in self.cols_through.keys()
            }

            cols_action.update({"curtail": np.array([0])})

            pass

        cols_action["dex 0 generation"] = available_power

        # block_vec = np.concatenate([[cols_action[k] for k in cols_action.keys() if k is not "curtail"], np.array([available_power])], axis=0)
        block_vec = np.array(
            [cols_action[k] for k in cols_action.keys() if k != "curtail"]
        )

        block_vec[-1] -= cols_action["curtail"]

        yex_ctrl = (
            np.block(
                [
                    self.control_model.Cex,
                    self.control_model.Dexct,
                    self.control_model.Dexsp,
                    self.control_model.Fexex,
                ]
            )
            @ block_vec
        )

        yco_ctrl = (
            np.block(
                [
                    self.control_model.Cco,
                    self.control_model.Dcoct,
                    self.control_model.Dcosp,
                    self.control_model.Fcoex,
                ]
            )
            @ block_vec
        )
        yze_ctrl = (
            np.block(
                [
                    self.control_model.Cze,
                    self.control_model.Dzect,
                    self.control_model.Dzesp,
                    self.control_model.Fzeex,
                ]
            )
            @ block_vec
        )
        ygt_ctrl = (
            np.block(
                [
                    self.control_model.Cgt,
                    self.control_model.Dgtct,
                    self.control_model.Dgtsp,
                    self.control_model.Fgtex,
                ]
            )
            @ block_vec
        )

        if np.any(np.abs(yze_ctrl) > 1):
            pass
        if np.any(np.abs(ygt_ctrl) > 1):
            pass

        self.yco_ctrl = yco_ctrl

        if yex_ctrl < 0.9 * self.reference:
            pass

        if np.any(np.array([bes_duration, tes_duration, h2s_duration]) < 1):
            pass

        G = self.assign_edges(G, cols_action)

        self.cols_action = cols_action

        action_gen = (
            np.sum(
                [
                    cols_action[k]
                    for k in [
                        "usp 0 generation (to battery)",
                        "usp 1 generation (to thermal_energy_storage)",
                        "usp 2 generation (to electrolyzer)",
                        "usp 3 generation (to steel)",
                    ]
                ]
            )
            + cols_action["curtail"]
        )
        if not np.isclose(action_gen, available_power, rtol=0.05):
            pass

        electrolyzer_input = np.sum(
            [
                cols_action[k]
                for k in [
                    "usp 2 generation (to electrolyzer)",
                    "usp 1 battery (to electrolyzer)",
                ]
            ]
        )
        if electrolyzer_input > 720000:
            pass

        return G

    def gen_sum(self, actions):
        return np.sum(
            [
                actions[k]
                for k in [
                    "usp 0 generation (to battery)",
                    "usp 1 generation (to thermal_energy_storage)",
                    "usp 2 generation (to electrolyzer)",
                    "usp 3 generation (to steel)",
                ]
            ]
        )  # + actions["curtail"]

    def calc_storage_charge(self, charge_energy, x_measured):
        x_bes = x_measured[
            np.where(np.array(self.control_model.n_label) == "x 0 battery linear")[0][0]
        ]
        x_tes = x_measured[
            np.where(
                np.array(self.control_model.n_label)
                == "x 1 thermal_energy_storage linear"
            )[0][0]
        ]
        x_h2s = x_measured[
            np.where(
                np.array(self.control_model.n_label) == "x 0 hydrogen_storage linear"
            )[0][0]
        ]

        bes_duration = (
            (1 / self.reference)
            * (x_bes - self.control_model.bounds_verbose["battery"]["x_lb"])
            / self.P_to_steel
        )
        tes_duration = (
            (1 / self.reference)
            * (
                x_tes
                - self.control_model.bounds_verbose["thermal_energy_storage"]["x_lb"][1]
            )
            / self.Q_to_steel
        )
        h2s_duration = (
            (1 / self.reference)
            * (x_h2s - self.control_model.bounds_verbose["hydrogen_storage"]["x_lb"])
            / (1 / self.steel_from_H2)
        )

        bes_soc = (x_bes - self.control_model.bounds_verbose["battery"]["x_lb"]) / (
            self.control_model.bounds_verbose["battery"]["x_ub"]
            - self.control_model.bounds_verbose["battery"]["x_lb"]
        )
        tes_soc = (
            x_tes
            - self.control_model.bounds_verbose["thermal_energy_storage"]["x_lb"][1]
        ) / (
            self.control_model.bounds_verbose["thermal_energy_storage"]["x_ub"][1]
            - self.control_model.bounds_verbose["thermal_energy_storage"]["x_lb"][1]
        )
        h2s_soc = (
            x_h2s - self.control_model.bounds_verbose["hydrogen_storage"]["x_lb"]
        ) / (
            self.control_model.bounds_verbose["hydrogen_storage"]["x_ub"]
            - self.control_model.bounds_verbose["hydrogen_storage"]["x_lb"]
        )

        states = np.array([x_bes, x_tes, x_h2s])
        # durations = np.array([bes_duration, tes_duration, h2s_duration])

        x_ubs = np.array(
            [
                self.control_model.bounds_verbose["battery"]["x_ub"][0],
                self.control_model.bounds_verbose["thermal_energy_storage"]["x_ub"][1],
                self.control_model.bounds_verbose["hydrogen_storage"]["x_ub"][0],
            ]
        )
        x_lbs = np.array(
            [
                self.control_model.bounds_verbose["battery"]["x_lb"][0],
                self.control_model.bounds_verbose["thermal_energy_storage"]["x_lb"][1],
                self.control_model.bounds_verbose["hydrogen_storage"]["x_lb"][0],
            ]
        )

        u_ubs = np.array(
            [
                self.control_model.bounds_verbose["battery"]["u_ub"][0],
                self.control_model.bounds_verbose["thermal_energy_storage"]["u_ub"][0],
                self.control_model.bounds_verbose["hydrogen_storage"]["u_ub"][0],
            ]
        )

        duration_factors = np.array(
            [self.P_to_steel, self.Q_to_steel, (1 / self.steel_from_H2)]
        )

        kwh_factors = np.array([1, 1, 1 / self.H2_from_P])

        durations = (1 / self.reference) * (states - x_lbs) / duration_factors

        charge_durations_state = (
            (1 / self.reference) * (x_ubs - states) / duration_factors
        )
        charge_durations_input = (u_ubs) / duration_factors

        duration_even_split = (kwh_factors * duration_factors) / np.sum(
            kwh_factors * duration_factors
        )

        # (charge_energy * duration_even_split)  /duration_factors / kwh_factors / self.reference

        # How much commodity to charge each with if even
        (charge_energy * duration_even_split) / kwh_factors

        u_ub1 = (x_ubs - states) * kwh_factors

        u_ub2 = u_ubs * kwh_factors

        max_charge_durations = (
            np.min(np.stack([u_ub1, u_ub2]), axis=0)
            / kwh_factors
            / duration_factors
            / self.reference
        )
        max_absorbable_energy = np.sum(np.min(np.stack([u_ub1, u_ub2]), axis=0))

        charge_equalize = (
            np.min(
                np.stack([np.max(durations) - durations, max_charge_durations]), axis=0
            )
            * duration_factors
            * self.reference
            * kwh_factors
        )
        energy_to_equalize = np.sum(charge_equalize)

        if max_absorbable_energy > charge_energy:
            # dont curtail

            curtail = 0
            charge_energy_post_curtail = charge_energy
            pass
        else:
            # curtail

            curtail = charge_energy - max_absorbable_energy
            charge_energy_post_curtail = max_absorbable_energy
            pass

        # possible_durations = charge_energy / duration_factors / self.reference

        # np.max(durations) - durations

        np.min(
            np.stack([np.max(durations) - durations, charge_durations_state]), axis=0
        )

        energy_to_equalize_hours = np.sum(
            np.min(
                np.stack([np.max(durations) - durations, charge_durations_state]),
                axis=0,
            )
            * duration_factors
            * kwh_factors
        )

        if energy_to_equalize > charge_energy_post_curtail:
            # Cannot equalize storage charging hours
            # So distribute energy in that ratio

            charge_energies = (
                (charge_energy_post_curtail / energy_to_equalize)
                * charge_equalize
                / kwh_factors
            )

            # charge_energies = (charge_energy_post_curtail / energy_to_equalize ) * np.min(np.stack([np.max(durations) - durations, charge_durations_state]), axis=0) * duration_factors * kwh_factors

            bes_charge = charge_energies[0]
            tes_charge = charge_energies[1]
            h2s_charge = charge_energies[2]

            pass

        else:

            # Can equalize storage
            # so do that

            # charge_energies = np.min(np.stack([np.max(durations) - durations, charge_durations_state]), axis=0) * duration_factors * kwh_factors
            charge_energies = charge_equalize / kwh_factors

            bes_charge = charge_energies[0]
            tes_charge = charge_energies[1]
            h2s_charge = charge_energies[2]

            # then evenly distribute the rest

            remaining_charge_energy = charge_energy_post_curtail - np.sum(
                charge_energies * kwh_factors
            )

            equalize_charges = np.array([bes_charge, tes_charge, h2s_charge])

            absorbable_energy = np.min(
                [(u_ubs - equalize_charges), (x_ubs - states)], axis=0
            )

            remaining_split = (
                remaining_charge_energy
                * absorbable_energy
                / np.sum(absorbable_energy)
                / kwh_factors
            )

            # remaining_split = duration_even_split * remaining_charge_energy /kwh_factors

            bes_charge += remaining_split[0]
            tes_charge += remaining_split[1]
            h2s_charge += remaining_split[2]

            pass

        cols_charge_action = {k: 0 for k in self.cols_through.keys()}

        # battery
        cols_charge_action["uct 0 battery"] = bes_charge
        cols_charge_action["usp 0 generation (to battery)"] = bes_charge

        # thermal energy storage
        cols_charge_action["uct 0 thermal_energy_storage"] = tes_charge
        cols_charge_action["usp 1 generation (to thermal_energy_storage)"] = tes_charge

        # Hydrogen storage
        cols_charge_action["uct 0 hydrogen_storage"] = h2s_charge
        cols_charge_action["usp 0 electrolyzer (to hydrogen_storage)"] = h2s_charge
        cols_charge_action["usp 2 generation (to electrolyzer)"] = (
            h2s_charge / self.H2_from_P
        )

        if h2s_charge > 0:
            pass

        # Should equal charge_energy
        bes_charge + tes_charge + h2s_charge / self.H2_from_P

        if np.any(durations < 1):
            pass

        if not np.isclose(self.gen_sum(cols_charge_action) + curtail, charge_energy):
            pass

        return cols_charge_action, curtail

    def calc_storage_duration_available(self, x_measured):
        x_bes = x_measured[
            np.where(np.array(self.control_model.n_label) == "x 0 battery linear")[0][0]
        ]
        x_tes = x_measured[
            np.where(
                np.array(self.control_model.n_label)
                == "x 1 thermal_energy_storage linear"
            )[0][0]
        ]
        x_h2s = x_measured[
            np.where(
                np.array(self.control_model.n_label) == "x 0 hydrogen_storage linear"
            )[0][0]
        ]

        bes_duration = (
            (1 / self.reference)
            * (x_bes - self.control_model.bounds_verbose["battery"]["x_lb"])
            / self.P_to_steel
        )
        tes_duration = (
            (1 / self.reference)
            * (
                x_tes
                - self.control_model.bounds_verbose["thermal_energy_storage"]["x_lb"][1]
            )
            / self.Q_to_steel
        )
        h2s_duration = (
            (1 / self.reference)
            * (x_h2s - self.control_model.bounds_verbose["hydrogen_storage"]["x_lb"])
            / (1 / self.steel_from_H2)
        )

        bes_soc = (x_bes - self.control_model.bounds_verbose["battery"]["x_lb"]) / (
            self.control_model.bounds_verbose["battery"]["x_ub"]
            - self.control_model.bounds_verbose["battery"]["x_lb"]
        )
        tes_soc = (
            x_tes
            - self.control_model.bounds_verbose["thermal_energy_storage"]["x_lb"][1]
        ) / (
            self.control_model.bounds_verbose["thermal_energy_storage"]["x_ub"][1]
            - self.control_model.bounds_verbose["thermal_energy_storage"]["x_lb"][1]
        )
        h2s_soc = (
            x_h2s - self.control_model.bounds_verbose["hydrogen_storage"]["x_lb"]
        ) / (
            self.control_model.bounds_verbose["hydrogen_storage"]["x_ub"]
            - self.control_model.bounds_verbose["hydrogen_storage"]["x_lb"]
        )

        states = np.array([x_bes, x_tes, x_h2s])
        # durations = np.array([bes_duration, tes_duration, h2s_duration])

        x_ubs = np.array(
            [
                self.control_model.bounds_verbose["battery"]["x_ub"][0],
                self.control_model.bounds_verbose["thermal_energy_storage"]["x_ub"][1],
                self.control_model.bounds_verbose["hydrogen_storage"]["x_ub"][0],
            ]
        )
        x_lbs = np.array(
            [
                self.control_model.bounds_verbose["battery"]["x_lb"][0],
                self.control_model.bounds_verbose["thermal_energy_storage"]["x_lb"][1],
                self.control_model.bounds_verbose["hydrogen_storage"]["x_lb"][0],
            ]
        )

        u_ubs = np.array(
            [
                self.control_model.bounds_verbose["battery"]["u_ub"][0],
                self.control_model.bounds_verbose["thermal_energy_storage"]["u_ub"][0],
                self.control_model.bounds_verbose["hydrogen_storage"]["u_ub"][0],
            ]
        )

        duration_factors = np.array(
            [self.P_to_steel, self.Q_to_steel, (1 / self.steel_from_H2)]
        )

        kwh_factors = np.array([1, 1, 1 / self.H2_from_P])

        durations = (1 / self.reference) * (states - x_lbs) / duration_factors

        return durations[0], durations[1], durations[2]

    def assign_edges(self, G, actions):

        #     assert (np.array(list(split_dict.values())) >= 0).all()

        for node in list(G.nodes):
            G.nodes[node].update({"dispatch_split": np.array([1])})
            G.nodes[node].update({"dispatch_ctrl": np.array([0])})

        for edge in list(G.edges):
            G.edges[edge].update({"dispatch": 0})

        for node in list(G.nodes):
            if node in ["generation", "battery", "electrolyzer"]:
                out_edges = list(G.out_edges(node))
                dispatch_split = np.zeros(len(out_edges))

                for i in range(len(out_edges)):
                    action_key = [
                        k
                        for k in actions.keys()
                        if ((f"{node} (" in k) and (out_edges[i][-1] in k))
                    ][0]

                    dispatch_split[i] = actions[action_key]

                G.nodes[node].update({"dispatch_split": dispatch_split})

            if node == "battery":
                G.nodes[node].update(
                    {
                        "dispatch_ctrl": np.array(
                            [actions["uct 0 battery"], actions["uct 1 battery"]]
                        )
                    }
                )
            elif node == "hydrogen_storage":
                G.nodes[node].update(
                    {
                        "dispatch_ctrl": np.array(
                            [
                                actions["uct 0 hydrogen_storage"],
                                actions["uct 1 hydrogen_storage"],
                            ]
                        )
                    }
                )
            elif node == "thermal_energy_storage":
                G.nodes[node].update(
                    {
                        "dispatch_ctrl": np.array(
                            [
                                actions["uct 0 thermal_energy_storage"],
                                actions["uct 1 thermal_energy_storage"],
                            ]
                        )
                    }
                )
            elif node == "generation":
                G.nodes[node].update({"dispatch_ctrl": np.array([actions["curtail"]])})
        return G

    # def step_heuristic(self, G, available_power, forecast, x_measured, step_index):

    #     split_dict = {}
    #     ctrl_dict = {}

    #     available_excess = available_power - self.power_total_setpoint
    #     available_deficit = np.max([0, -available_excess])
    #     available_excess = np.max([0, available_excess])

    #     # Hydrogen path
    #     h2_path = self.ratio_to_hydrogen * available_power

    #     bes2el = 0
    #     gen2el = h2_path

    #     h2_generation = (gen2el + bes2el) / self.electrolyzer_efficiency_kwhpkg

    #     el2h2s = h2_generation - self.h2_setpoint
    #     h2s2hx = -el2h2s

    #     el2h2s = np.max([0, el2h2s])
    #     h2s2hx = np.max([0, h2s2hx])
    #     el2hx = h2_generation - el2h2s
    #     # el2hx = self.h2_setpoint - el2h2s

    #     h2s_charging = el2h2s
    #     h2s_discharging = h2s2hx

    #     # Power path
    #     P_path = self.ratio_to_steel * available_power

    #     gen2bes = P_path - self.power_steel_setpoint
    #     bes2steel = -gen2bes
    #     gen2steel = P_path - gen2bes

    #     gen2bes = np.max([0, gen2bes])
    #     bes2steel = np.max([0, bes2steel])

    #     bes_charging = gen2bes
    #     bes_discharging = bes2steel

    #     # Heat path

    #     Q_path = self.ratio_to_heating * available_power
    #     gen2tes = Q_path - self.power_heat_setpoint
    #     tes2hx = -gen2tes
    #     gen2hx = Q_path - gen2tes

    #     tes2hx = np.max([0, tes2hx])
    #     gen2tes = np.max([0, gen2tes])

    #     tes_charging = gen2tes
    #     tes_discharging = tes2hx

    #     bes2tes = 0
    #     bes2hx = 0

    #     split_dict = {
    #         ("generation", "battery"): gen2bes,
    #         ("generation", "electrolyzer"): gen2el,
    #         ("generation", "thermal_energy_storage"): gen2tes,
    #         ("generation", "heat_exchanger"): gen2hx,
    #         ("generation", "steel"): gen2steel,
    #         ("battery", "electrolyzer"): bes2el,
    #         ("battery", "thermal_energy_storage"): bes2tes,
    #         ("battery", "heat_exchanger"): bes2hx,
    #         ("battery", "steel"): bes2steel,
    #         ("electrolyzer", "hydrogen_storage"): el2h2s,
    #         ("electrolyzer", "heat_exchanger"): el2hx,
    #         ("hydrogen_storage", "heat_exchanger"): h2s2hx,
    #         ("thermal_energy_storage", "heat_exchanger"): tes2hx,
    #         # ('heat_exchanger', 'steel'): hx2steel
    #     }
    #     ctrl_dict = {
    #         "bes_charge": bes_charging,
    #         "bes_discharge": bes_discharging,
    #         "h2s_charge": h2s_charging,
    #         "h2s_discharge": h2s_discharging,
    #         "tes_charge": tes_charging,
    #         "tes_discharge": tes_discharging,
    #     }

    #     assert (np.array(list(split_dict.values())) >= 0).all()

    #     for node in list(G.nodes):
    #         G.nodes[node].update({"dispatch_split": np.array([1])})
    #         G.nodes[node].update({"dispatch_ctrl": np.array([0])})

    #     for edge in list(G.edges):
    #         G.edges[edge].update({"dispatch": 0})

    #     for node in list(G.nodes):
    #         if node in ["generation", "battery", "electrolyzer"]:
    #             out_edges = list(G.out_edges(node))
    #             dispatch_split = np.zeros(len(out_edges))
    #             for i in range(len(out_edges)):
    #                 dispatch_split[i] = split_dict[out_edges[i]]
    #             G.nodes[node].update({"dispatch_split": dispatch_split})

    #         if node == "battery":
    #             G.nodes[node].update(
    #                 {
    #                     "dispatch_ctrl": np.array(
    #                         [ctrl_dict["bes_charge"], ctrl_dict["bes_discharge"]]
    #                     )
    #                 }
    #             )
    #         elif node == "hydrogen_storage":
    #             G.nodes[node].update(
    #                 {
    #                     "dispatch_ctrl": np.array(
    #                         [ctrl_dict["h2s_charge"], ctrl_dict["h2s_discharge"]]
    #                     )
    #                 }
    #             )
    #         elif node == "thermal_energy_storage":
    #             G.nodes[node].update(
    #                 {
    #                     "dispatch_ctrl": np.array(
    #                         [ctrl_dict["tes_charge"], ctrl_dict["tes_discharge"]]
    #                     )
    #                 }
    #             )
    #     return G

    def step_from_stored(self, G, step_index):

        for node in list(G.nodes):
            G.nodes[node].update({"dispatch_split": np.array([1])})
            G.nodes[node].update({"dispatch_ctrl": np.array([0])})

        for edge in list(G.edges):
            G.edges[edge].update({"dispatch": 0})

        for node in list(G.nodes):
            if node == "generation":

                out_edges = list(G.out_edges(node))
                dispatch_split = np.zeros(len(out_edges))
                for i in range(len(out_edges)):
                    dispatch_split[i] = self.df_edges[out_edges[i]].iloc[step_index]

                G.nodes[node].update({"dispatch_split": dispatch_split})

            elif node == "battery":

                out_edges = list(G.out_edges(node))
                dispatch_split = np.zeros(len(out_edges))
                for i in range(len(out_edges)):
                    dispatch_split[i] = self.df_edges[out_edges[i]].iloc[step_index]

                G.nodes[node].update({"dispatch_split": dispatch_split})

                G.nodes[node].update(
                    {
                        "dispatch_ctrl": np.array(
                            [
                                self.df_ctrl["bes_charge"].iloc[step_index],
                                self.df_ctrl["bes_discharge"].iloc[step_index],
                            ]
                        )
                    }
                )

            elif node == "electrolyzer":

                out_edges = list(G.out_edges(node))
                dispatch_split = np.zeros(len(out_edges))
                for i in range(len(out_edges)):
                    dispatch_split[i] = self.df_edges[out_edges[i]].iloc[step_index]

                G.nodes[node].update({"dispatch_split": dispatch_split})

            elif node == "hydrogen_storage":
                G.nodes[node].update(
                    {
                        "dispatch_ctrl": np.array(
                            [
                                self.df_ctrl["h2s_charge"].iloc[step_index],
                                self.df_ctrl["h2s_discharge"].iloc[step_index],
                            ]
                        )
                    }
                )

            elif node == "thermal_energy_storage":
                G.nodes[node].update(
                    {
                        "dispatch_ctrl": np.array(
                            [
                                self.df_ctrl["tes_charge"].iloc[step_index],
                                self.df_ctrl["tes_discharge"].iloc[step_index],
                            ]
                        )
                    }
                )

        return G
