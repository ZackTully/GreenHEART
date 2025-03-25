import numpy as np
import pandas as pd
import networkx as nx


from greenheart.simulation.technologies.steel.steel import Feedstocks


class DispatchHeuristicController:

    def __init__(self):

        self.use_stored = False

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
        edges_path = "/Users/ztully/Documents/hybrids_code/GH_scripts/greenheart_scripts/minnesota_reference_design/01-minnesota-steel/saved_data/sequential_edges.csv"
        df_edges = pd.read_csv(edges_path, header=[0, 1], index_col=0)
        self.edges = list(df_edges.columns)
        self.df_edges = df_edges

        ctrl_path = "/Users/ztully/Documents/hybrids_code/GH_scripts/greenheart_scripts/minnesota_reference_design/01-minnesota-steel/saved_data/sequential_ctrl.csv"
        df_ctrl = pd.read_csv(ctrl_path, header=0, index_col=0)
        self.ctrl = list(df_ctrl.columns)
        self.df_ctrl = df_ctrl

        self.horizon = 1

    def step(self, G, available_power, forecast, x_measured, step_index):
        if self.use_stored:
            return self.step_from_stored(G, step_index)
        else:
            return self.step_heuristic(
                G, available_power, forecast, x_measured, step_index
            )

    def step_heuristic(self, G, available_power, forecast, x_measured, step_index):

        split_dict = {}
        ctrl_dict = {}

        available_excess = available_power - self.power_total_setpoint
        available_deficit = np.max([0, -available_excess])
        available_excess = np.max([0, available_excess])


        # Hydrogen path
        h2_path = self.ratio_to_hydrogen * available_power

        bes2el = 0
        gen2el = h2_path

        h2_generation = (gen2el + bes2el) / self.electrolyzer_efficiency_kwhpkg

        el2h2s = h2_generation - self.h2_setpoint
        h2s2hx = -el2h2s

        el2h2s = np.max([0, el2h2s])
        h2s2hx = np.max([0, h2s2hx])
        el2hx = h2_generation - el2h2s
        # el2hx = self.h2_setpoint - el2h2s

        h2s_charging = el2h2s
        h2s_discharging = h2s2hx


        # Power path
        P_path = self.ratio_to_steel * available_power


        gen2bes = P_path - self.power_steel_setpoint
        bes2steel = -gen2bes
        gen2steel = P_path - gen2bes

        gen2bes = np.max([0, gen2bes])
        bes2steel = np.max([0, bes2steel])


        bes_charging = gen2bes
        bes_discharging = bes2steel

        # Heat path

        Q_path = self.ratio_to_heating * available_power
        gen2tes = Q_path - self.power_heat_setpoint
        tes2hx = -gen2tes
        gen2hx = Q_path - gen2tes

        tes2hx = np.max([0, tes2hx])
        gen2tes = np.max([0, gen2tes])

        tes_charging = gen2tes
        tes_discharging = tes2hx

        bes2tes = 0
        bes2hx = 0

        split_dict = {
            ('generation', 'battery'): gen2bes, 
            ('generation', 'electrolyzer'): gen2el,
            ('generation', 'thermal_energy_storage'): gen2tes,
            ('generation', 'heat_exchanger'): gen2hx,
            ('generation', 'steel'): gen2steel,
            ('battery', 'electrolyzer'): bes2el,
            ('battery', 'thermal_energy_storage'): bes2tes,
            ('battery', 'heat_exchanger'): bes2hx,
            ('battery', 'steel'): bes2steel,
            ('electrolyzer', 'hydrogen_storage'): el2h2s,
            ('electrolyzer', 'heat_exchanger'): el2hx,
            ('hydrogen_storage', 'heat_exchanger'): h2s2hx,
            ('thermal_energy_storage', 'heat_exchanger'): tes2hx,
            # ('heat_exchanger', 'steel'): hx2steel
        }
        ctrl_dict = {
            "bes_charge": bes_charging,
            "bes_discharge": bes_discharging,
            "h2s_charge": h2s_charging,
            "h2s_discharge": h2s_discharging,
            "tes_charge": tes_charging,
            "tes_discharge": tes_discharging,
        }

        assert (np.array(list(split_dict.values())) >= 0).all()




        # power_to_hydrogen = available_power * self.ratio_to_hydrogen
        # bes2el = np.max([0, self.power_h2_setpoint - power_to_hydrogen])

        # split_dict.update({("generation", "electrolyzer"): power_to_hydrogen})
        # split_dict.update({("battery", "electrolyzer"): bes2el})

        # h2_generation = (
        #     power_to_hydrogen + bes2el
        # ) / self.electrolyzer_efficiency_kwhpkg

        # h2s_charging = np.max([0, h2_generation - self.h2_setpoint])
        # h2s_discharging = np.max([0, -(h2_generation - self.h2_setpoint)])

        # ctrl_dict.update({"h2s_charge": h2s_charging, "h2s_discharge": h2s_discharging})

        # split_dict.update({("electrolyzer", "hydrogen_storage"): h2s_charging})
        # split_dict.update(
        #     {("electrolyzer", "heat_exchanger"): h2_generation - h2s_charging}
        # )

        # power_to_heating = available_power * self.ratio_to_heating
        # tes_charging = np.max([0, -(self.power_heat_setpoint - power_to_heating)])
        # tes_discharging = np.max([0, (self.power_heat_setpoint - power_to_heating)])
        # ctrl_dict.update({"tes_charge": tes_charging, "tes_discharge": tes_discharging})

        # split_dict.update({("generation", "thermal_energy_storage"): tes_charging})
        # split_dict.update({("battery", "thermal_energy_storage"): 0})

        # split_dict.update(
        #     {("generation", "heat_exchanger"): power_to_heating - tes_charging}
        # )
        # split_dict.update({("battery", "heat_exchanger"): 0})

        # # split_dict.update({("thermal_energy_storage", "heat_exchanger"): tes_discharging})

        # power_to_steel = available_power * self.ratio_to_steel
        # split_dict.update({("generation", "steel"): power_to_steel})
        # bes2steel = np.max([0, self.power_steel_setpoint - power_to_steel])
        # split_dict.update({("battery", "steel"): bes2steel})

        # bes_charging = available_power - np.sum(
        #     [split_dict[key] for key in split_dict.keys() if key[0] == "generation"]
        # )
        # bes_discharging = np.sum(
        #     [split_dict[key] for key in split_dict.keys() if key[0] == "battery"]
        # )

        # ctrl_dict.update({"bes_charge": bes_charging, "bes_discharge": bes_discharging})

        # split_dict.update({("generation", "battery"): bes_charging})

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
                    dispatch_split[i] = split_dict[out_edges[i]]
                G.nodes[node].update({"dispatch_split": dispatch_split})

            if node == "battery":
                G.nodes[node].update(
                    {
                        "dispatch_ctrl": np.array(
                            [ctrl_dict["bes_charge"], ctrl_dict["bes_discharge"]]
                        )
                    }
                )
            elif node == "hydrogen_storage":
                G.nodes[node].update(
                    {
                        "dispatch_ctrl": np.array(
                            [ctrl_dict["h2s_charge"], ctrl_dict["h2s_discharge"]]
                        )
                    }
                )
            elif node == "thermal_energy_storage":
                G.nodes[node].update(
                    {
                        "dispatch_ctrl": np.array(
                            [ctrl_dict["tes_charge"], ctrl_dict["tes_discharge"]]
                        )
                    }
                )
        return G

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
