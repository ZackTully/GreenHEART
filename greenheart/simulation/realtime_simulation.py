import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl

import time

from greenheart.simulation.technologies.dispatch.dispatch import GreenheartDispatch
from greenheart.simulation.technologies.dispatch.control_model import ControlModel
from greenheart.simulation.realtime_node import Node

# Greenheart imports

from greenheart.simulation.technologies.ammonia.ammonia import (
    AmmoniaCapacityModelConfig,
)
from greenheart.simulation.technologies.heat.heat_conversion.joule_heater import (
    JouleHeater,
)
from greenheart.simulation.technologies.heat.heat_exchange.heat_exchanger import (
    HeatExchanger,
)
from greenheart.simulation.technologies.heat.heat_storage.thermal_energy_storage import (
    ThermalEnergyStorage,
)


from greenheart.simulation.technologies.hydrogen.electrolysis.run_PEM_master_STEP import (
    run_PEM_clusters_step,
)

from greenheart.simulation.technologies.hydrogen.h2_storage.hydrogen_storage import (
    HydrogenStorage,
)
from greenheart.simulation.technologies.steel.steel import SteelModel

from greenheart.simulation.technologies.electricity.battery import Battery


from greenheart.tools.eco.utilities import ceildiv


# Utilty imports
from hopp.utilities import load_yaml


# Simulation model for greenheart components


class RealTimeSimulation:
    def __init__(self, config, hopp_interface):

        self.verbose = True

        self.config = config
        self.rts_config = self.config.greenheart_config["realtime_simulation"]
        self.hi = hopp_interface

        if "stop_index" in self.rts_config:
            self.stop_index = self.rts_config["stop_index"]
        else:
            self.stop_index = 8760 + 15

        self.setup_simulation_model(config, hopp_interface)
        self.setup_record_keeping()



    def setup_simulation_model(self, config, hopp_interface):

        GH_tech_options = [
            "generation",
            "curtail",
            "battery",
            "electrolyzer",
            "joule_heater",
            "hydrogen_storage",
            "heat_exchanger",
            "thermal_energy_storage",
            "steel",
            "output",
        ]

        GH_techs = []
        for key in config.greenheart_config.keys():
            if key in GH_tech_options:
                GH_techs.append(key)

        # graph_config_fpath = config.greenheart_config["system"]["system_graph_config"]
        graph_config_fpath = config.greenheart_config["realtime_simulation"]["system"][
            "system_graph_config"
        ]
        graph_config = load_yaml(graph_config_fpath)
        network_config = graph_config["network"]

        edges = network_config
        nodes = []
        for edge in network_config:
            for node in edge:
                if node not in nodes:
                    nodes.append(node)

        GH_techs = nodes

        if "traversal_order" in graph_config.keys():
            self.node_order = graph_config["traversal_order"]

        if "print_locs" in graph_config.keys():
            self.print_locs = graph_config["print_locs"]

        self.edge_order = edges

        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)

        for degree in list(G.out_degree):
            if degree[1] > 1:
                G.nodes[degree[0]].update({"split": True})
            else:
                G.nodes[degree[0]].update({"split": False})

        self.G = G

        # Instantiate the individual steppable models of each technology

        RT_techs = {}

        for GH_tech in GH_techs:

            if GH_tech == "generation":
                RT_techs.update(self._setup_generation_node())
            elif GH_tech == "curtail":
                RT_techs.update(self._setup_curtail_node())
            elif GH_tech == "output":
                RT_techs.update(self._setup_output_node())
            elif GH_tech == "battery":
                RT_techs.update(self._setup_battery_node())
            elif GH_tech == "electrolyzer":
                RT_techs.update(self._setup_electrolyzer_node())
            elif GH_tech == "hydrogen_storage":
                RT_techs.update(self._setup_hydrogen_storage_node())
            elif GH_tech == "thermal_energy_storage":
                RT_techs.update(self._setup_thermal_energy_storage_node())
            elif GH_tech == "heat_exchanger":
                RT_techs.update(self._setup_heat_exchanger_node())
            elif GH_tech == "steel":
                RT_techs.update(self._setup_steel_node())

        # Build the connections with a graph network

        for node in self.G.nodes:
            model = None

            ionode = Node(
                name=node,
                model=RT_techs[node]["model"],
                expected_inputs=RT_techs[node]["model_inputs"],
                expected_outputs=RT_techs[node]["model_outputs"],
                splitting_node=self.G.nodes[node]["split"],
                in_degree=self.G.in_degree[node],
                out_degree=self.G.out_degree[node],
            )

            # Add an assertion that there is one source and one sink
            is_source = False
            is_sink = False
            if graph_config["source_node"] == node:
                is_source = True
            if graph_config["sink_node"] == node:
                is_sink = True

            self.G.nodes[node].update(
                {"ionode": ionode, "is_source": is_source, "is_sink": is_sink}
            )

        for node in list(self.G.nodes):
            # assert (
            #     not self.G.nodes[node]["model"] == None
            # ), f"no model for node: {node}"

            assert (
                not self.G.nodes[node]["ionode"].model == None
            ), f"no model for node: {node}"

        # self.plot_system_graph()

        self.technologies = RT_techs
        # Print or log the control input format

        # find which nodes are splitting nodes by which ones have out degree > 1

        []

    def plot_system_graph(self):

        fig, ax = plt.subplots(1, 1, layout="constrained")
        ax.set_axis_off()
        G = self.G

        # Make multipartite layout for plotting
        indeg = G.in_degree
        root_node = [node for node in indeg if node[1] == 0]
        node_layers = []

        node_layers.append([])
        if len(root_node) > 1:
            shortest_path_list = []
            for rn in root_node:
                node_layers[0].append(rn[0])
                shortest_path_list.append(
                    nx.single_source_shortest_path(G, source=rn[0])
                )
        else:
            node_layers[0].append(root_node[0][0])

            shortest_paths = nx.single_source_shortest_path(G, source=root_node[0][0])
            shortest_path_list = [shortest_paths]

        for spl in shortest_path_list:
            for key in spl.keys():
                path_length = len(spl[key])
                if len(node_layers) <= path_length:
                    node_layers.append([])

                node_layers[path_length].append(key)
        G_comp = nx.DiGraph()
        for i in range(len(node_layers)):
            G_comp.add_nodes_from(node_layers[i], layer=i)

        G_comp.add_edges_from(G.edges)
        layout = nx.multipartite_layout(
            G_comp, subset_key="layer", align="vertical", scale=1
        )

        for key in layout.keys():
            coords = layout[key]
            coords[0] += np.random.randn(1) * 0.05
            layout[key] = coords

        if hasattr(self, "print_locs"):
            layout = self.print_locs

        nodes = nx.draw_networkx_nodes(G, pos=layout, ax=ax)
        nodes.set_edgecolor("white")
        nodes.set_facecolor("white")
        labels = nx.draw_networkx_labels(G, pos=layout, ax=ax)
        edges = nx.draw_networkx_edges(G, pos=layout, ax=ax)

        # latex_graph = nx.to_latex(G, pos=nx.rescale_layout_dict(layout, scale=3))
        # latex_graph = nx.to_latex(G, pos=nx.rescale_layout(np.array(list({key:np.array(value) for key, value in layout.items()}.values()), dtype=float), scale=3))
        # latex_graph = nx.to_latex(G, pos={key: nx.rescale_layout(np.array(value, dtype=float), scale=3) for key, value in layout.items()})
        latex_graph = nx.to_latex(G, pos={key: np.array(value, dtype=float) * 2 for key, value in layout.items()})

    def step_system_state_function(self, G_dispatch, generation_available, step_index):

        if hasattr(self, "node_order"):
            node_order = self.node_order

        simulated_edges = list(self.G.edges)
        simulated_IO = {}
        for edge in simulated_edges:
            simulated_IO.update({edge: {"simulated": None}})

        for node in list(self.G.nodes):
            self.G.nodes[node].update({"wasted_output": np.zeros(4)})

        for edge in list(self.G.edges):
            self.G.edges[edge].update({"simulated": None})

        for node in node_order:

            if node == "generation":
                this_node_input = generation_available
                # self.G.nodes[node]["model"].set_output(generation_available)
                self.G.nodes[node]["ionode"].model.set_output(generation_available)

            sim_in_edges = self.G.in_edges(node)

            this_node_input = []

            for in_edge in sim_in_edges:
                # Check that upstream values have been simulated
                # edge_data = G_simulated.get_edge_data(in_edge[0], in_edge[1])["value"]
                edge_data = self.G.get_edge_data(in_edge[0], in_edge[1])["simulated"]
                # assert edge_data is not None, "edge data is none, needs to be run first"
                # assert not np.isnan(edge_data)

                this_node_input.append(edge_data)

            if len(this_node_input) > 0:
                this_node_input = np.stack(this_node_input)
            else:
                this_node_input = np.zeros((1, 4))

            node_dispatch_split = np.array(self.G.nodes[node]["dispatch_split"])
            node_dispatch_ctrl = np.array(self.G.nodes[node]["dispatch_ctrl"])

            node_output = self.G.nodes[node]["ionode"].step(
                this_node_input,
                node_dispatch_ctrl,
                node_dispatch_split,
                step_index,
            )

            sim_out_edges = self.G.out_edges(node)
            wasted_output = np.zeros_like(node_output)

            self.G.nodes[node].update({"wasted_output": wasted_output})
            for i, out_edge in enumerate(list(sim_out_edges)):
                simulated_IO[out_edge]["simulated"] = node_output[:, i]

            nx.set_edge_attributes(self.G, simulated_IO)

            # To check that the edge attributes are being updated
            # nx.get_edge_attributes(G_simulated, "value")

        return self.G

    def simulate(self, dispatcher: GreenheartDispatch, hopp_results):
        # Get generation signals

        self.dispatcher = dispatcher

        self.setup_ctrl_sysid()


        gen_profiles = {}

        for hopp_tech in hopp_results["annual_energies"]["technologies"].keys():
            if hopp_tech in ["pv", "wind"]:
                gen_profiles.update(
                    {
                        hopp_tech: hopp_results["annual_energies"]["technologies"][
                            hopp_tech
                        ].generation_profile
                    }
                )

        # Is there an existing method to get the hybrid generation profile from the HOPP_result

        hybrid_profile = np.array(gen_profiles["pv"]) + np.array(gen_profiles["wind"])

        self.hybrid_profile = hybrid_profile

        # Loop for everything downstream of generation
        error_feedback = False
        t0 = time.time()

        for i in range(len(hybrid_profile)):

            if i > self.stop_index:
                print("stopping at realtime simulator stop index")
                break

            if i < (len(hybrid_profile) - dispatcher.controller.horizon):

                forecast = hybrid_profile[i : i + dispatcher.controller.horizon]
            else:
                # forecast = np.ones(dispatcher.controller.horizon) * hybrid_profile[i]
                forecast = np.concatenate([hybrid_profile[i:], hybrid_profile[-1] * np.ones(dispatcher.controller.horizon - (len(hybrid_profile) - i))])

            # x0 = np.zeros(len([node for node in self.node_order if (node in ["battery", "hydrogen_storage", "thermal_energy_storage"])]))
            x0 = []
            for state_node in ["battery", "thermal_energy_storage", "hydrogen_storage"]:
                if state_node in self.G:
                    if state_node == "battery":
                        if self.G.nodes["battery"]["ionode"].model.use_hopp_outputs:
                            if i == 0:
                                state = self.G.nodes._nodes['battery']['ionode'].model.hopp_battery.config.initial_SOC / 100 * self.G.nodes._nodes['battery']['ionode'].model.hopp_battery.config.system_capacity_kwh
                            else:

                                if self.G.nodes["battery"]["ionode"].model.hopp_battery.outputs.SOC[i-1] < self.G.nodes["battery"]["ionode"].model.hopp_battery._system_model.ParamsCell.minimum_SOC:
                                    # If the bound is violated by only a little, just use the lower bound instead
                                    assert (self.G.nodes["battery"]["ionode"].model.hopp_battery._system_model.ParamsCell.minimum_SOC - self.G.nodes["battery"]["ionode"].model.hopp_battery.outputs.SOC[i-1]) <= 1e-3, "Battery minimum SOC violated"
                                    state = self.G.nodes["battery"]["ionode"].model.hopp_battery._system_model.ParamsCell.minimum_SOC / 100 * self.G.nodes._nodes['battery']['ionode'].model.hopp_battery.config.system_capacity_kwh
                                else:
                                    state = self.G.nodes["battery"]["ionode"].model.hopp_battery.outputs.SOC[i-1] / 100 * self.G.nodes._nodes['battery']['ionode'].model.hopp_battery.config.system_capacity_kwh
                        else:
                            state = self.G.nodes["battery"]["ionode"].model.storage_state
                    elif state_node == "hydrogen_storage":
                        state = self.G.nodes["hydrogen_storage"][
                            "ionode"
                        ].model.storage_state
                    elif state_node == "thermal_energy_storage":
                        state = (
                            self.G.nodes["thermal_energy_storage"][
                                "ionode"
                            ].model._SOC()
                            * self.G.nodes["thermal_energy_storage"][
                                "ionode"
                            ].model.H_capacity_kWh
                        )
                    x0.append(state)
            x0 = np.array(x0)

            # if i > 0:
            #     mpc_state = self.dispatcher.controller.x_store[-1][:, i - self.dispatcher.controller.step_index_store[-1]]
            #     if np.any((np.abs(x0 - mpc_state) / (0.5 * (x0 + mpc_state))) > 0.1):
            #         pass


            self.G = dispatcher.step(
                self.G,
                hybrid_profile[i],
                forecast=forecast,
                x_measured=x0,
                step_index=i,
            )


            


            if "generation" in self.G.nodes:
                if "grid_purchase" in self.G.nodes["generation"]:
                    grid_power = self.G.nodes["generation"]["grid_purchase"]
                else:
                    grid_power = 0

            self.G = self.step_system_state_function(
                self.G, hybrid_profile[i] + grid_power, i
            )




            if (not (i % 5)) and (self.verbose):
                print(
                    f"\r {(i / len(hybrid_profile)* 100) :.1f} % , {time.time() - t0:.2f} seconds, {(1 - i/len(hybrid_profile)) * (time.time() - t0) / ((i+1) / len(hybrid_profile)) :.2f} seconds longer \t\t\t\t",
                    end="",
                )

            self.record_states(i, self.G, grid_power)
            # Check on the error

            self.save_ctrl_for_sysid(step_index=i)

            if self.dispatcher.use_MPC:
                sim_edges_full = self.system_states[:, i, :]
                sim_edges = np.zeros(sim_edges_full.shape[0])
                for j in range(sim_edges_full.shape[0]):
                    sim_edges[j] = np.sum(sim_edges_full[j, 0:-1])

                if not (i % self.dispatcher.update_period) or (i == 0):
                    mpc_edges = self.dispatcher.controller.ysp_store[
                        np.where(
                            np.array(self.dispatcher.controller.step_index_store) == i
                        )[0][0]
                    ]
                else:
                    mpc_edges = self.dispatcher.controller.ysp_store[
                        np.where(
                            np.array(self.dispatcher.controller.step_index_store)
                            == self.dispatcher.previous_update
                        )[0][0]
                    ]
                mpc_edges = mpc_edges[0:-1, i - self.dispatcher.previous_update]

                mpc_edges_permuted = np.zeros(mpc_edges.shape)

                for k in range(len(self.edge_order)):
                    index = [
                        ind
                        for ind in range(len(self.dispatcher.controller.pco_label))
                        if (
                            (
                                self.dispatcher.controller.pco_label[ind].split(" ")[2]
                                == self.edge_order[k][0]
                            )
                            and (
                                self.dispatcher.controller.pco_label[ind]
                                .split(" ")[-1]
                                .split(")")[0]
                                == self.edge_order[k][1]
                            )
                        )
                    ]
                    mpc_edges_permuted[k] = mpc_edges[index]

                mpc_edges = mpc_edges_permuted

                edge_error = sim_edges - mpc_edges
                # edge_percent_error = (sim_edges - mpc_edges) / (0.5 * (sim_edges + mpc_edges))

                error_dict = {
                    str(self.edge_order[k]): edge_error[k]
                    for k in range(len(self.edge_order))
                }

                sim_u_curtail = {
                    node: self.G.nodes[node]["ionode"].u_curtail_store[i]
                    for node in self.node_order
                }
                sim_u_passthrough = {
                    node: self.G.nodes[node]["ionode"].u_passthrough_store[i]
                    for node in self.node_order
                }

                self.record_error(
                    error_dict,
                    sim_u_curtail,
                    sim_u_passthrough,
                    sim_edges,
                    mpc_edges,
                    step_index=i,
                )
            
            y_steel = self.G.nodes["steel"]["ionode"].model.steel_store_tonne[i]
            ref = self.config.greenheart_config["realtime_simulation"]["dispatch"]["mpc"]["reference"]

            if (i >= 1) and (np.abs(y_steel - ref) / ref >= 0.5):

                pass

        # print("")
        # self.input_error = {}
        # for node in self.node_order:
        #     self.input_error.update(
        #         {
        #             node: {
        #                 "curtail": self.G.nodes[node]["ionode"].u_curtail_store,
        #                 "passthrough": self.G.nodes[node]["ionode"].u_passthrough_store,
        #             }
        #         }
        #     )
        for node in self.G.nodes:
            if hasattr(self.G.nodes[node]["ionode"].model, "consolidate_sim_outcome"):
                self.G.nodes[node]["ionode"].model.consolidate_sim_outcome()
        print("")

        self.models = {
            node: self.G.nodes[node]["ionode"].model for node in self.node_order
        }

        []

    def setup_ctrl_sysid(self):

        # Make a bunch of dicts with 8760 length

        self.sysid = {}

        for label in self.dispatcher.controller.n_label:
            self.sysid.update({label: np.zeros(8760)})


        for label in self.dispatcher.controller.mct_label:
            self.sysid.update({label: np.zeros(8760)})


        for label in self.dispatcher.controller.msp_label:
            self.sysid.update({label: np.zeros(8760)})


        for label in self.dispatcher.controller.oex_label:
            self.sysid.update({label: np.zeros(8760)})
        
        
        for label in self.dispatcher.controller.pco_label:
            self.sysid.update({label: np.zeros(8760)})
        
        
        for label in self.dispatcher.controller.pex_label:
            self.sysid.update({label: np.zeros(8760)})
        
        # pze
        # pgt - do these need to be recorded here too?

        pass 


    def save_ctrl_for_sysid(self, step_index=None):

        # Think about doing this with the un-simplified system model rather than the simplified one with coupling

        # save x uct usp dex inputs
        # save yco yex yze ygt outputs


        # states
        for label in self.dispatcher.controller.n_label:
            node_name = label.split(" ")[2]
            if node_name == "battery":
                state = self.G.nodes["battery"]["ionode"].model.storage_state
            elif node_name == "thermal_energy_storage":
                state =  self.G.nodes["thermal_energy_storage"]["ionode"].model._SOC() * self.G.nodes["thermal_energy_storage"]["ionode"].model.H_capacity_kWh
            elif node_name == "hydrogen_storage":
                state = self.G.nodes["hydrogen_storage"]["ionode"].model.storage_state

            self.sysid[label][step_index] = state



        # control inputs
                
        for label in self.dispatcher.controller.mct_label:
            node_name = label.split(" ")[2]
            uct_index = int(label.split(" ")[1])
            self.sysid[label][step_index] = self.G.nodes[node_name]["dispatch_ctrl"][uct_index]


        # splitting inputs
                
        for label in self.dispatcher.controller.msp_label:
            source_node = label.split(" ")[2]
            sink_node = label[label.find("(")+1:label.find(")")].split(" ")[1]

            usp_index = int(label.split(" ")[1])

            self.sysid[label][step_index] = self.G.nodes[source_node]["dispatch_split"][usp_index]


        # disturbance
                
        # Output stuff
                
        # yco
        
        for label in self.dispatcher.controller.pco_label:
            source_node = label.split(" ")[2]
            sink_node = label[label.find("(")+1:label.find(")")].split(" ")[1]

            output_domain = self.G.nodes[source_node]["ionode"].output_list[0:-1]
            output_index = np.where(output_domain)[0]

            for i, edge in enumerate(self.edge_order):    
                if (edge[0] == source_node) and (edge[1] == sink_node):
                    # This is probably the right case then
                    self.sysid[label][step_index] = self.system_states[i, step_index, output_index]




        # yex
        yex_label = self.dispatcher.controller.pex_label[0] 
        yex_node = yex_label.split(" ")[-1]
        self.sysid[yex_label][step_index] = self.G.nodes[yex_node]["ionode"].model.steel_store_tonne[step_index]


        # dex
        dex_label = self.dispatcher.controller.oex_label[0]
        self.sysid[dex_label][step_index] = self.hybrid_profile[step_index] + self.G.nodes["generation"]["grid_purchase"]
        # yze = 0
        # ygt = 0

        pass

    def setup_record_keeping(self):
        duration = 8760

        # index_dict = {}
        # for i, edge in enumerate(self.G.edges):
        #     index_dict.update({edge: i})

        index_dict = {tuple(self.edge_order[i]): i for i in range(len(self.edge_order))}

        self.index_dict = index_dict
        self.system_states = np.zeros((len(self.G.edges), duration, 4))
        # self.node_waste = np.zeros((len(self.G.nodes), duration, 4))

        self.grid_power_store = np.zeros((1, duration))

        self.sim_edge_store = np.zeros((len(self.G.edges), duration))
        self.mpc_edge_store = np.zeros((len(self.G.edges), duration))

        self.edge_error_store = np.zeros((len(self.G.edges), duration))
        # passthrough_dims = [self.G.nodes[node]["ionode"].u_passthrough_store.shape[1] for node in self.node_order]
        # curtail_dims = [self.G.nodes[node]["ionode"].u_curtail_store.shape[1] for node in self.node_order]
        # self.input_error = {self.node_order[i]: {curtail}}

        self.curtail_store = {}
        self.passthrough_store = {}
        for node in self.node_order:
            n_inputs = np.sum(self.G.nodes[node]["ionode"].input_list)
            if self.G.nodes[node]["ionode"].inputs["T"]:
                n_inputs -= 1

            self.curtail_store.update({node: np.zeros((duration, n_inputs))})
            self.passthrough_store.update({node: np.zeros((duration, n_inputs))})

        # self.curtail_store = np.zeros((len(self.G.nodes), duration))
        # self.passthrough_store = np.zeros((len(self.G.nodes), duration))

    def record_error(
        self,
        error_dict,
        node_curtail=None,
        node_passthrough=None,
        sim_edges=None,
        mpc_edges=None,
        step_index=None,
    ):

        self.edge_error_store[:, step_index] = list(error_dict.values())

        self.sim_edge_store[:, step_index] = sim_edges
        self.mpc_edge_store[:, step_index] = mpc_edges

        for node in self.node_order:
            self.curtail_store[node][step_index, :] = node_curtail[node]
            self.passthrough_store[node][step_index, :] = node_passthrough[node]

        # self.curtail_store[:, step_index] = np.concatenate(list(node_curtail.values()))
        # self.passthrough_store[:, step_index] = np.concatenate(list(node_passthrough.values()))

    def record_states(self, time_step, simulated_IO, grid_power):
        self.grid_power_store[0, time_step] = grid_power

        values = nx.get_edge_attributes(simulated_IO, "simulated")
        for key in values.keys():
            self.system_states[self.index_dict[key], time_step, :] = values[key]

        # for i, node in enumerate(list(simulated_IO.nodes)):
        #     # self.node_waste[i, time_step, :] = simulated_IO.nodes[node]["wasted_output"]
        #     # TODO come back to this it is messy
        #     self.node_waste[i, time_step, :] = np.sum(
        #         simulated_IO.nodes[node]["wasted_output"], axis=1
        #     )

        []

    def _setup_generation_node(self):
        inputs = {"power": False, "Qdot": False, "mdot": False, "T": False}
        outputs = {"power": True, "Qdot": False, "mdot": False, "T": False}

        out_degree = self.G.out_degree["generation"]

        component_dict = {
            "generation": {
                "model": StandinNode(out_degree),
                "model_inputs": inputs,
                "model_outputs": outputs,
            }
        }
        return component_dict

    def _setup_curtail_node(self):
        inputs = {"power": True, "Qdot": False, "mdot": False, "T": False}
        outputs = {"power": False, "Qdot": False, "mdot": False, "T": False}
        component_dict = {
            "curtail": {
                "model": StandinNode(),
                "model_inputs": inputs,
                "model_outputs": outputs,
            }
        }

        return component_dict

    def _setup_output_node(self):
        inputs = {"power": True, "Qdot": True, "mdot": True, "T": True}
        outputs = {"power": False, "Qdot": False, "mdot": False, "T": False}
        component_dict = {
            "output": {
                "model": StandinNode(),
                "model_inputs": inputs,
                "model_outputs": outputs,
            }
        }

        return component_dict

    def _setup_battery_node(self):
        inputs = {"power": True, "Qdot": False, "mdot": False, "T": False}
        outputs = {"power": True, "Qdot": False, "mdot": False, "T": False}
        component_dict = {
            "battery": {
                "model": Battery(config=self.config, battery_config=self.config.hopp_config["technologies"]["battery"], hopp_interface=self.hi),
                "model_inputs": inputs,
                "model_outputs": outputs,
            }
        }

        return component_dict

    def _setup_electrolyzer_node(self):

        electrical_generation_timeseries = np.zeros(8760)
        electrolyzer_size_mw = self.config.greenheart_config["electrolyzer"]["rating"]
        n_pem_clusters = int(
            ceildiv(
                electrolyzer_size_mw,
                self.config.greenheart_config["electrolyzer"]["cluster_rating_MW"],
            )
        )
        electrolyzer_capex_kw = self.config.greenheart_config["electrolyzer"][
            "electrolyzer_capex"
        ]
        electrolyzer_direct_cost_kw = electrolyzer_capex_kw
        useful_life = self.config.greenheart_config["project_parameters"][
            "project_lifetime"
        ]

        pem_param_dict = {
            "eol_eff_percent_loss": self.config.greenheart_config["electrolyzer"][
                "eol_eff_percent_loss"
            ],
            "uptime_hours_until_eol": self.config.greenheart_config["electrolyzer"][
                "uptime_hours_until_eol"
            ],
            "include_degradation_penalty": self.config.greenheart_config[
                "electrolyzer"
            ]["include_degradation_penalty"],
            "turndown_ratio": self.config.greenheart_config["electrolyzer"][
                "turndown_ratio"
            ],
        }
        user_defined_pem_param_dictionary = pem_param_dict
        verbose = False

        electrolyzer_model = run_PEM_clusters_step(
            electrical_generation_timeseries,
            electrolyzer_size_mw,
            n_pem_clusters,
            electrolyzer_direct_cost_kw,
            useful_life,
            user_defined_pem_param_dictionary,
            verbose=verbose,
            step_model=self.config.realtime_simulation,
        )

        inputs = {"power": True, "Qdot": False, "mdot": False, "T": False}
        outputs = {"power": False, "Qdot": False, "mdot": True, "T": True}
        component_dict = {
            "electrolyzer": {
                "model": electrolyzer_model,
                "model_inputs": inputs,
                "model_outputs": outputs,
            }
        }
        return component_dict

    def _setup_hydrogen_storage_node(self):
        inputs = {"power": False, "Qdot": False, "mdot": True, "T": True}
        outputs = {"power": False, "Qdot": False, "mdot": True, "T": True}
        component_dict = {
            "hydrogen_storage": {
                "model": HydrogenStorage(),
                "model_inputs": inputs,
                "model_outputs": outputs,
            }
        }

        return component_dict

    def _setup_thermal_energy_storage_node(self):
        inputs = {"power": True, "Qdot": False, "mdot": False, "T": False}
        outputs = {"power": False, "Qdot": True, "mdot": False, "T": False}
        component_dict = {
            "thermal_energy_storage": {
                "model": ThermalEnergyStorage(),
                "model_inputs": inputs,
                "model_outputs": outputs,
            }
        }

        return component_dict

    def _setup_heat_exchanger_node(self):
        inputs = {"power": True, "Qdot": True, "mdot": True, "T": True}
        outputs = {"power": False, "Qdot": False, "mdot": True, "T": True}
        component_dict = {
            "heat_exchanger": {
                "model": HeatExchanger(),
                "model_inputs": inputs,
                "model_outputs": outputs,
            }
        }

        return component_dict

    def _setup_steel_node(self):

        config = self.config.greenheart_config["steel"]["costs"]["feedstocks"]

        inputs = {"power": True, "Qdot": False, "mdot": True, "T": True}
        outputs = {"power": True, "Qdot": False, "mdot": True, "T": True}
        component_dict = {
            "steel": {
                "model": SteelModel(self.config.greenheart_config),
                "model_inputs": inputs,
                "model_outputs": outputs,
            }
        }

        return component_dict

    def get_component(self, component_name):
        return self.G.nodes[component_name]["ionode"].model

    def unpack_component(self, component_name, make_plot=False):
        # get inputs outputs, component model

        # Do graph stuff

        # input edges
        # output edges

        rt_node = self.G.nodes[component_name]["ionode"]

        passthrough = rt_node.u_passthrough_store
        input_curtail = rt_node.u_curtail_store
        split_curtail = rt_node.u_curtail_split_store[
            :, np.where(rt_node.output_list)[0]
        ].T[0, :]

        if rt_node.inputs["T"]:

            disturbance = rt_node.disturbance_store[:, 0:-1]
        else:
            disturbance = rt_node.disturbance_store

        in_edges = self.G.in_edges(component_name)

        in_edge_index = []
        for in_edge in in_edges:
            for i, edge in enumerate(list(self.G.edges)):
                if in_edge == edge:
                    in_edge_index.append(i)

        # in_edge_data = self.system_states[
        #     in_edge_index, :, np.where(rt_node.input_list)[0]
        # ]
        in_edge_data = self.system_states[:, :, np.where(rt_node.input_list)[0]][
            in_edge_index, :, :
        ]

        out_edges = self.G.out_edges(component_name)

        out_edge_index = []
        for out_edge in out_edges:
            for i, edge in enumerate(list(self.G.edges)):
                if out_edge == edge:
                    out_edge_index.append(i)
        out_edge_data = self.system_states[:, :, np.where(rt_node.output_list)[0]][
            out_edge_index, :, :
        ]
        # out_edge_data = self.system_states[
        #     out_edge_index, :, np.where(rt_node.output_list)[0]
        # ]

        # In these get the states and LLC signals

        # comp_local_data should have u, x, y, control?

        if component_name == "generation":
            comp_local_data = self.unpack_generation()
        elif component_name == "battery":
            comp_local_data = self.unpack_battery()
        elif component_name == "thermal_energy_storage":
            comp_local_data = self.unpack_thermal_energy_storage()
        else:
            comp_local_data = {"uct": [], "x": [], "y": []}

        component_data = {
            "passthrough": passthrough,
            "input_curtail": input_curtail,
            "split_curtail": split_curtail,
            "disturbance": disturbance,
            "in_edges": list(in_edges),
            "in_data": in_edge_data,
            "out_edges": list(out_edges),
            "out_data": out_edge_data,
            "uct": comp_local_data["uct"],
            "x": comp_local_data["x"],
            "y": comp_local_data["y"],
        }

        if make_plot:
            self.plot_component(component_name, component_data)

        return component_data

    def plot_component(self, component_name, component_data):

        # TODO make this flexible for components with multi-domain inputs

        fig, ax = plt.subplots(
            4, 1, sharex="all", layout="constrained", figsize=(10, 4)
        )

        fig.suptitle(component_name)

        ax[0].plot(component_data["disturbance"], label="d")
        ax[0].fill_between(
            np.arange(0, len(component_data["disturbance"]), 1),
            component_data["disturbance"][:, 0],
            component_data["disturbance"][:, 0] - component_data["input_curtail"][:, 0],
            label="input curtail",
        )
        ax[0].fill_between(
            np.arange(0, len(component_data["disturbance"]), 1),
            component_data["disturbance"][:, 0] - component_data["input_curtail"][:, 0],
            component_data["disturbance"][:, 0]
            - component_data["input_curtail"][:, 0]
            - component_data["passthrough"][:, 0],
            label="passthrough",
        )
        ax[0].fill_between(
            np.arange(0, len(component_data["disturbance"]), 1),
            component_data["disturbance"][:, 0]
            - component_data["input_curtail"][:, 0]
            - component_data["passthrough"][:, 0],
            np.zeros(len(component_data["disturbance"])),
            label="model input",
        )

        # ax[0] plot incoming edges

        ax[0].legend()

        ax[1].plot(component_data["x"], label="x")

        # ax[2].plot(component_data["y"], label="y")

        for i in range(len(component_data["out_edges"])):
            ax[2].plot(
                component_data["out_data"][i, :],
                label=str(component_data["out_edges"][i]),
            )
        ax[2].fill_between(
            np.arange(0, len(component_data["disturbance"]), 1),
            np.zeros(len(component_data["disturbance"])),
            component_data["split_curtail"],
            label="split curtail",
        )

        ax[2].legend()

        if self.stop_index < 8760:
            ax[0].set_xlim([0, self.stop_index])

        pass

    def plot_component_with_control(self, component_name, component_data):
        pass

    def unpack_thermal_energy_storage(self):

        generation_local_data = {
            "uct": [],
            "x": self.G.nodes["thermal_energy_storage"]["ionode"].model.M_hot_store,
            "y": [],
        }

        return generation_local_data

    def unpack_battery(self):

        generation_local_data = {
            "uct": self.G.nodes["battery"]["ionode"].model.store_charge_power,
            "x": self.G.nodes["battery"]["ionode"].model.store_storage_state,
            "y": [],
        }

        return generation_local_data

    def unpack_generation(self):
        uct = []
        x = []
        y = []

        generation_local_data = {"uct": uct, "x": x, "y": y}
        return generation_local_data

    def plot_edges(self):
        fig, ax = plt.subplots(
            len(self.edge_order), 1, sharex="col", layout="constrained"
        )
        colors = ["black", "red", "blue"]
        labels = ["power", "heat", "hydrogen"]
        for i in range(len(self.edge_order)):
            for j in range(3):
                ax[i].fill_between(
                    np.arange(0, self.system_states.shape[1], 1),
                    np.zeros(self.system_states.shape[1]),
                    self.system_states[i, :, j],
                    step="post",
                    color=colors[j],
                    label=labels[j],
                )
                ylabel = f"{self.edge_order[i][0][0:4]} to {self.edge_order[i][1][0:4]}"
                ax[i].set_ylabel(ylabel)

            ax[i].set_ylim([0, ax[i].get_ylim()[1]])
            ax[i].legend()

        if self.stop_index < 8760:
            ax[0].set_xlim([0, self.stop_index])

        fig.align_ylabels()

    def plot_nodes(self, data="edges", figsize = (15, 8), fname=None, save=False):
        fig, ax = plt.subplots(
            len(self.node_order),
            1,
            sharex="all",
            sharey="all",
            layout="constrained",
            figsize=figsize,
        )

        ax = ax[:, None]

        if data == "edges":

            normalized_states = np.zeros(self.system_states.shape)
            # for i in range(self.system_states.shape[2]):
            #     normalized_states[:, :, i] = np.nan_to_num(
            #         self.system_states[:, :, i] / np.max(self.system_states[:, :, i])
            #     )

            normalized_states[:, :, 0] = np.nan_to_num(self.system_states[:, :, 0] / np.max(self.system_states[:, :, [0, 1]]))
            normalized_states[:, :, 1] = np.nan_to_num(self.system_states[:, :, 1] / np.max(self.system_states[:, :, [0, 1]]))
            normalized_states[:, :, 2] = np.nan_to_num(self.system_states[:, :, 2] / np.max(self.system_states[:, :, 2]))

            plot_states = normalized_states
        elif data == "error":
            normalized_states = np.zeros(self.system_states.shape)

            normalized_states[:, :, 0] = np.nan_to_num(self.system_states[:, :, 0] / np.max(self.system_states[:, :, [0, 1]]))
            normalized_states[:, :, 1] = np.nan_to_num(self.system_states[:, :, 1] / np.max(self.system_states[:, :, [0, 1]]))
            normalized_states[:, :, 2] = np.nan_to_num(self.system_states[:, :, 2] / np.max(self.system_states[:, :, 2]))

            plot_states = normalized_states            

        colors = ["black", "red", "blue"]
        cmaps = ["Greys", "Reds", "Blues"]
        edgecolors = ["orange", "yellow", "cyan", "magenta", "green"]
        labels = ["P", "Q", "H2"]
        for i in range(len(self.node_order)):
            node = self.node_order[i]

            ylabel = "\n".join(node.split("_"))

            ax[i, 0].set_ylabel(ylabel)

            incoming = [
                self.edge_order[k][0]
                for k in range(len(self.edge_order))
                if node == self.edge_order[k][1]
            ]
            outgoing = [
                self.edge_order[k][1]
                for k in range(len(self.edge_order))
                if node == self.edge_order[k][0]
            ]
            in_index = np.array(
                [
                    k
                    for k in range(len(self.edge_order))
                    if node == self.edge_order[k][1]
                ]
            )
            out_index = np.array(
                [
                    k
                    for k in range(len(self.edge_order))
                    if node == self.edge_order[k][0]
                ]
            )
            for j in range(3):
                # for j in [0]

                j_ax = 0

                n_fills = len(in_index) + len(out_index)
                cmap_level = 0.25

                start = np.zeros(plot_states.shape[1])

                if (node == "generation") and (j == 0):
                    # ax[i, j_ax].step(
                    #     np.arange(0, self.system_states.shape[1], 1),
                    #     -self.hybrid_profile / np.max(self.system_states[:, :, 0]),
                    #     color="black",
                    #     linewidth=1,
                    #     where="post",
                    #     label="Hybrid gen.",
                    # )
                    ax[i, j_ax].fill_between(
                        np.arange(0, self.system_states.shape[1], 1),
                        np.zeros(len(self.hybrid_profile)),
                        self.hybrid_profile / np.max(self.system_states[:, :, 0]),
                        color=mpl.colormaps[cmaps[j]](0.8),
                        linewidth=0,
                        step="post",
                        label="Hybrid gen.",
                    )
                    curtail = self.G.nodes[node]["ionode"].u_curtail_store / np.max(
                        self.system_states[:, :, 0]
                    )
                    grid = (
                        self.grid_power_store / np.max(self.system_states[:, :, 0])
                    )[0, :]
                    ax[i, j_ax].fill_between(
                        np.arange(0, plot_states.shape[1], 1),
                        self.hybrid_profile / np.max(self.system_states[:, :, 0]),
                        self.hybrid_profile / np.max(self.system_states[:, :, 0])
                        + grid,
                        step="post",
                        alpha=1,
                        linewidth=0,
                        label="grid",
                        color="darkviolet",
                        # color=mpl.colormaps[cmaps[j]](cmap_level + 0.35),
                    )

                    stop = -curtail[:, 0]
                    ax[i, j_ax].fill_between(
                        np.arange(0, plot_states.shape[1], 1),
                        start,
                        start + stop,
                        step="post",
                        alpha=1,
                        linewidth=0,
                        label=f"curtail",
                        color=mpl.colormaps[cmaps[j]](cmap_level - 0.15),
                    )
                    start += stop

                if node == "battery":

                    axt = ax[i, 0].twinx()

                    axt.plot(
                        self.G.nodes["battery"]["ionode"].model.store_storage_state
                        / self.G.nodes["battery"]["ionode"].model.max_capacity_kWh,
                        color="black",
                        linewidth=1,
                        label="BES SOC",
                    )
                    axt.set_ylim([0, 1])
                    axt.set_yticks([])
                    bes_soc_legend = axt.get_legend_handles_labels()

                if node == "hydrogen_storage":

                    axt = ax[i, 0].twinx()

                    axt.plot(
                        self.G.nodes["hydrogen_storage"][
                            "ionode"
                        ].model.store_storage_state
                        / self.G.nodes["hydrogen_storage"][
                            "ionode"
                        ].model.max_capacity_kg,
                        color="blue",
                        linewidth=1,
                        label="H2S SOC",
                    )
                    axt.set_ylim([0, 1])
                    axt.set_yticks([])
                    h2s_soc_legend = axt.get_legend_handles_labels()

                if node == "thermal_energy_storage":

                    axt = ax[i, 0].twinx()

                    axt.plot(
                        self.G.nodes["thermal_energy_storage"][
                            "ionode"
                        ].model.SOC_store,
                        color="red",
                        linewidth=1,
                        label="TES SOC",
                    )
                    axt.set_ylim([0, 1])
                    axt.set_yticks([])
                    tes_soc_legend = axt.get_legend_handles_labels()

                for k in range(len(in_index)):
                    stop = plot_states[in_index[k], :, j]
                    if np.sum(stop) != 0:
                        ax[i, j_ax].fill_between(
                            np.arange(0, plot_states.shape[1], 1),
                            start,
                            start + stop,
                            step="post",
                            alpha=1,
                            linewidth=0,
                            label=f"from {incoming[k][0:4]}",
                            color=mpl.colormaps[cmaps[j]](cmap_level),
                        )

                    cmap_level += 0.15
                    start += stop

                start = np.zeros(plot_states.shape[1])
                if node == "generation":
                    start = -curtail[:, 0]
                for k in range(len(out_index)):
                    stop = -plot_states[out_index[k], :, j]
                    if np.sum(stop) != 0:
                        ax[i, j_ax].fill_between(
                            np.arange(0, plot_states.shape[1], 1),
                            start,
                            start + stop,
                            step="post",
                            alpha=1,
                            linewidth=0,
                            label=f"to {outgoing[k][0:4]}",
                            color=mpl.colormaps[cmaps[j]](cmap_level),
                        )

                    start += stop
                    cmap_level += 0.15

                if (node == "steel") and (j == 2):
                    steel_output = self.G.nodes["steel"][
                        "ionode"
                    ].model.steel_store_tonne
                    steel_output = steel_output / np.max(steel_output)
                    ax[i, j_ax].fill_between(
                        np.arange(0, plot_states.shape[1], 1),
                        np.zeros(len(steel_output)),
                        -steel_output,
                        step="post",
                        alpha=1,
                        linewidth=0,
                        label=f"Steel output",
                        color="darkgreen",
                    )

                []

        # ax[0, 0].set_title("Power")
        # ax[0, 1].set_title("Heat")
        # ax[0, 2].set_title("Hydrogen")

        if self.stop_index / self.dispatcher.update_period <= 50:
            xtick_locs = np.arange(0, self.stop_index, self.dispatcher.update_period)
            ax[-1, 0].set_xticks(xtick_locs, xtick_locs, rotation=90)
            # ax[-1, j].tick_params(axis="x", direction="in")
        else:
            update_locs = np.arange(0, self.stop_index, self.dispatcher.update_period)
            xtick_locs = np.arange(
                0,
                self.stop_index,
                int(
                    np.round(self.stop_index / 50 / self.dispatcher.update_period)
                    * self.dispatcher.update_period
                ),
            )
            ax[-1, 0].set_xticks(xtick_locs, xtick_locs, rotation=90)
            []

        legend_kwargs = {
            "fontsize": 10,
            "borderpad": 0.2,
            "handlelength": 1.2,
            "handleheight": 0.6,
            "handletextpad": 0.25,
            # "loc": "upper right",
            "loc": "upper center",
            "ncols": 8,
        }

        for i in range(ax.shape[0]):
            for j in range(ax.shape[1]):
                # ax[i,j].set_ylim([0, ax[i,j].get_ylim()[1]])
                ax[i, j].set_ylim(
                    [
                        -np.max(np.abs(ax[i, j].get_ylim())),
                        np.max(np.abs(ax[i, j].get_ylim())),
                    ]
                )
                if ax.shape[1] == 1:
                    ax[i, j].set_yticks([])
                ax[i, j].axhline(0, linewidth=0.5, color="black", alpha=0.5, zorder=0.5)
                ax[i, j].tick_params(axis="x", direction="in")

                if self.node_order[i] == "battery":
                    handles, labels = ax[i, j].get_legend_handles_labels()
                    handles.append(bes_soc_legend[0][0])
                    labels.append(bes_soc_legend[1][0])
                    ax[i, j].legend(handles, labels, **legend_kwargs)

                elif self.node_order[i] == "hydrogen_storage":
                    handles, labels = ax[i, j].get_legend_handles_labels()
                    handles.append(h2s_soc_legend[0][0])
                    labels.append(h2s_soc_legend[1][0])
                    ax[i, j].legend(handles, labels, **legend_kwargs)
                elif self.node_order[i] == "thermal_energy_storage":
                    handles, labels = ax[i, j].get_legend_handles_labels()
                    handles.append(tes_soc_legend[0][0])
                    labels.append(tes_soc_legend[1][0])
                    ax[i, j].legend(handles, labels, **legend_kwargs)
                else:
                    handles, labels = ax[i, j].get_legend_handles_labels()

                ax[i, j].legend(handles, labels, **legend_kwargs)

        if self.stop_index <= 8760:
            ax[0, 0].set_xlim([0, self.stop_index])
        else:
            ax[0, 0].set_xlim([0, 8760])

        if save:

            fig.savefig(f"{fname}{'_8760.pdf'}", format="pdf")
            # ax[0,0].set_xlim([2800, 3150])
            # fig.savefig(f"{fname}{'_zoom.pdf'}", format="pdf")

        []


    def reformat_stored(self):
        pass

    def plot_edge_error(self):
        pass

    def plot_input_error(self):
        # plot curtail or passthrough
        pass


class RealTimeSimulationOutput:
    def __init__(self):
        pass


class StandinNode:
    def __init__(self, out_degree=1):
        self.output = 0
        self.out_degree = 1
        # self.out_degree = out_degree
        self.create_control_model()

    def create_control_model(self):
        n = 0
        m = 0
        p = 1
        # m = self.out_degree
        # p = self.out_degree
        o = 1

        A = np.zeros((n, n))
        B = np.zeros((n, m))
        C = np.zeros((p, n))
        D = np.zeros((p, m))
        E = np.zeros((n, o))
        # F = np.zeros((p, o))
        F = np.array([[1]])

        bounds_dict = {
            "u_lb": np.array([0] * m),
            "u_ub": np.array([None] * m),
            "x_lb": np.array([]),
            "x_ub": np.array([]),
            "y_lb": np.array([0] * p),
            "y_ub": np.array([None] * p),
        }

        self.control_model = ControlModel(
            A=A, B=B, C=C, D=D, E=E, F=F, bounds=bounds_dict
        )

        self.control_model.set_disturbance_domain([1, 0, 0])
        self.control_model.set_output_domain([1, 0, 0])

    def set_output(self, output):
        self.output = output

    def step(self, input, dispatch=None, step_index=None):

        u_passthrough = 0
        if dispatch >= -1:
            dispatch = np.max([0.0, dispatch[0]])
        assert dispatch >= 0
        u_curtail = dispatch
        output = self.output - u_curtail
        return output, u_passthrough, u_curtail


class Forecaster:
    def __init__(self):
        pass

    
