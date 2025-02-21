import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

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

        self.config = config
        self.hi = hopp_interface

        self.stop_index = 8760 * 30

        self.setup_simulation_model(config, hopp_interface)
        self.setup_record_keeping()

    def setup_simulation_model(self, config, hopp_interface):
        possible_technologies = [
            "battery",
            "TES",
            "H2 storage",
            "electrolyzer",
            "steel",
        ]

        # check the config files to find which technologies are actually in the config
        hopp_techs = config.hopp_config["technologies"].keys()

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

        graph_config_fpath = config.greenheart_config["system"]["system_graph_config"]
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
            # ionode = IONode(
            #     name=node,
            #     model=RT_techs[node]["model"],
            #     expected_inputs=RT_techs[node]["model_inputs"],
            #     expected_outputs=RT_techs[node]["model_outputs"],
            #     splitting_node=self.G.nodes[node]["split"],
            #     in_degree=self.G.in_degree[node],
            #     out_degree=self.G.out_degree[node],
            # )

            # Add an assertion that there is one source and one sink
            is_source = False
            is_sink = False
            if graph_config["source_node"] == node:
                is_source = True
            if graph_config["sink_node"] == node:
                is_sink = True

            self.G.nodes[node].update({"ionode": ionode, "is_source":is_source, "is_sink":is_sink})

        # nx.draw_networkx(G, with_labels=True)

        # [edge for edge in nx.edge_bfs(G, "generation")]
        # models = [G.nodes[node]["model"] for node in G.nodes]

        # self.G = G
        # self.G = G

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

        nodes = nx.draw_networkx_nodes(G, pos=layout, ax=ax)
        nodes.set_edgecolor("white")
        nodes.set_facecolor("white")
        labels = nx.draw_networkx_labels(G, pos=layout, ax=ax)
        edges = nx.draw_networkx_edges(G, pos=layout, ax=ax)

        latex_graph = nx.to_latex(G, pos=nx.rescale_layout_dict(layout, scale=3))

    def step_system_state_function(self, G_dispatch, generation_available, step_index):

        if hasattr(self, "node_order"):
            node_order = self.node_order
        else:

            # Hard code for now but need to come back to make this general
            node_order = [
                "generation",
                "curtail",
                "electrolyzer",
                "hydrogen_storage",
                "steel",
            ]

            # Hard code for heat example
            # node_order = [
            #     "generation",
            #     "curtail",
            #     "battery",
            #     "electrolyzer",
            #     "hydrogen_storage",
            #     "joule_heater",
            #     "thermal_energy_storage",
            #     "heat_exchanger",
            #     "output",
            # ]

            node_order = ["generation", "curtail", "electrolyzer", "heat_exchanger"]

        # G_simulated = self.G.copy()

        # for node in list(G_simulated.nodes):
        #     G_simulated.nodes[node].update({"wasted_output": 0})

        simulated_edges = list(self.G.edges)
        simulated_IO = {}
        for edge in simulated_edges:
            simulated_IO.update({edge: {"simulated": None}})

        # nx.set_edge_attributes(G_simulated, simulated_IO)

        for node in list(self.G.nodes):
            self.G.nodes[node].update({"wasted_output": np.zeros(4)})

        for edge in list(self.G.edges):
            self.G.edges[edge].update({"simulated": None})

        for node in node_order:

            # dispatch_IO_edges = G_dispatch.edges(node)
            # simulated_IO_edges = G_simulated.edges(node)

            if node == "generation":
                this_node_input = generation_available
                # self.G.nodes[node]["model"].set_output(generation_available)
                self.G.nodes[node]["ionode"].model.set_output(generation_available)

            # Get the input to the node from the simulated graph
            # These are only the edges coming into node node
            # sim_in_edges = G_simulated.in_edges(node)
            sim_in_edges = self.G.in_edges(node)

            # this_node_input = 0
            this_node_input = []

            for in_edge in sim_in_edges:
                # Check that upstream values have been simulated
                # edge_data = G_simulated.get_edge_data(in_edge[0], in_edge[1])["value"]
                edge_data = self.G.get_edge_data(in_edge[0], in_edge[1])["simulated"]
                # assert edge_data is not None, "edge data is none, needs to be run first"
                # assert not np.isnan(edge_data)

                # this_node_input += edge_data
                this_node_input.append(edge_data)

            if len(this_node_input) > 0:
                this_node_input = np.stack(this_node_input)
                # this_node_input = np.concatenate(this_node_input, axis=1)
            else:
                this_node_input = np.zeros((1, 4))

            # Gather inputs from edge list - All inputs to node should already be in simulated_IO
            # Gather outputs from dispatch list

            # dispatch_out_edges = G_dispatch.out_edges(node)
            dispatch_out_edges = self.G.out_edges(node)
            # dispatch_values = nx.get_edge_attributes(G_dispatch, "value")
            dispatch_values = nx.get_edge_attributes(self.G, "dispatch")
            node_dispatch_values = {
                key: dispatch_values[key] for key in list(dispatch_out_edges)
            }

            # dispatch_out_total = np.sum(list(node_dispatch_values.values()))
            # This may cause a problem for heat mass mixing
            if len(node_dispatch_values) > 0:
                dispatch_out_total = np.sum(
                    np.stack(list(node_dispatch_values.values())), axis=0
                )
            else:
                dispatch_out_total = np.zeros(4)

            # node_output = self.G.nodes[node]["ionode"].step(
            #     this_node_input, dispatch_out_total, step_index
            # )

            node_dispatch_split = np.array(self.G.nodes[node]["dispatch_split"])
            node_dispatch_ctrl = np.array(self.G.nodes[node]["dispatch_ctrl"])

            node_output = self.G.nodes[node]["ionode"].step(
                this_node_input,
                node_dispatch_ctrl,
                node_dispatch_split,
                step_index,
            )

            # node_output = self.G.nodes[node]["ionode"].step(
            #     this_node_input,
            #     dispatch_out_total,
            #     step_index,
            #     node_dispatch_split,
            #     node_dispatch_ctrl,
            # )

            sim_out_edges = self.G.out_edges(node)

            wasted_output = np.zeros_like(node_output)
            # # for j in range(len(node_output)):
            # wasted_output = np.where(
            #     node_output > dispatch_out_total, (node_output - dispatch_out_total), 0
            # )
            # node_output -= wasted_output

            # G_simulated.nodes[node].update({"wasted_output": wasted_output})
            self.G.nodes[node].update({"wasted_output": wasted_output})

            # for out_edge in list(sim_out_edges):
            #     scaled_output = node_output * np.nan_to_num(
            #         node_dispatch_values[out_edge] / dispatch_out_total
            #     )
            #     simulated_IO[out_edge]["simulated"] = scaled_output
            for i, out_edge in enumerate(list(sim_out_edges)):
                simulated_IO[out_edge]["simulated"] = node_output[:, i]

            nx.set_edge_attributes(self.G, simulated_IO)

            # To check that the edge attributes are being updated
            # nx.get_edge_attributes(G_simulated, "value")

        return self.G

    def simulate(self, dispatcher: GreenheartDispatch, hopp_results):
        # Get generation signals

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

        # G_dispatch = self.G.copy()

        t0 = time.time()

        for i in range(len(hybrid_profile)):

            if i > self.stop_index:
                print("stopping at realtime simulator stop index")
                break

            if i < (len(hybrid_profile) - dispatcher.controller.horizon):

                forecast = hybrid_profile[i: i + dispatcher.controller.horizon]
            else:
                forecast = np.ones(dispatcher.controller.horizon) * hybrid_profile[i]

            # TODO improve this: 
            x0 = np.zeros( 2)
            x0[0] = self.G.nodes["battery"]["ionode"].model.storage_state
            x0[1] = self.G.nodes["hydrogen_storage"]["ionode"].model.storage_state

            self.G = dispatcher.step(self.G, hybrid_profile[i], forecast=forecast, x_measured = x0,  step_index=i)
            self.G = self.step_system_state_function(self.G, hybrid_profile[i], i)

            if not (i % 5):
                print(f"\r {(i / len(hybrid_profile)* 100) :.1f} % , {time.time() - t0:.2f} seconds, {(1 - i/len(hybrid_profile)) * (time.time() - t0) / ((i+1) / len(hybrid_profile)) :.2f} seconds longer \t\t\t\t", end="")

            # TODO update to use self.G not G_error
            if error_feedback:

                safety_count = 0
                max_error = 1e6
                allowable_error = 1e1

                while (max_error > allowable_error) and (safety_count < 10):
                    safety_count += 1

                    max_error = 0
                    G_error = G_simulated.copy()
                    for edge in list(G_error.edges):
                        error = (
                            G_dispatch.edges[edge]["value"]
                            - G_simulated.edges[edge]["value"]
                        )

                        max_error = np.max([max_error, np.max(np.abs(error))])

                        G_error.edges[edge].update({"error": error})

                    # nx.get_edge_attributes(G_error, "error")
                    G_dispatch = dispatcher.step(G_dispatch, hybrid_profile[i], G_error)
                    G_simulated = self.step_system_state_function(
                        G_dispatch, hybrid_profile[i], i
                    )

                    []

            self.record_states(i, self.G)
        print("")
        for node in self.G.nodes:
            if hasattr(self.G.nodes[node]["ionode"].model, "consolidate_sim_outcome"):
                self.G.nodes[node]["ionode"].model.consolidate_sim_outcome()

    def setup_record_keeping(self):
        duration = 8760

        index_dict = {}
        for i, edge in enumerate(self.G.edges):
            index_dict.update({edge: i})

        self.index_dict = index_dict
        self.system_states = np.zeros((len(self.G.edges), duration, 4))
        self.node_waste = np.zeros((len(self.G.nodes), duration, 4))

    def record_states(self, time_step, simulated_IO):

        values = nx.get_edge_attributes(simulated_IO, "simulated")
        for key in values.keys():
            self.system_states[self.index_dict[key], time_step, :] = values[key]

        for i, node in enumerate(list(simulated_IO.nodes)):
            # self.node_waste[i, time_step, :] = simulated_IO.nodes[node]["wasted_output"]
            # TODO come back to this it is messy
            self.node_waste[i, time_step, :] = np.sum(
                simulated_IO.nodes[node]["wasted_output"], axis=1
            )

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
                "model": Battery(self.config.hopp_config["technologies"]["battery"]),
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

        # if "use_step_model" in self.config.greenheart_config["electrolyzer"].keys():
        #     step_model = self.config.greenheart_config["electrolyzer"]["use_step_model"]
        # else:
        #     step_model = False

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

    # def _setup_joule_heater_node(self):
    #     inputs =
    #     outputs =
    #     component_dict =

    #     return component_dict
    # if GH_tech == "joule_heater":
    #     # RT_techs.update({"joule_heater": JouleHeater()})
    #     RT_techs.update(
    #         {
    #             "joule_heater": {
    #                 "model": JouleHeater(),
    #                 "model_inputs": {
    #                     "power": True,
    #                     "Qdot": False,
    #                     "mdot": False,
    #                     "T": False,
    #                 },
    #                 "model_outputs": {
    #                     "power": False,
    #                     "Qdot": True,
    #                     "mdot": False,
    #                     "T": False,
    #                 },
    #             }
    #         }
    #     )

    # def _setup__node(self):
    #     inputs =
    #     outputs =
    #     component_dict =

    #     return component_dict

    def get_component(self, component_name):
        return self.G.nodes[component_name]["ionode"].model


class RealTimeSimulationOutput:
    def __init__(self):
        pass


class StandinNode:
    def __init__(self, out_degree = 1):
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


        self.control_model = ControlModel(A=A, B=B, C=C, D=D, E=E, F=F, bounds=bounds_dict)


    def set_output(self, output):
        self.output = output

    def step(self, input, dispatch=None, step_index=None):
        u_passthrough = 0
        u_curtail = 0
        return self.output, u_passthrough, u_curtail


class IONode:
    # Splitting and throw away should happen here in IONODE

    def __init__(
        self,
        name,
        model,
        expected_inputs,
        expected_outputs,
        splitting_node,
        in_degree=None,
        out_degree=None,
    ):
        self.inputs = expected_inputs
        self.input_list = [
            self.inputs["power"],
            self.inputs["Qdot"],
            self.inputs["mdot"],
            self.inputs["T"],
        ]

        self.outputs = expected_outputs
        self.input_list = [
            self.outputs["power"],
            self.outputs["Qdot"],
            self.outputs["mdot"],
            self.outputs["T"],
        ]

        self.name = name
        self.model = model
        self.splitting_node = splitting_node
        if self.name == "generation":
            self.in_degree = 1
        else:
            self.in_degree = in_degree

        if (self.name == "output") or (self.name == "steel"):
            self.out_degree = 1
        else:
            self.out_degree = out_degree

        if out_degree == 0:
            self.out_degree = 1

        # NOTE Dont forget this
        # if hasattr(self.model, "control_model") and (self.out_degree > 1):
        #     self.model.control_model.make_splitting_node(self.out_degree)

        # self.make_fake_state_space()
            
        

    def step(
        self,
        graph_input,
        graph_dispatch,
        step_index,
        node_dispatch_split=None,
        node_dispatch_ctrl=None,
    ):

        # model_dispatch = self.consolidate_dispatch(graph_dispatch)

        model_input = self.consolidate_inputs(graph_input)
        model_dispatch_ctrl = self.model_dispatch_ctrl(model_input, node_dispatch_ctrl)
        model_output, u_passthrough, u_curtail = self.model.step(model_input, model_dispatch_ctrl, step_index)

        # TODO send u_passthrough downstream

        if self.name == "electrolyzer":
            model_output = [model_output, 80]
        elif self.name == "hydrogen_storage":
            model_output = [model_output, 20]
        # elif self.name == "heat_exchanger":
        #     model_output = (model_output, 900)

        graph_output = self.consolidate_output(model_output, node_dispatch_split)

        return graph_output

    def model_dispatch_ctrl(self, model_input, node_dispatch_ctrl=None):
        if node_dispatch_ctrl is not None:
            # model_dispatch_ctrl = node_dispatch_ctrl * model_input

            if isinstance(node_dispatch_ctrl, np.ndarray):
                if len(node_dispatch_ctrl) == 1:
                    model_dispatch_ctrl = node_dispatch_ctrl[0]
                else:
                    model_dispatch_ctrl = node_dispatch_ctrl
            else:
                model_dispatch_ctrl = node_dispatch_ctrl
        else:
            model_dispatch_ctrl = None

        return model_dispatch_ctrl

    def consolidate_inputs(self, graph_input):
        if graph_input.size == 0:
            return 0

        graph_input = np.atleast_2d(np.array(graph_input))
        Pin = np.sum(graph_input[:, 0])
        Qin = np.sum(graph_input[:, 1])
        mdotin = np.sum(graph_input[:, 2])
        Tin = np.nan_to_num(
            np.dot(graph_input[:, 2], graph_input[:, 3]) / np.sum(graph_input[:, 2])
        )

        inputs_dict = {"power": Pin, "Qdot": Qin, "mdot": mdotin, "T": Tin}

        model_input = []
        for key in self.inputs.keys():
            if self.inputs[key]:
                model_input.append(inputs_dict[key])

        if len(model_input) == 1:
            model_input = model_input[0]

        return model_input

    def consolidate_dispatch(self, graph_dispatch):
        if graph_dispatch.size == 0:
            return 0

        graph_dispatch = np.atleast_2d(np.array(graph_dispatch))
        Pin = np.sum(graph_dispatch[:, 0])
        Qin = np.sum(graph_dispatch[:, 1])
        mdotin = np.sum(graph_dispatch[:, 2])
        Tin = np.nan_to_num(
            np.dot(graph_dispatch[:, 2], graph_dispatch[:, 3])
            / np.sum(graph_dispatch[:, 2])
        )

        dispatch_dict = {"power": Pin, "Qdot": Qin, "mdot": mdotin, "T": Tin}

        model_dispatch = []
        for key in self.outputs.keys():
            if self.outputs[key]:
                model_dispatch.append(dispatch_dict[key])

        if len(model_dispatch) == 1:
            model_dispatch = model_dispatch[0]

        return model_dispatch

    def consolidate_output(self, model_output, node_dispatch_split=None):

        if len(node_dispatch_split) == 1:
            graph_output = np.zeros((4, 1))
            graph_output[np.where(self.input_list)[0], 0] = (
                node_dispatch_split * model_output
            )

        elif len(node_dispatch_split) >= 1:
            # if node_dispatch_split is not None:

            # TODO check that the dimensions of model_output are compatable with node_dispatch

            graph_output = np.zeros((4, len(node_dispatch_split)))
            graph_output[np.where(self.input_list)[0], :] = (
                np.array([model_output]).T @ np.array([node_dispatch_split])
                # node_dispatch_split * model_output
            )

            # if self.outputs["T"]:
            if self.name == "electrolyzer":
                graph_output[3,:] = model_output[1]
        

        else:

            graph_output = np.array([[0, 0, 0, 0]], dtype=float).T

            count = 0
            for i, key in enumerate(self.outputs.keys()):
                if self.outputs[key]:
                    if isinstance(model_output, float):
                        graph_output[i] = model_output
                    else:
                        graph_output[i] = model_output[count]
                    count += 1

        # TODO add check here so that the splitting doesn't output more than the model output

        return graph_output

    def make_fake_state_space(self):
        if self.splitting_node:
            m = self.out_degree
            n = 0
            o = self.in_degree
            p = self.out_degree
            
            self.A = np.zeros((n, n))
            self.B = np.zeros((n, m))
            self.E = np.zeros((n, o))
            self.C = np.zeros((p, n))
            self.D = np.zeros((p, m))
            self.D = np.eye(p)
            self.F = np.zeros((p, o))
        else:

            self.A = np.array([[0.39698364, -1.68707227], [0.06748289, 0.90310532]])
            self.B = np.array([[0.06748289], [0.00387579]])
            self.C = np.array([[0.0, 25.0]])
            self.D = np.array([[0.0]])
            self.E = np.array([[0.06748289], [0.00387579]])
            self.F = np.array([[0.0]])

            self.m = self.B.shape[1]
            self.n = self.A.shape[0]
            self.p = self.C.shape[0]
            self.o = self.E.shape[1]
