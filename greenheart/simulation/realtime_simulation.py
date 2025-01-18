import numpy as np
import networkx as nx

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
from greenheart.simulation.technologies.hydrogen.electrolysis.electrolyzer import (
    Electrolyzer,
)

from greenheart.simulation.technologies.hydrogen.electrolysis.run_PEM_master_STEP import (
    run_PEM_clusters_step,
)

from greenheart.simulation.technologies.hydrogen.h2_storage.hydrogen_storage import (
    HydrogenStorage,
)
from greenheart.simulation.technologies.steel.steel_dynamic_model import Steel

from greenheart.tools.eco.utilities import ceildiv


# Utilty imports
from hopp.utilities import load_yaml


# Simulation model for greenheart components


class RealTimeSimulation:
    def __init__(self, config, hopp_interface):

        self.config = config
        self.hi = hopp_interface

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
            "ammonia",
            "electrolyzer",
            "h2_storage",
            "joule_heater",
            "thermal_energy_storage",
            "heat_exchanger",
            "steel",
        ]

        GH_techs = []
        for key in config.greenheart_config.keys():
            if key in GH_tech_options:
                GH_techs.append(key)

        # Instantiate the individual steppable models of each technology

        RT_techs = {}

        for GH_tech in GH_techs:
            if GH_tech == "ammonia":
                pass
            if GH_tech == "electrolyzer":
                # ==================================
                # =                                =
                # =    electrolyzer configuraion   =
                # =                                =
                # ==================================

                electrical_generation_timeseries = np.zeros(8760)
                electrolyzer_size_mw = config.greenheart_config["electrolyzer"][
                    "rating"
                ]
                n_pem_clusters = int(
                    ceildiv(
                        electrolyzer_size_mw,
                        config.greenheart_config["electrolyzer"]["cluster_rating_MW"],
                    )
                )
                electrolyzer_capex_kw = config.greenheart_config["electrolyzer"][
                    "electrolyzer_capex"
                ]
                electrolyzer_direct_cost_kw = electrolyzer_capex_kw
                useful_life = config.greenheart_config["project_parameters"][
                    "project_lifetime"
                ]

                pem_param_dict = {
                    "eol_eff_percent_loss": config.greenheart_config["electrolyzer"][
                        "eol_eff_percent_loss"
                    ],
                    "uptime_hours_until_eol": config.greenheart_config["electrolyzer"][
                        "uptime_hours_until_eol"
                    ],
                    "include_degradation_penalty": config.greenheart_config[
                        "electrolyzer"
                    ]["include_degradation_penalty"],
                    "turndown_ratio": config.greenheart_config["electrolyzer"][
                        "turndown_ratio"
                    ],
                }
                user_defined_pem_param_dictionary = pem_param_dict
                verbose = False

                if "use_step_model" in config.greenheart_config["electrolyzer"].keys():
                    step_model = config.greenheart_config["electrolyzer"][
                        "use_step_model"
                    ]
                else:
                    step_model = False

                electrolyzer_model = run_PEM_clusters_step(
                    electrical_generation_timeseries,
                    electrolyzer_size_mw,
                    n_pem_clusters,
                    electrolyzer_direct_cost_kw,
                    useful_life,
                    user_defined_pem_param_dictionary,
                    verbose=verbose,
                    step_model=step_model,
                )

                # RT_techs.update({"electrolyzer": Electrolyzer()})
                RT_techs.update({"electrolyzer": electrolyzer_model})
            if GH_tech == "h2_storage":
                RT_techs.update({"hydrogen_storage": HydrogenStorage()})
            if GH_tech == "joule_heater":
                RT_techs.update({"joule_heater": JouleHeater()})
            if GH_tech == "thermal_energy_storage":
                RT_techs.update({"thermal_energy_storage": ThermalEnergyStorage()})
            if GH_tech == "heat_exchanger":
                RT_techs.update({"heat_exchanger": HeatExchanger()})
            if GH_tech == "steel":
                RT_techs.update({"steel": Steel()})

        # Build the connections with a graph network

        graph_config_fpath = config.greenheart_config["system"]["system_graph_config"]
        graph_config = load_yaml(graph_config_fpath)
        network_config = graph_config["network"]

        edges = network_config
        nodes = []
        for edge in network_config:
            for node in edge:
                if node not in nodes:
                    nodes.append(node)

        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)

        for node in G.nodes:
            model = None
            if (node == "generation") or (node == "curtail"):
                model = StandinNode()
            if node == "electrolyzer":
                model = RT_techs["electrolyzer"]
            if node == "hydrogen_storage":
                model = RT_techs["hydrogen_storage"]
            if node == "joule_heater":
                model = RT_techs["joule_heater"]
            if node == "thermal_energy_storage":
                model = RT_techs["thermal_energy_storage"]
            if node == "heat_exchanger":
                model = RT_techs["heat_exchanger"]
            if node == "steel":
                model = RT_techs["steel"]

            G.nodes[node].update({"model": model})

        # nx.draw_networkx(G, with_labels=True)

        # [edge for edge in nx.edge_bfs(G, "generation")]
        # models = [G.nodes[node]["model"] for node in G.nodes]

        self.system_graph = G

        self.technologies = RT_techs

        # Print or log the control input format

    def step_system_state_function(self, G_dispatch, generation_available, step_index):

        # TODO: will need a way to deal with cycles

        # NOTE: maybe there will be multiple graph objects with the same edges and nodes
        # 1. system model
        # 2. node status (has it been run yet)
        # 3. controller storage of inputs (stored in the edges)

        # Either here or in the controller
        # Pre-process and dont allow any cycles. Break the cycle at a logical place

        # Need to order the nodes in a logical way so that they can be computed as close to sequentially as possible.
        traversal = [edge for edge in nx.edge_bfs(self.system_graph, "generation")]
        upstream = [edge[0] for edge in traversal]
        downstream = [edge[1] for edge in traversal]

        # Hard code for now but need to come back to make this general
        node_order = [
            "generation",
            "curtail",
            "electrolyzer",
            "hydrogen_storage",
            "steel",
        ]

        # # Make standin dispatch_io and simulated io
        # G_dispatch = self.system_graph.copy()

        # dispatch_edges = list(G_dispatch.edges)
        # dispatch_IO = {}
        # for edge in dispatch_edges:
        #     dispatch_IO.update({edge: {"value": 2}})
        #     # dispatch_edge_dict.update({edge: [2]})

        # nx.set_edge_attributes(G_dispatch, dispatch_IO)
        G_simulated = self.system_graph.copy()

        for node in list(G_simulated.nodes):
            G_simulated.nodes[node].update({"wasted_output": 0})

        simulated_edges = list(G_simulated.edges)
        simulated_IO = {}
        for edge in simulated_edges:
            simulated_IO.update({edge: {"value": None}})

        nx.set_edge_attributes(G_simulated, simulated_IO)

        for node in node_order:

            dispatch_IO_edges = G_dispatch.edges(node)
            simulated_IO_edges = G_simulated.edges(node)

            if node == "generation":
                this_node_input = generation_available
                self.system_graph.nodes[node]["model"].set_output(generation_available)

            # Get the input to the node from the simulated graph
            # These are only the edges coming into node node
            sim_in_edges = G_simulated.in_edges(node)

            this_node_input = 0

            for in_edge in sim_in_edges:
                # Check that upstream values have been simulated
                edge_data = G_simulated.get_edge_data(in_edge[0], in_edge[1])["value"]
                assert edge_data is not None, "edge data is none, needs to be run first"
                assert not np.isnan(edge_data)

                this_node_input += edge_data

            # Gather inputs from edge list - All inputs to node should already be in simulated_IO
                

            # Gather outputs from dispatch list
                
            dispatch_out_edges = G_dispatch.out_edges(node)

            dispatch_values = nx.get_edge_attributes(G_dispatch, "value")
            node_dispatch_values = {
                key: dispatch_values[key] for key in list(dispatch_out_edges)
            }

            dispatch_out_total = np.sum(list(node_dispatch_values.values()))



            # Include the dispatch signal to the simulation model 
            # If it is an input-output model then the dispatch signal is ignored
            # If it is a controllable model then the low-level controller tries to match the dispatch signal of output
            # dispatch input can be the sum of all dispatch outputs? worry about tracking within the model, worry about splitting it in this forloop
            node_output = self.system_graph.nodes[node]["model"].step(
                this_node_input, dispatch_out_total, step_index
            )

            assert not np.isnan(node_output)

            # TODO if dispatch says less than node_output then throw some away
            # else if dispatch says more than node_output, then send it all downstream

            # distribute the node outputs into the simulated graph as close to the dispatch graph as possibl
            # put node_output in simulated_IO as close to dispatch_IO as possible

            sim_out_edges = G_simulated.out_edges(node)


            # If the node makes more output than the dispatcher planned for
            if node_output > dispatch_out_total:
                # TODO: dont forget to come back and track this
                wasted_output = node_output - dispatch_out_total
                node_output = dispatch_out_total - wasted_output
            else:
                wasted_output = 0

            G_simulated.nodes[node].update({"wasted_output": wasted_output})

            # record the node output in the simulated graph and update the edge data accordingly
            for out_edge in list(sim_out_edges):

                # Split the node output to downstream edges proportionally to the dispatch signal so under-production is shared evenly.
                scaled_output = node_output * (
                    node_dispatch_values[out_edge] / dispatch_out_total
                )

                if dispatch_out_total == 0:
                    scaled_output = 0

                simulated_IO[out_edge]["value"] = scaled_output

            nx.set_edge_attributes(G_simulated, simulated_IO)

            # To check that the edge attributes are being updated
            # nx.get_edge_attributes(G_simulated, "value")

            []

        # Traverse the system graph
        # Call the component input/output
        # check whether anything is violated
        # See how close you can get to the control decision

        return G_simulated

    def simulate(self, dispatcher, hopp_results):
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

        # Loop for everything downstream of generation

        G_dispatch = self.system_graph.copy()

        for i in range(len(hybrid_profile)):
            # dispatch_IO = dispatcher.step()

            G_dispatch = dispatcher.step(G_dispatch, hybrid_profile[i])
            simulated_IO = self.step_system_state_function(
                G_dispatch, hybrid_profile[i], i
            )

            self.record_states(i, simulated_IO)

        # TODO: For all components, run consolidate sim outcome

        for node in self.system_graph.nodes:
            if hasattr(
                self.system_graph.nodes[node]["model"], "consolidate_sim_outcome"
            ):
                self.system_graph.nodes[node]["model"].consolidate_sim_outcome()

        []

    def setup_record_keeping(self):
        duration = 8760

        index_dict = {}
        for i, edge in enumerate(self.system_graph.edges):
            index_dict.update({edge: i})

        self.index_dict = index_dict
        self.system_states = np.zeros((len(self.system_graph.edges), duration))
        self.node_waste = np.zeros((len(self.system_graph.nodes), duration))

        []

    def record_states(self, time_step, simulated_IO):

        values = nx.get_edge_attributes(simulated_IO, "value")
        for key in values.keys():
            self.system_states[self.index_dict[key], time_step] = values[key]

        for i, node in enumerate(list(simulated_IO.nodes)):
            self.node_waste[i, time_step] = simulated_IO.nodes[node]["wasted_output"]

        []

    # Flag in greenheart config for whether or not to construct real-time model
    # read in the greenheart config file
    # extract technical parameters from the config file
    # like electrolyzer sizing h2 compressor
    # ignore generation for now but maybe need to address later, or just say it can be curtailed

    """ From greenheart config
        electrolyzer sizing, rating, type
        h2 storage type
        h2 storage capacity maybe
    """

    """ From hopp config
        wind n_turbs, rating
        pv capacity
        battery rating and charge rate
    """

    """
    Run HOPP - Take the generation profile from HOPP

    Run real-time simulation 

    give the outputs of real-time simulation back to HOPP

    Use the other outputs of real-time simulation with other parts of the analysis in GreenHEART
    
    """


class RealTimeSimulationOutput:
    def __init__(self):
        pass


class StandinNode:
    def __init__(self):
        self.output = 0

    def set_output(self, output):
        self.output = output

    def step(self, input, dispatch=None, step_index=None):
        return self.output
