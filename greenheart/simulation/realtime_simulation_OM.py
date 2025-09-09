import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from hopp.utilities import load_yaml

import openmdao.api as om


import logging
from logging import handlers

import time


from greenheart.simulation.realtime_node import (
    Node,
    setup_generation_node,
    setup_battery_node,
    setup_electrolyzer_node,
    setup_hydrogen_storage_node,
    setup_thermal_energy_storage_node,
    setup_heat_exchanger_node,
    setup_steel_node,
)


class SubSystem(om.ExplicitComponent):
    name: str


    def initialize(self):
        self.options.declare("ionode")

    def setup(self):
        self.add_input("step_index")

    def unpack_inputs(self, inputs):

        # Separate disturbance inputs
        # Separate u_control
        # Separate u_split

        disturbance_inputs = []
        control_inputs = []
        splitting_inputs = []

        for inp in self.list_inputs():
            if inp[0].startswith("d"):

                origin = "_".join(inp[0].split("_")[1:])

                if origin in ["generation", "battery", "external"]:
                    disturbance_inputs.append(np.array([inp[1]["val"][0], 0, 0, 0]))

                elif origin in ["electrolyzer", "hydrogen_storage"]:
                    disturbance_inputs.append(np.array([0, 0, inp[1]["val"][0], 20]))

                elif origin in ["heat_exchanger"]:
                    disturbance_inputs.append(np.array([0, 0, inp[1]["val"][0], 900]))

                elif origin in ["thermal_energy_storage"]:
                    disturbance_inputs.append(np.array([0, inp[1]["val"][0],0, 0]))



            elif inp[0].startswith("uc"):
                control_inputs.append(inp[1]["val"][0])
            elif inp[0].startswith("us"):
                splitting_inputs.append(inp[1]["val"][0])


        disturbance_inputs = np.stack(disturbance_inputs)
        control_inputs = np.array(control_inputs)
        splitting_inputs = np.array(splitting_inputs)
        if len( splitting_inputs) == 0:
            splitting_inputs = np.array([1])

        return disturbance_inputs, control_inputs, splitting_inputs

    def pack_outputs(self, node_outputs, outputs):


        for i, outp in enumerate( outputs):
            
            if self.name in ["generation", "battery", "external"]:
                outputs[outp] = node_outputs[0, i]


            elif self.name in ["electrolyzer", "hydrogen_storage"]:
                outputs[outp] = node_outputs[2, i]


            elif self.name in ["heat_exchanger"]:
                outputs[outp] = node_outputs[2, i]


            elif self.name in ["thermal_energy_storage"]:
                outputs[outp] = node_outputs[1, i]

        return outputs

    def compute(self, inputs, outputs):
        step_index = int(inputs["step_index"][0])


        if self.name == "generation":
            self.options["ionode"].model.output = inputs["d_external"]
            # self.options["ionode"].model.output = np.array([inputs["d_external"][0], 0, 0, 0])

        args = (*self.unpack_inputs(inputs), step_index)

        node_outputs = self.options["ionode"].step(*args)

        outputs = self.pack_outputs(node_outputs, outputs)




class System(om.Group):
    pass


class RealTimeSimulation:
    def __init__(
        self,
        config,
        hopp_interface=None,
        hybrid_profile=None,
    ):

        self.greenheart_config = config
        self.config = config.greenheart_config["realtime_simulation"]
        self.subsystem_config = load_yaml(self.config["component_config"])
        self.graph_config = self.config["system"]["system_graph_config"]

        self.hopp_interface = hopp_interface
        self.hybrid_profile = hybrid_profile

        self.start_index = self.config.get("start_index", 0)
        self.stop_index = self.config.get("stop_index", 8760)

        self.verbose = True
        self.save_ctrl = False
        self.use_networkx_model = False

        self.hybrid_profile = []

        self._setup_bookkeeping()
        self._setup_simulation_model()

    def _setup_logging(self):
        pass

    def _setup_simulation_model(self):

        # Read config
        # Collect system models

        self._setup_networkx_graph()
        self._setup_openmdao_system()

    def _setup_networkx_graph(self):

        graph_config = load_yaml(self.graph_config)

        # Graph edges, in order
        edges = graph_config["network"]
        self.edge_order = edges
        edge_nodes = list(set([node for edge in edges for node in edge]))

        # Graph nodes, in order
        self.node_order = graph_config["traversal_order"]
        assert all(
            [en in self.node_order for en in edge_nodes]
        ), f"Subsystem listed in edges that wasn't listed in nodes: {[en for en in edge_nodes if en not in self.node_order]}"

        if "print_locs" in graph_config.keys():
            self.print_locs = graph_config["print_locs"]

        # Initialize networkx graph object, add nodes and edges
        G = nx.DiGraph()
        G.add_nodes_from(self.node_order)
        G.add_edges_from(self.edge_order)
        self.G = G

        # Instantiate the individual steppable models of each technology
        setup_method_map = dict(
            generation=setup_generation_node,
            battery=setup_battery_node,
            electrolyzer=setup_electrolyzer_node,
            hydrogen_storage=setup_hydrogen_storage_node,
            thermal_energy_storage=setup_thermal_energy_storage_node,
            heat_exchanger=setup_heat_exchanger_node,
            steel=setup_steel_node,
        )

        for node in self.G.nodes:
            subsystem_dict = setup_method_map[node](
                self.G,
                self.greenheart_config,
                self.hopp_interface,
                self.subsystem_config,
            )[node]
            ionode = Node(
                name=node,
                model=subsystem_dict["model"],
                expected_inputs=subsystem_dict["model_inputs"],
                expected_outputs=subsystem_dict["model_outputs"],
                splitting_node=(True if self.G.out_degree[node] > 1 else False),
                in_degree=self.G.in_degree[node],
                out_degree=self.G.out_degree[node],
            )

            is_source = graph_config["source_node"] == node
            is_sink = graph_config["sink_node"] == node
            self.G.nodes[node].update(
                {"ionode": ionode, "is_source": is_source, "is_sink": is_sink}
            )

    def _setup_openmdao_system(self):


        controllable_nodes = ["battery", "thermal_energy_storage", "hydrogen_storage"]



        promoted_inputs = {}
        promoted_outputs = {}

        self.system = System()

        connections = []

        for node in self.node_order:
            om_subsys = SubSystem(ionode=self.G.nodes[node]["ionode"])

            om_subsys.name = node


            promoted_inputs.update({node:["step_index", "step_index"]})
            promoted_outputs.update({node:[]})

            for edge in self.edge_order:
                # If node is at the destination end
                if edge[1] == node:
                    om_subsys.add_input(f"d_{edge[0]}")

                # If node is at the origin end
                if edge[0] == node:
                    om_subsys.add_output(f"y_{edge[1]}")
                
                    if self.G.nodes[node]["ionode"].splitting_node:

                        om_subsys.add_input(f"us_{edge[1]}")
                        promoted_inputs[node].append((f"us_{edge[1]}", f"us_{edge[0]}_{edge[1]}"))

                connection = (f"{edge[0]}.y_{edge[1]}", f"{edge[1]}.d_{edge[0]}")
                if connection not in connections:
                    connections.append(connection)

            
            if node == "generation":
                om_subsys.add_input(f"uc_curtail")
                promoted_inputs[node].append((f"uc_curtail", "uc_curtail"))

            if node in controllable_nodes:
                om_subsys.add_input(f"uc_{node}_charge")
                om_subsys.add_input(f"uc_{node}_discharge")
                promoted_inputs[node].append((f"uc_{node}_charge", f"uc_{node}_charge"))
                promoted_inputs[node].append((f"uc_{node}_discharge", f"uc_{node}_discharge"))



            if self.G.nodes[node]["is_source"]:
                om_subsys.add_input("d_external")
                promoted_inputs[node].append(("d_external", "d_external"))
                # promoted_inputs.update({node:[("d_external", "d_external")]})

            if self.G.nodes[node]["is_sink"]:
                om_subsys.add_output("y_external")
                promoted_outputs[node].append(("y_external", "y_external"))
                # promoted_outputs.update({node:["y_external", "y_external"]})

            self.system.add_subsystem(
                name=node,
                subsys=om_subsys,
                promotes_inputs=promoted_inputs[node],
                promotes_outputs=promoted_outputs[node],
            )

        # for edge in self.edge_order:
        for connection in connections:
            self.system.connect(connection[0], connection[1])

        self.problem = om.Problem()
        self.model = self.problem.model
        self.model.add_subsystem("System", self.system)

        self.problem.setup()
        self.problem.final_setup()

        self.om_inputs = promoted_inputs
        self.om_outputs = promoted_outputs

    def _setup_bookkeeping(self):
        pass

    def record_error(self):
        pass

    def record_states(self):
        pass

    def step_system(self):
        if self.use_networkx_model:
            self._step_networkx_system()
        else:
            self._step_openmdao_system()

    def _step_networkx_system(self):
        pass

    def _step_openmdao_system(self):

        for inp in self.om_inputs:
            self.problem.set_val(inp)

        self.problem.run_model()

        pass

    def simulate(self):
        # ==============================================================================
        #                                Simulation loop
        # ==============================================================================

        time_steps = np.arange(self.start_index, self.stop_index, 1)

        for step_index in time_steps:

            forecast = self.forecaster.get_forecast()
            state_measurement = self._get_state_measurement()

            # Update control signals in the graph
            self.G = self.dispatcher.step(
                self.G,
                self.hybrid_profile[step_index],
                forecast=forecast,
                x_measured=state_measurement,
                step_index=step_index,
            )

            if "generation" in self.G.nodes:
                if "grid_purchase" in self.G.nodes["generation"]:
                    grid_power = self.G.nodes["generation"]["grid_purchase"]
                else:
                    grid_power = 0

            # Step the simulation models
            self.G = self.step_system(
                self.G, self.hybrid_profile[step_index] + grid_power, step_index
            )

            self.record_states()

            if self.save_ctrl:
                self._save_ctrl_for_sysid()

            if self.save_error:
                self.record_error()

        self.consolidate_simulation_outcome()

    def consolidate_simulation_outcome(self):
        self.models = {
            node: self.G.nodes[node]["ionode"].model for node in self.node_order
        }

        for node in self.G.nodes:
            if hasattr(self.G.nodes[node]["ionode"].model, "consolidate_sim_outcome"):
                self.G.nodes[node]["ionode"].model.consolidate_sim_outcome()

    def _get_state_measurement(self):
        pass

    def unpack_subsystem(self):
        # Was unpack_component
        pass

    def plot_subsystem(self):
        # was plot_componenet
        pass

    def plot_edges(self):
        pass

    def plot_nodes(self):
        pass

    def plot_system_graph(self):
        pass

    def _setup_ctrl_sysid(self):
        pass

    def _save_ctrl_for_sysid(self):
        pass
