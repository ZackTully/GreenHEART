import numpy as np

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
from greenheart.simulation.technologies.dispatch.control_model import ControlModel


class Node:
    def __init__(
        self,
        name: str,
        model,
        expected_inputs: dict,
        expected_outputs: dict,
        in_degree: int,
        out_degree: int,
    ):
        """
        Node for each of the subsystems at the nodes of the RealTimeSimulator graph.
        This node class wraps around each of the subsystem models and translates the
        graph edges into subsystem inputs for each of the steppable models.

        Args:
            name (str): name
            model (): steppable model of the subsystem represented at this node.
            expected_inputs (dict): dict of input domains.
            expected_outputs (dict): dict of output domains.
            in_degree (int): Number of incoming edges.
            out_degree (int): Number of outgoing edges.
        """

        self.T_electrolyzer_output = 20  # [C]
        self.T_hydrogen_storage_output = 20  # [C]

        self.inputs = expected_inputs
        self.input_list = [
            self.inputs["power"],
            self.inputs["Qdot"],
            self.inputs["mdot"],
            self.inputs["T"],
        ]

        self.outputs = expected_outputs
        self.output_list = [
            self.outputs["power"],
            self.outputs["Qdot"],
            self.outputs["mdot"],
            self.outputs["T"],
        ]

        self.name = name
        self.model = model
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

        if self.inputs["T"]:
            self.u_curtail_store = np.zeros((8760, np.sum(self.input_list) - 1))
            self.u_passthrough_store = np.zeros((8760, np.sum(self.input_list) - 1))
            self.disturbance_store = np.zeros((8760, np.sum(self.input_list) - 1))
        else:
            self.u_curtail_store = np.zeros((8760, np.sum(self.input_list)))
            self.u_passthrough_store = np.zeros((8760, np.sum(self.input_list)))
            self.disturbance_store = np.zeros((8760, np.sum(self.input_list)))

        if self.name == "generation":
            self.u_curtail_store = np.zeros((8760, 1))
            self.disturbance_store = np.zeros((8760, 1))
            self.u_passthrough_store = np.zeros((8760, 1))
        # else:
        #     self.disturbance_store = np.zeros((8760, np.sum(self.input_list)))
        self.u_curtail_split_store = np.zeros((8760, 4))

        self.splitting_method = "fractional"  # fractional or absolute

    def __repr__(self):
        return self.name

    def consolidate_incoming_edges(self, incoming_edges: list) -> np.ndarray:
        """
        Take a list of one or more graph edges as inputs and consolidate into one edge

        Args:
            incoming_edges (list): list of 4-length numpy arrays representing the incoming edges.

        Returns:
            np.ndarray: consolidated inputs
        """

        node_input = np.sum(incoming_edges, axis=0)
        # Temperature = mass-weighted sum of incoming temperatures

        if np.any(incoming_edges[:, 2] != 0):
            node_input[3] = np.nan_to_num(
                np.dot(incoming_edges[:, 2], incoming_edges[:, 3])
                / np.sum(incoming_edges[:, 2])
            )

        return node_input

    def format_model_input(self, node_input: list | np.ndarray) -> list | float:
        """
        Take a single graph edge as input with power, heat, H2 mass and temperature
        reformat to only the arguments that the model needs

        Args:
            node_input (list | np.ndarray):

        Returns:
            list | float: _description_
        """

        model_input = node_input[np.where(self.input_list)]
        return model_input

    def step(
        self,
        incoming_edges: list,
        u_control: np.ndarray,
        u_split: np.ndarray,
        step_index: int,
    ) -> np.ndarray:
        """
        Step this node forward one timestep. Called by RealTimeSimulator

        Args:
            incoming_edges (list): list of 4-length numpy arrays representing the incoming edges.
            u_control (np.ndarry): Control inputs for the steppable subsystem model if any.
            u_split (np.ndarray): Splitting fraction for the subsystem model outputs to control how much of the output goes to each outgoing edge.
            step_index (int):

        Returns:
            np.ndarray: list of 4-length numpy arrays representing the outgoing edges
        """

        # Consolidate inputs and format the disturbance
        node_input = self.consolidate_incoming_edges(incoming_edges)
        model_disturbance = self.format_model_input(node_input)

        if self.name == "generation":
            model_disturbance = self.model.output

        # Step subsystem model
        y_model, u_passthrough, u_curtail = self.model.step(
            model_disturbance, u_control, step_index
        )

        # Format outputs
        model_output = self.format_model_output(y_model, u_passthrough)
        outgoing_edges, split_curtail = self.splitting(
            model_output, u_split, step_index
        )

        # Store input history
        self.store_disturbance(model_disturbance, step_index=step_index)
        self.store_passthrough(u_passthrough=u_passthrough, step_index=step_index)
        self.store_curtail(u_curtail, split_curtail, step_index)

        return outgoing_edges

    def store_disturbance(self, model_disturbance, step_index=0):
        if self.inputs["T"]:
            self.disturbance_store[step_index, :] = model_disturbance[0:-1]
        else:
            self.disturbance_store[step_index, :] = model_disturbance

    def store_passthrough(self, u_passthrough=None, step_index=0):
        self.u_passthrough_store[step_index, :] = u_passthrough

    def store_curtail(self, u_curtail=None, split_curtail=None, step_index=0):
        self.u_curtail_store[step_index, :] = u_curtail
        self.u_curtail_split_store[step_index, :] = split_curtail

    def format_model_output(
        self,
        y_model: np.ndarray | float,
        u_passthrough: np.ndarray | float,
    ) -> np.ndarray:
        """
        Format the model outputs translating from the steppable model output to the
        graph edge format expected by RealTimeSimulator.

        Args:
            y_model (np.ndarray | float): Steppable subsystem model output
            u_passthrough (np.ndarray | float): Subsystem model input that could not be used

        Returns:
            np.ndarray: 4-length array in the format of a graph edge.
        """
        output_passthrough = np.zeros((1, 4))

        if self.inputs["T"]:
            output_passthrough[0, np.where(self.input_list[:-1])] = u_passthrough
        else:
            output_passthrough[0, np.where(self.input_list)] = u_passthrough

        output_model = np.zeros((1, 4))
        output_model[0, np.where(self.output_list)] = y_model

        # Set the appropriate output temperature if relevant.
        if self.name == "electrolyzer":
            output_model[0, 3] = self.T_electrolyzer_output  # degree C
            output_passthrough[0, 3] = self.T_electrolyzer_output
        elif self.name == "hydrogen_storage":
            output_model[0, 3] = self.T_hydrogen_storage_output  # degree C
            output_passthrough[0, 3] = self.T_hydrogen_storage_output
        elif self.name == "heat_exchanger":
            output_model[0, 3] = self.model.Tout_desired
            output_passthrough[0, 3] = self.model.Tout_desired

        model_output = output_model + output_passthrough
        if np.any(model_output[:, 2] != 0):
            model_output[0, 3] = np.nan_to_num(
                (
                    output_model[0, 2] * output_model[0, 3]
                    + output_passthrough[0, 2] * output_passthrough[0, 3]
                )
                / (output_model[0, 2] + output_passthrough[0, 2])
            )
        return model_output

    def splitting(
        self, model_output: np.ndarray, u_split: np.ndarray, step_index: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """_summary_

        Args:
            model_output (np.ndarray): _description_
            u_split (np.ndarray): _description_
            step_index (int): _description_

        Returns:
            outgoing_edges (np.ndarray):
            split_curtail (np.ndarray):
        """
        if (u_split < 0).any():
            assert np.min(u_split) >= -1, f"{u_split = }"
            u_split = np.where(u_split < 0, 0.0, u_split)

        if self.splitting_method == "fractional":
            split = np.nan_to_num(u_split / np.sum(u_split))
        elif self.splitting_method == "absolute":
            split = u_split
        else:
            print("no splitting method")

        # Assuming u_split is fractional not absolute
        outgoing_edges = np.outer(split, model_output)
        outgoing_edges[:, 3] = model_output[0, 3]  # fix temperature

        split_curtail = model_output - np.sum(outgoing_edges, axis=0)

        assert np.all(outgoing_edges >= -1)

        return outgoing_edges.T, split_curtail


class StandinNode:
    """
    Standin node to be used as a subsystem model for those subsystems that don't have a
    subsystem model. This class is used by the generation node.
    """

    def __init__(self, out_degree=1, is_generation=False, is_electricity_output=False):

        self.is_generation = is_generation
        self.is_electricity_output = is_electricity_output

        self.output = 0
        self.out_degree = 1
        self.create_control_model()

    def create_control_model(self):
        """
        Control model state space equation.
        """

        if self.is_generation:

            n, m, p, o = 0, 0, 1, 1

            # n = 0
            # m = 0
            # p = 1
            # o = 1

            A = np.zeros((n, n))
            B = np.zeros((n, m))
            C = np.zeros((p, n))
            D = np.zeros((p, m))
            E = np.zeros((n, o))
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
        elif self.is_electricity_output:

            n, m, p, o = 0, 0, 1, 1

            A = np.zeros((n, n))
            B = np.zeros((n, m))
            C = np.zeros((p, n))
            D = np.zeros((p, m))
            E = np.zeros((n, o))
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
            # self.control_model.set_output_domain([1, 0, 0])
            self.control_model.set_disturbance_reshape([1, 0 ,0])


    def set_output(self, output):
        self.output = output

    def step(self, input, dispatch=None, step_index=None):
        """
        Step method for the generation node.

        Args:
            input (_type_): _description_
            dispatch (_type_, optional): _description_. Defaults to None.
            step_index (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        u_passthrough = 0
        if dispatch >= -1:
            dispatch = np.max([0.0, dispatch[0,0]])
        assert dispatch >= 0
        u_curtail = dispatch
        actual_curtail = min(dispatch, input[0])
        output = self.output - actual_curtail
        if output < 0:
            if output > -1:
                output = 0.0
            else:
                assert False, f"Generation node output was negative: {output:.6f} kW"

        return output, u_passthrough, u_curtail


def setup_generation_node(G, config, hi, component_config):
    inputs = {"power": True, "Qdot": False, "mdot": False, "T": False}
    outputs = {"power": True, "Qdot": False, "mdot": False, "T": False}

    out_degree = G.out_degree["generation"]

    component_dict = {
        "generation": {
            "model": StandinNode(out_degree, is_generation=True),
            "model_inputs": inputs,
            "model_outputs": outputs,
        }
    }
    return component_dict


def setup_electricity_output_node(G, config, hi, component_config):
    inputs = {"power": True, "Qdot": False, "mdot": False, "T": False}
    outputs = {"power": True, "Qdot": False, "mdot": False, "T": False}

    out_degree = G.out_degree["electricity_output"]

    component_dict = {
        "electricity_output": {
            "model": StandinNode(out_degree, is_electricity_output=True),
            "model_inputs": inputs,
            "model_outputs": outputs,
        }
    }
    return component_dict


def setup_battery_node(G, config, hi, component_config):
    inputs = {"power": True, "Qdot": False, "mdot": False, "T": False}
    outputs = {"power": True, "Qdot": False, "mdot": False, "T": False}
    component_dict = {
        "battery": {
            "model": Battery(
                config=config,
                battery_config=config.hopp_config["technologies"]["battery"],
                hopp_interface=hi,
            ),
            "model_inputs": inputs,
            "model_outputs": outputs,
        }
    }
    return component_dict


def setup_electrolyzer_node(G, config, hi, component_config):
    electrical_generation_timeseries = np.zeros(8760)
    electrolyzer_size_mw = config.greenheart_config["electrolyzer"]["rating"]
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
    useful_life = config.greenheart_config["project_parameters"]["project_lifetime"]

    pem_param_dict = {
        "eol_eff_percent_loss": config.greenheart_config["electrolyzer"][
            "eol_eff_percent_loss"
        ],
        "uptime_hours_until_eol": config.greenheart_config["electrolyzer"][
            "uptime_hours_until_eol"
        ],
        "include_degradation_penalty": config.greenheart_config["electrolyzer"][
            "include_degradation_penalty"
        ],
        "turndown_ratio": config.greenheart_config["electrolyzer"]["turndown_ratio"],
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
        step_model=config.realtime_simulation,
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


def setup_hydrogen_storage_node(G, config, hi, component_config):
    inputs = {"power": False, "Qdot": False, "mdot": True, "T": True}
    outputs = {"power": False, "Qdot": False, "mdot": True, "T": True}
    component_dict = {
        "hydrogen_storage": {
            "model": HydrogenStorage(component_config["hydrogen_storage"]),
            "model_inputs": inputs,
            "model_outputs": outputs,
        }
    }

    return component_dict


def setup_thermal_energy_storage_node(G, config, hi, component_config):
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


def setup_heat_exchanger_node(G, config, hi, component_config):
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


def setup_steel_node(G, config, hi, component_config):
    inputs = {"power": True, "Qdot": False, "mdot": True, "T": True}
    outputs = {"power": True, "Qdot": False, "mdot": True, "T": True}
    component_dict = {
        "steel": {
            "model": SteelModel(config.greenheart_config),
            "model_inputs": inputs,
            "model_outputs": outputs,
        }
    }

    return component_dict
