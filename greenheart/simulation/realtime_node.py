import numpy as np


class Node:
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
        self.output_list = [
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

    def consolidate_incoming_edges(self, incoming_edges):

        # Take a list of one or more graph edges as inputs and consolidate into one edge

        node_input = np.sum(incoming_edges, axis=0)
        # Temperature = mass-weighted sum of incoming temperatures
        node_input[3] = np.nan_to_num(
            np.dot(incoming_edges[:, 2], incoming_edges[:, 3])
            / np.sum(incoming_edges[:, 2])
        )

        return node_input

    def format_model_input(self, node_input):

        # Take a single graph edge as input with power, heat, H2 mass and temperature
        # reformat to only the arguments that the model needs

        model_input = node_input[np.where(self.input_list)]
        return model_input

    def step(self, incoming_edges, u_control, u_split, step_index):

        node_input = self.consolidate_incoming_edges(incoming_edges)
        model_disturbance = self.format_model_input(node_input)
        if self.name == "generation":
            model_disturbance = self.model.output

        self.store_disturbance(model_disturbance, step_index=step_index)
        y_model, u_passthrough, u_curtail = self.model.step(
            model_disturbance, u_control, step_index
        )

        # assert y_model >= 0

        model_output = self.format_model_output(y_model, u_passthrough)
        outgoing_edges, split_curtail = self.splitting(
            model_output, u_split, step_index
        )
        self.store_passthrough(u_passthrough=u_passthrough, step_index=step_index)
        self.store_curtail(
            u_curtail=u_curtail, split_curtail=split_curtail, step_index=step_index
        )

        return outgoing_edges

    def store_disturbance(self, model_disturbance, step_index=0):
        # if model_disturbance.shape[0] > 0:
        if self.inputs["T"]:
            self.disturbance_store[step_index, :] = model_disturbance[0:-1]
        else:
            self.disturbance_store[step_index, :] = model_disturbance

    def store_passthrough(self, u_passthrough=None, step_index=0):
        self.u_passthrough_store[step_index, :] = u_passthrough

    def store_curtail(self, u_curtail=None, split_curtail=None, step_index=0):
        self.u_curtail_store[step_index, :] = u_curtail
        self.u_curtail_split_store[step_index, :] = split_curtail

    def format_model_output(self, y_model, u_passthrough):
        output_passthrough = np.zeros((1, 4))
        output_passthrough[0, np.where(self.input_list)] = u_passthrough

        output_model = np.zeros((1, 4))
        output_model[0, np.where(self.output_list)] = y_model

        if self.name == "electrolyzer":
            output_model[0, 3] = 80  # degree C
            output_passthrough[0, 3] = 80
        elif self.name == "hydrogen_storage":
            output_model[0, 3] = 20  # degree C
            output_passthrough[0, 3] = 20
        elif self.name == "heat_exchanger":
            output_model[0, 3] = self.model.Tout_desired
            output_passthrough[0, 3] = self.model.Tout_desired

        model_output = output_model + output_passthrough
        model_output[0, 3] = np.nan_to_num(
            (
                output_model[0, 2] * output_model[0, 3]
                + output_passthrough[0, 2] * output_passthrough[0, 3]
            )
            / (output_model[0, 2] + output_passthrough[0, 2])
        )
        return model_output

    def splitting(self, model_output, u_split, step_index):

        # if np.isclose(u_split, np.zeros(len(u_split)), atol=1):
        #     pass

        if (u_split < 0).any():
            assert np.min(u_split) >= -1
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
        # split_curtail = np.subtract(model_output, outgoing_edges)

        # self.store_curtail(split_curtail=split_curtail, step_index=step_index)

        # Double check that splitting hasn't changed the total output
        # assert np.isclose(
        #     np.sum(outgoing_edges, axis=0)[0:3], model_output[0, 0:3], 1e-6
        # ).all()

        # np.sum(outgoing_edges, axis=0)[0:3]- model_output[0, 0:3]

        assert np.all(outgoing_edges >= 0)

        return outgoing_edges.T, split_curtail


if __name__ == "__main__":

    step_index = 10
    graph_input = np.array([[0, 0, 0, 0]])
    node_dispatch_split = np.array([[1, 0]])
    node_dispatch_control = np.array([[0]])

    pass
