import numpy as np
import types

from greenheart.simulation.realtime_node import Node


class StandinModel:
    def __init__(self):
        self.output = np.array([1, 2, 3, 4])
        self.Tout_desired = 50

    def step(self, model_disturbance, u_control, step_index):
        # Returns y_model, u_passthrough, u_curtail
        return 200, np.array([10, 20, 30]), np.array([100, 200, 300])


def make_node(name="test", model=None):
    expected_inputs = {"power": False, "Qdot": True, "mdot": True, "T": True}
    expected_outputs = {"power": False, "Qdot": False, "mdot": True, "T": True}
    if model is None:
        model = StandinModel()
    return Node(
        name=name,
        model=model,
        expected_inputs=expected_inputs,
        expected_outputs=expected_outputs,
        splitting_node=False,
        in_degree=1,
        out_degree=1,
    )


def test_node_repr():
    node = make_node(name="foo")
    assert repr(node) == "foo"


def test_consolidate_incoming_edges():
    node = make_node()

    # If there is only one incoming edge, then the node input should be that edge
    incoming_edges = np.array([[1, 2, 3, 4]])
    node_input = node.consolidate_incoming_edges(incoming_edges)
    assert (node_input == incoming_edges).all()

    incoming_edges = np.array([[1, 2, 3, 10], [2, 3, 4, 20]], dtype=float)
    node_input = node.consolidate_incoming_edges(incoming_edges)
    assert node_input.shape == (4,)

    # Temperature should be mass-weighted average
    assert np.isclose(node_input[3], (3 * 10 + 4 * 20) / (3 + 4))


def test_format_model_input():
    node = make_node()
    node_input = np.array([1, 2, 3, 4])
    result = node.format_model_input(node_input)
    assert np.all(result == node_input[1:])

    # node.inputs["Qdot"] = False
    # node.input_list[1] = False
    # result = node.format_model_input(node_input)
    # assert len(result) == 3
    # assert (result == np.array([1, 3, 4])).all()


def test_store_disturbance_T_true():
    node = make_node()
    disturbance = np.array([1, 2, 3])
    node.store_disturbance(disturbance, step_index=0)

    # Only the first two are stored because the last one is temperature, which is not saved
    assert np.all(node.disturbance_store[0, :] == disturbance[0:2])




def test_store_passthrough():
    node = make_node()
    u_passthrough = np.array([1, 2, 3])
    node.store_passthrough(u_passthrough=u_passthrough[0:2], step_index=0)
    assert np.all(node.u_passthrough_store[0, :] == u_passthrough[0:2])


def test_store_curtail():
    node = make_node()
    u_curtail = np.array([1, 2])
    split_curtail = np.array([4, 5, 6, 7])
    node.store_curtail(u_curtail=u_curtail, split_curtail=split_curtail, step_index=0)
    assert np.all(node.u_curtail_store[0, :3] == u_curtail)
    assert np.all(node.u_curtail_split_store[0, :4] == split_curtail)


def test_format_model_output_electrolyzer():
    node = make_node(name="electrolyzer")
    y_model = np.array([1, 2])
    u_passthrough = np.array([1, 2, 3])
    result = node.format_model_output(y_model, u_passthrough)
    assert result.shape == (1, 4)
    # Temperature should be set to T_electrolyzer_output
    assert result[0, 3] == node.T_electrolyzer_output


def test_format_model_output_hydrogen_storage():
    node = make_node(name="hydrogen_storage")
    y_model = np.array([1, 2])
    u_passthrough = np.array([1, 2, 3])
    result = node.format_model_output(y_model, u_passthrough)
    assert result[0, 3] == node.T_hydrogen_storage_output


def test_format_model_output_heat_exchanger():
    node = make_node(name="heat_exchanger")
    y_model = np.array([1, 2])
    u_passthrough = np.array([1, 2, 3])
    result = node.format_model_output(y_model, u_passthrough)
    assert result[0, 3] == node.model.Tout_desired


def test_splitting_fractional():
    node = make_node()
    node.splitting_method = "fractional"
    model_output = np.array([[1, 2, 3, 4]])
    u_split = np.array([1, 1, 1, 1]) / 4
    step_index = 0
    outgoing_edges, split_curtail = node.splitting(model_output, u_split, step_index)
    assert outgoing_edges.shape[0] == 4
    assert split_curtail.shape[1] == 4

    assert (split_curtail[0, 0:3] == 0).all()
    assert ((outgoing_edges[0:3, 0] - 0.25 * model_output[0, 0:3]) == 0).all()
    assert outgoing_edges[3, 0] == model_output[0, 3]


# def test_splitting_absolute():
#     node = make_node()
#     node.splitting_method = "absolute"
#     model_output = np.array([[1,2,3,4]])
#     u_split = np.array([1,1,1,1])
#     step_index = 0
#     outgoing_edges, split_curtail = node.splitting(model_output, u_split, step_index)
#     assert outgoing_edges.shape[0] == 4
#     assert split_curtail.shape[0] == 4


def test_step_runs():
    node = make_node()
    incoming_edges = np.array([[1, 2, 3, 4]])
    u_control = np.array([1])
    u_split = np.array([1])
    step_index = 0
    outgoing_edges = node.step(incoming_edges, u_control, u_split, step_index)
    assert outgoing_edges.shape[0] == 4
