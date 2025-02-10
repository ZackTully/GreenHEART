import numpy as np
import scipy
import networkx as nx
import matplotlib.pyplot as plt
import casadi as ca

from hopp.utilities import load_yaml


class DispatchModelPredictiveController:

    def __init__(self, config, simulation_graph):

        system_graph = load_yaml(
            config.greenheart_config["system"]["system_graph_config"]
        )

        nodes = system_graph["traversal_order"]
        traversal_order = system_graph["traversal_order"]


        self.horizon = 5
        self.G = simulation_graph
        self.traversal_order = traversal_order


        self.build_control_model(traversal_order, simulation_graph)
        self.setup_optimization()

    def build_control_model(self, traversal_order, simulation_graph):

        G = simulation_graph

        A_list = []
        B_list = []
        C_list = []
        D_list = []
        E_list = []
        F_list = []

        constraint_dict = {}

        # for node in traversal_order:
        #     A_list.append(G.nodes[node]["ionode"].model.A)
        #     B_list.append(G.nodes[node]["ionode"].model.B)
        #     C_list.append(G.nodes[node]["ionode"].model.C)
        #     D_list.append(G.nodes[node]["ionode"].model.D)
        #     E_list.append(G.nodes[node]["ionode"].model.E)
        #     F_list.append(G.nodes[node]["ionode"].model.F)
        for node in traversal_order:
            A_list.append(G.nodes[node]["ionode"].A)
            B_list.append(G.nodes[node]["ionode"].B)
            C_list.append(G.nodes[node]["ionode"].C)
            D_list.append(G.nodes[node]["ionode"].D)
            E_list.append(G.nodes[node]["ionode"].E)
            F_list.append(G.nodes[node]["ionode"].F)

            # Collect the constraints from the model
            if hasattr(G.nodes[node]["ionode"], "constraints"):
                constraint_dict.update({node: G.node[node]["ionode"].constraints})

        A = scipy.linalg.block_diag(*(a for a in A_list))
        B = scipy.linalg.block_diag(*(b for b in B_list))
        C = scipy.linalg.block_diag(*(c for c in C_list))
        D = scipy.linalg.block_diag(*(d for d in D_list))
        E = scipy.linalg.block_diag(*(e for e in E_list))
        F = scipy.linalg.block_diag(*(f for f in F_list))

        n = A.shape[0] # number of states
        m = B.shape[1] # number of controllable inputs
        o = E.shape[1] # number of disturbances
        p = C.shape[0] # number of outputs
        q = len(G.edges) # number of edges

        self.A, self.B, self.C, self.D, self.E, self.F = A, B, C, D, E, F
        self.n, self.m, self.o, self.p, self.q = n, m, o, p, q

        # Get the graph constraint matrix

        # assume generation is the source and steel is the sink
        # if no steel then assume it is the heat_exchanger

        M_inc = np.concatenate(
            [
                np.array([[1] + [0] * (len(G.nodes) - 1)]).T,
                nx.incidence_matrix(G, oriented=True).toarray(),
                np.array([[0] * (len(G.nodes) - 1) + [-1]]).T,
            ],
            axis=1,
        )

        M_inc_in = np.where(M_inc > 0, M_inc, 0)
        M_inc_out = np.where(M_inc < 0, -M_inc, 0)

        p_ins = []
        p_outs = []

        for i, node in enumerate(traversal_order):

            p_in = np.zeros((int(np.sum(M_inc_in[i, :])), M_inc.shape[1]))
            in_inds = np.where(M_inc_in[i, :] == 1)[0]
            for j in range(len(in_inds)):
                p_in[j, in_inds[j]] = 1
            p_ins.append(p_in)

            p_out = np.zeros((int(np.sum(M_inc_out[i, :])), M_inc.shape[1]))
            out_inds = np.where(M_inc_out[i, :] == 1)[0]
            for j in range(len(out_inds)):
                p_out[j, out_inds[j]] = 1
            p_outs.append(p_out)

        P_in = np.concatenate(p_ins, axis=0)
        P_out = np.concatenate(p_outs, axis=0)

        self.M_inc = M_inc
        self.M_inc_in = M_inc_in
        self.M_inc_out = M_inc_out
        self.p_ins = p_ins
        self.p_outs = p_outs
        self.P_in = P_in
        self.P_out = P_out

        # Output sorting 
        # edges = list(G.edges)

        # output_sort = []

        # count = 0

        # for node in self.traversal_order:
        #     node_edges = [edge for edge in edges if (node == edge[0])]
        #     # for j in G.nodes[node]["ionode"].out_degree:
        #     for node_edge in node_edges:
        #         output_sort.append((count, node_edge))
        #         count += 1


        # Output sorting

        output_sort = []
        count = 0
        for node in traversal_order:

            for j in range(G.nodes[node]["ionode"].out_degree):
                output_sort.append((count, node ))
                count += 1

        # Separate splitting control and controllable control
        # splitting_sort = []
        splitting_sort = {}
        # ctrl_sort = []
        ctrl_sort = {}
        count = 0
        for node in traversal_order:

            if G.nodes[node]["split"]:
                for j in range(G.nodes[node]["ionode"].out_degree):
                    # splitting_sort.append((count, node ))
                    splitting_sort.update({count:node})
                    count += 1
            else:
                for j in range(G.nodes[node]["ionode"].m):
                    # ctrl_sort.append((count, node))
                    ctrl_sort.update({count:node})
                    count += 1
        self.splitting_sort = splitting_sort
        self.ctrl_sort = ctrl_sort


        []
       

    def setup_optimization(self):

        opti: ca.Opti = ca.Opti()
        opti.solver("ipopt")

        # Variables
        u_var = opti.variable(self.m, self.horizon)
        x_var = opti.variable(self.n, self.horizon + 1)
        y_var = opti.variable(self.p, self.horizon)
        e_var = opti.variable(self.q + 2, self.horizon)

        # Parameters
        e_src_param = opti.parameter(1, self.horizon)
        x0_param = opti.parameter(self.n, 1)
        
        # Initial conditions and forecasted disturbance
        opti.subject_to(x_var[:, 0] == x0_param)
        opti.subject_to(e_var[0, :] == e_src_param)

        # Dynamics constraint
        for i in range(self.horizon):
            dk = self.M_inc_in @ e_var[:, i]

            xkp1 = self.A @ x_var[:, i] + self.B @ u_var[:, i] + self.E @ dk
            yk = self.C @ x_var[:, i] + self.D @ u_var[:, i] + self.F @ dk

            opti.subject_to(x_var[:, i + 1] == xkp1)
            opti.subject_to(y_var[:, i] == yk)
            opti.subject_to(self.P_out @ e_var[:, i] == y_var[:, i])

        # Bounds

        # Objective
        opti.minimize(self.objective(u_var, x_var, y_var))

        self.opti = opti
        self.opt_vars = {"u": u_var, "x": x_var, "y":y_var, "e":e_var}
        self.opt_params = {"e_src":e_src_param, "x0":x0_param}


    def objective(self, u, x, y):

        Qu = np.diag(np.ones(self.m))
        Qx = np.diag(np.ones(self.n))
        Qy = np.diag(np.ones(self.p))

        objective_value = 0
        for i in range(self.horizon):
            objective_value += (
                (u[:, i]).T @ Qu @ (u[:, i])
                + (x[:, i]).T @ Qx @ (x[:, i])
                + (y[:, i]).T @ Qy @ (y[:, i])
            )

        return objective_value


    def update_optimization_parameters(self, x0, src_forecast):
        self.opti.set_value(self.opt_params["e_src"], src_forecast)
        self.opti.set_value(self.opt_params["x0"], x0)

    def update_optimization_constraints(self):
        pass

    def compute_trajectory(self, x0, forecast):
        self.update_optimization_parameters(x0, forecast)

        sol = self.opti.solve()

        u = sol.value(self.opt_vars["u"])
        x = sol.value(self.opt_vars["x"])
        y = sol.value(self.opt_vars["y"])
        e = sol.value(self.opt_vars["e"])

        return u[:, 0]
