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

        self.horizon = 24
        self.G = simulation_graph
        self.traversal_order = traversal_order

        self.build_control_model(traversal_order, simulation_graph)
        self.setup_optimization()

    def build_control_model(self, traversal_order, simulation_graph):

        G = simulation_graph

        A_list = []
        B_list = []
        Bc_list = []
        Bs_zero_list = []
        E_list = []
        Eblock_list = []
        Ds_list = []
        Ds_flat_list = []
        C_list = []
        C_zero_list = []
        D_list = []
        Dc_list = []
        Dc_zero_list = []
        F_list = []
        Fblock_list = []
        Fblock_zero_list = []

        n_list = []
        mc_list = []
        ms_list = []
        p_list = []
        pzero_list = []
        o_list = []

        # A_list = []
        # B_list = []
        # C_list = []
        # D_list = []
        # E_list = []
        # F_list = []

        u_lb_list = []
        u_ub_list = []
        x_lb_list = []
        x_ub_list = []
        y_lb_list = []
        y_ub_list = []

        constraint_dict = {}

        for node in traversal_order:
            assert hasattr(G.nodes[node]["ionode"].model, "control_model")

            cm = G.nodes[node]["ionode"].model.control_model

            in_degree = G.nodes[node]["ionode"].in_degree
            out_degree = G.nodes[node]["ionode"].out_degree

            # First row
            A_list.append(cm.A)
            Bc_list.append(cm.B)
            Bs_zero_list.append(np.zeros((cm.A.shape[0], out_degree)))
            Eblock_list.append(np.repeat(cm.E, in_degree, axis=1))

            # Second row
            C_zero_list.append(np.zeros((out_degree * cm.C.shape[0], cm.C.shape[1])))
            # C_zero_list.append(np.zeros_like(cm.C))
            Dc_zero_list.append(np.zeros((out_degree * cm.D.shape[0], cm.D.shape[1])))
            # Dc_zero_list.append(np.zeros_like(cm.D))
            Ds_list.append(np.eye(out_degree))
            Fblock_zero_list.append(
                np.zeros((out_degree * cm.F.shape[0], in_degree * cm.F.shape[1]))
            )

            # Third row
            C_list.append(cm.C)
            Dc_list.append(cm.D)
            Ds_flat_list.append(
                np.concatenate([np.eye(cm.C.shape[0])] * out_degree, axis=1)
            )
            # Ds_flat_list.append(
            #     scipy.linalg.block_diag(
            #         *(np.ones((1, cm.C.shape[0])) for n in range(out_degree))
            #     )
            # )

            # Already have Ds
            Fblock_list.append(np.repeat(cm.F, in_degree, axis=1))

            # Others
            B_list.append(cm.B)
            D_list.append(cm.D)
            E_list.append(cm.E)
            F_list.append(cm.F)

            # repeat B and D columns (in_degree) times
            # repeat CDF rows (out_degree) times

            # Should break if the control model has inconsistent dimensions
            subsystem_block = np.block([[cm.A, cm.B, cm.E], [cm.C, cm.D, cm.F]])

            # Dimensions
            n_list.append(cm.A.shape[0])
            mc_list.append(cm.B.shape[1])
            ms_list.append(out_degree)
            p_list.append(cm.C.shape[0] * out_degree)
            o_list.append(cm.E.shape[1] * in_degree)
            pzero_list.append(cm.C.shape[0])

            []

            u_lb_list.append(cm.u_lb)
            u_ub_list.append(cm.u_ub)
            x_lb_list.append(cm.x_lb)
            x_ub_list.append(cm.x_ub)
            y_lb_list.append(cm.y_lb)
            y_ub_list.append(cm.y_ub)

        # for node in traversal_order:
        #     A_list.append(G.nodes[node]["ionode"].model.A)
        #     B_list.append(G.nodes[node]["ionode"].model.B)
        #     C_list.append(G.nodes[node]["ionode"].model.C)
        #     D_list.append(G.nodes[node]["ionode"].model.D)
        #     E_list.append(G.nodes[node]["ionode"].model.E)
        #     F_list.append(G.nodes[node]["ionode"].model.F)
        # for node in traversal_order:
        #     A_list.append(G.nodes[node]["ionode"].A)
        #     B_list.append(G.nodes[node]["ionode"].B)
        #     C_list.append(G.nodes[node]["ionode"].C)
        #     D_list.append(G.nodes[node]["ionode"].D)
        #     E_list.append(G.nodes[node]["ionode"].E)
        #     F_list.append(G.nodes[node]["ionode"].F)

        #     # Collect the constraints from the model
        #     if hasattr(G.nodes[node]["ionode"], "constraints"):
        #         constraint_dict.update({node: G.node[node]["ionode"].constraints})

        # First row
        A = scipy.linalg.block_diag(*(mat for mat in A_list))
        Bc = scipy.linalg.block_diag(*(mat for mat in Bc_list))
        Bs_zero = scipy.linalg.block_diag(*(mat for mat in Bs_zero_list))
        Eblock = scipy.linalg.block_diag(*(mat for mat in Eblock_list))

        # Second row
        C_zero = scipy.linalg.block_diag(*(mat for mat in C_zero_list))
        Dc_zero = scipy.linalg.block_diag(*(mat for mat in Dc_zero_list))
        Ds = scipy.linalg.block_diag(*(mat for mat in Ds_list))
        Fblock_zero = scipy.linalg.block_diag(*(mat for mat in Fblock_zero_list))

        # Third row
        C = scipy.linalg.block_diag(*(mat for mat in C_list))
        Dc = scipy.linalg.block_diag(*(mat for mat in Dc_list))
        Ds_flat = scipy.linalg.block_diag(*(mat for mat in Ds_flat_list))
        Fblock = scipy.linalg.block_diag(*(mat for mat in Fblock_list))

        # others
        B = scipy.linalg.block_diag(*(b for b in B_list))
        D = scipy.linalg.block_diag(*(d for d in D_list))
        E = scipy.linalg.block_diag(*(e for e in E_list))
        F = scipy.linalg.block_diag(*(f for f in F_list))



        self.A = A 
        self.Bc = Bc 
        self.Bs_zero = Bs_zero 
        self.Eblock = Eblock 
        self.C_zero = C_zero 
        self.Dc_zero = Dc_zero 
        self.Ds = Ds 
        self.Fblock_zero = Fblock_zero 
        self.C = C 
        self.Dc = Dc 
        self.Ds_flat = Ds_flat 
        self.Fblock = Fblock 
        
        
        # big block

        self.system_row1 = np.block([[A, Bc, Bs_zero, Eblock]])
        self.system_row2 = np.block([[C_zero, Dc_zero, Ds, Fblock_zero]])
        # TODO multiply Ds by flattening dimension corrector
        self.system_row3 = np.block([[C, Dc, -Ds_flat @ Ds, Fblock]])

        self.system = np.block(
            [[self.system_row1], [self.system_row2], [self.system_row3]]
        )

        cols = {}
        rows = {}

        count = 0
        col_type = ["x", "uc", "us", "d"]
        for i, col_list in enumerate([n_list, mc_list, ms_list, o_list]):
            for j, node in enumerate(traversal_order):
                for k in range(col_list[j]):
                    cols.update({count: f"{col_type[i]} {k}, {node}"})
                    count += 1

        count = 0
        row_type = ["xdot", "ysp", "split"]
        for i, row_list in enumerate([n_list, p_list, pzero_list]):
            for j, node in enumerate(traversal_order):
                for k in range(row_list[j]):
                    rows.update({count: f"{row_type[i]} {k}, {node}"})
                    count += 1

        self.cols = cols
        self.rows = rows

        # TODO Do this for the whole decision vector too

        # Save control index order for sorting outputs
        uc_order = {}
        us_order = {}

        uc_count = 0
        us_count = 0

        for i in range(len(cols)):
            node = cols[i].split(" ")[-1]
            if cols[i].startswith("uc"):

                # if node in uc_order:
                #     uc_order[node].append(count)
                # else:

                if node not in uc_order:
                    uc_order.update({node: []})

                uc_order[node].append(uc_count)

                uc_count += 1

            elif cols[i].startswith("us"):
                if node not in us_order:
                    us_order.update({node: []})

                us_order[node].append(us_count)
                us_count += 1

        self.uc_order = uc_order
        self.us_order = us_order

        ys_order = {}
        ys_count = 0

        for i in range(len(rows)):
            node = rows[i].split(" ")[-1]

            if rows[i].startswith("ysp"):

                if node not in ys_order:
                    ys_order.update({node: []})

                ys_order[node].append(ys_count)
                ys_count += 1

        self.ys_order = ys_order

        edge_order = {}
        edge_order.update({0: ("source", "generation")})
        for i, edge in enumerate(list(self.G.edges)):
            edge_order.update({i + 1: edge})

        edge_order.update({len(edge_order): ("steel", "sink")})

        self.edge_order = edge_order

        self.u_lb_list = np.concatenate(u_lb_list)
        self.u_ub_list = np.concatenate(u_ub_list)
        self.x_lb_list = np.concatenate(x_lb_list)
        self.x_ub_list = np.concatenate(x_ub_list)
        self.y_lb_list = np.concatenate(y_lb_list)
        self.y_lb_list = np.where(self.y_lb_list == None,  -np.inf, self.y_lb_list)
        self.y_ub_list = np.concatenate(y_ub_list)
        self.y_ub_list = np.where(self.y_ub_list == None,  np.inf, self.y_ub_list)



        # assert the shapes are correct

        # n = A.shape[0]  # number of states
        # m = B.shape[1]  # number of controllable inputs
        # o = E.shape[1]  # number of disturbances
        # p = C.shape[0]  # number of outputs
        # q = len(G.edges)  # number of edges

        # self.A, self.B, self.C, self.D, self.E, self.F = A, B, C, D, E, F
        # self.n, self.m, self.o, self.p, self.q = n, m, o, p, q

        q = len(G.edges)
        n = np.sum(n_list)
        mc = np.sum(mc_list)
        ms = np.sum(ms_list)
        p = np.sum(p_list)
        pz = np.sum(pzero_list)
        o = np.sum(o_list)

        self.n, self.mc, self.ms, self.p, self.pz, self.o, self.q = (
            n,
            mc,
            ms,
            p,
            pz,
            o,
            q,
        )

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
                output_sort.append((count, node))
                count += 1

        # Separate splitting control and controllable control
        # splitting_sort = []
        splitting_sort = {}
        # ctrl_sort = []
        ctrl_sort = {}
        # count = 0
        # for node in traversal_order:

        #     if G.nodes[node]["split"]:
        #         for j in range(G.nodes[node]["ionode"].out_degree):
        #             # splitting_sort.append((count, node ))
        #             splitting_sort.update({count: node})
        #             count += 1
        #     else:
        #         for j in range(G.nodes[node]["ionode"].m):
        #             # ctrl_sort.append((count, node))
        #             ctrl_sort.update({count: node})
        #             count += 1
        self.splitting_sort = splitting_sort
        self.ctrl_sort = ctrl_sort

        []

    def setup_optimization(self):

        opti: ca.Opti = ca.Opti()
        p_opts = {"print_time": False, "verbose": False}
        s_opts = {"print_level": 0}
        opti.solver("ipopt", p_opts, s_opts)

        # Variables
        uc_var = opti.variable(self.mc, self.horizon)
        us_var = opti.variable(self.ms, self.horizon)
        x_var = opti.variable(self.n, self.horizon + 1)
        ys_var = opti.variable(self.p, self.horizon)
        e_var = opti.variable(self.q + 2, self.horizon)

        # Parameters
        e_src_param = opti.parameter(1, self.horizon)
        x0_param = opti.parameter(self.n, 1)

        # Initial conditions and forecasted disturbance
        opti.subject_to(x_var[:, 0] == x0_param)
        opti.subject_to(e_var[0, :] == e_src_param)

        # Dynamics constraint
        for i in range(self.horizon):
            dk = self.P_in @ e_var[:, i]
            x_uc_us_d = ca.vertcat(x_var[:, i], uc_var[:, i], us_var[:, i], dk)
            xd_ysp_zero = self.system @ x_uc_us_d

            xkp1 = self.system_row1 @ x_uc_us_d
            ysp = self.system_row2 @ x_uc_us_d
            y_cons = self.system_row3 @ x_uc_us_d

            opti.subject_to(x_var[:, i + 1] == xkp1)
            opti.subject_to(ys_var[:, i] == ysp)
            opti.subject_to(y_cons == np.zeros((self.pz, 1)))
            opti.subject_to(self.P_out @ e_var[:, i] == ys_var[:, i])

            # dk = self.M_inc_in @ e_var[:, i]
            # xkp1 = self.A @ x_var[:, i] + self.B @ u_var[:, i] + self.E @ dk
            # yk = self.C @ x_var[:, i] + self.D @ u_var[:, i] + self.F @ dk

            # opti.subject_to(x_var[:, i + 1] == xkp1)
            # opti.subject_to(y_var[:, i] == yk)
            # opti.subject_to(self.P_out @ e_var[:, i] == y_var[:, i])

            opti.subject_to(us_var[:, i] >= np.zeros(self.ms))
            opti.subject_to(e_var[:, i] >= np.zeros(self.q + 2))

            opti.subject_to(uc_var[:,i] >= self.u_lb_list)
            opti.subject_to(uc_var[:,i] <= self.u_ub_list)

            opti.subject_to(x_var[:,i] >= self.x_lb_list)
            opti.subject_to(x_var[:,i] <= self.x_ub_list)

            opti.subject_to((self.Ds_flat @ ys_var[:,i]) >= self.y_lb_list)
            opti.subject_to((self.Ds_flat @ ys_var[:,i]) <= self.y_ub_list)



            # trickier constraints - maybe wont work


        # Bounds
        # Add constraint 
        # Objective
        opti.minimize(self.objective(x_var, uc_var, us_var, ys_var, e_var))

        self.opti = opti
        self.opt_vars = {
            "uc": uc_var,
            "us": us_var,
            "x": x_var,
            "ys": ys_var,
            "e": e_var,
        }
        # self.opt_vars = {"u": u_var, "x": x_var, "y": y_var, "e": e_var}
        self.opt_params = {"e_src": e_src_param, "x0": x0_param}

    def objective(self, x, uc, us, ys, e):

        reference = np.array([114679, 2440.5, 37e3])
        # reference = np.array([134228.5, 2440.5, 37e3])
        # reference = np.array([1.42e5, 2950, 37e3])
        # reference = np.array([1.42e5, 2950, 38e3])

        self.reference = reference
        reorder_edge = np.array(
            [
                [0, 0, 1, 1, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ]
        )
        Q_edge = np.diag(np.array([1, 0, 1]))

        objective_value = 0
        for i in range(self.horizon):

            tracking_term = (
                (reorder_edge @ e[:,i] - reference).T
                @ Q_edge
                @ (reorder_edge @ e[:,i] - reference)
            )

            # BES_sparsity_term = ca.fabs(e[1,i] * e[3,i])
            BES_sparsity_term = (e[1,i] * e[3,i])**2
            H2S_sparsity_term = (e[4,i] * e[6,i])**2
            # H2S_sparsity_term = ca.fabs(e[4,i] * e[6,i])

            objective_value += tracking_term + BES_sparsity_term  + H2S_sparsity_term

        return objective_value

    # def objective(self, x, uc, us, ys):

    #     Qx = np.diag(np.ones(self.n))
    #     Quc = np.diag(np.ones(self.mc))
    #     Qus = np.diag(np.ones(self.ms))
    #     Qys = np.diag(np.ones(self.p))

    #     objective_value = 0
    #     for i in range(self.horizon):
    #         objective_value += (
    #             (x[:, i]).T @ Qx @ (x[:, i])
    #             + (uc[:, i]).T @ Quc @ (uc[:, i])
    #             + (us[:, i]).T @ Qus @ (us[:, i])
    #             + (ys[:, i]).T @ Qys @ (ys[:, i])
    #         )

    #     return objective_value

    # def objective(self, u, x, y):

    #     Qu = np.diag(np.ones(self.m))
    #     Qx = np.diag(np.ones(self.n))
    #     Qy = np.diag(np.ones(self.p))

    #     objective_value = 0
    #     for i in range(self.horizon):
    #         objective_value += (
    #             (u[:, i]).T @ Qu @ (u[:, i])
    #             + (x[:, i]).T @ Qx @ (x[:, i])
    #             + (y[:, i]).T @ Qy @ (y[:, i])
    #         )

    #     return objective_value

    def update_optimization_parameters(self, x0, src_forecast):
        self.opti.set_value(self.opt_params["e_src"], src_forecast)
        self.opti.set_value(self.opt_params["x0"], x0)

    def update_optimization_constraints(self):
        pass

    def compute_trajectory(self, x0, forecast):
        self.update_optimization_parameters(x0, forecast)

        sol = self.opti.solve()

        # self.opti.debug.value_parameters()
        # self.opti.debug.value_variables()
        # self.opti.debug.stats()
        # self.opti.debug.arg()
        # self.opti.debug.constraints()
        # self.opti.debug.show_infeasibilities()

        uc = sol.value(self.opt_vars["uc"])
        us = sol.value(self.opt_vars["us"])
        x = sol.value(self.opt_vars["x"])
        ys = sol.value(self.opt_vars["ys"])
        e = sol.value(self.opt_vars["e"])

        if self.mc == 1:
            u_ctrl = uc[None, 0]
        else:
            u_ctrl = uc[:, 0]

        # self.plot_solution(sol, forecast)

        u_split = us[:, 0]
        # return u_ctrl, u_split
        # return u[:, 0]
        return uc, us
    
    def plot_solution(self, sol, forecast):
        

        uc = sol.value(self.opt_vars["uc"])
        us = sol.value(self.opt_vars["us"])
        x = sol.value(self.opt_vars["x"])
        ys = sol.value(self.opt_vars["ys"])
        e = sol.value(self.opt_vars["e"])

        fig, ax = plt.subplots(4,2, figsize=(10,10), sharex="all", layout="constrained")

        ax[0,0].fill_between(np.arange(0, len(forecast), 1), np.zeros(len(forecast)), forecast, alpha=.25, edgecolor=None, color="yellow", label="forecast")
        ax[0,0].plot(e[1:3,:].T, label="generation")

        ax[1,0].plot(e[1,:], label="battery charge")
        ax[1,0].plot(-e[3,:], label="battery discharge")

        ax[2,0].plot(forecast + e[3,:], label="gen + bes")
        
        ax[1,1].plot(e[4,:], label="H2S charge")
        ax[1,1].plot(-e[6,:], label="H2S discharge")

        ax[0,1].plot(np.sum(e[4:6,:], axis=0), label="H2 gen")

        ax[3, 1].plot(e[7,:], label="Steel")


        for i in range(ax.shape[0]):
            for j in range(ax.shape[1]):
                ax[i,j].legend()

    def print_matrix(self, matrix, rows, cols):
        pass