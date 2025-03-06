import numpy as np
import scipy
import networkx as nx
import matplotlib.pyplot as plt
import casadi as ca
import pprint
import sys
from io import StringIO


from hopp.utilities import load_yaml


class DispatchModelPredictiveController:

    step_index_store: list
    uc_store: list
    us_store: list
    dco_store: list

    def __init__(self, config, simulation_graph):

        system_graph = load_yaml(
            config.greenheart_config["system"]["system_graph_config"]
        )

        nodes = system_graph["traversal_order"]
        traversal_order = system_graph["traversal_order"]

        self.horizon = 48
        self.G = simulation_graph
        self.traversal_order = traversal_order

        self.allow_curtail_forecast = True
        self.include_edges = True
        self.use_sparsity_constraint = False

        self.curtail_storage = np.zeros(8760 + self.horizon)

        self.build_control_model(traversal_order, simulation_graph)
        self.setup_optimization()
        self.setup_solution_storage()

    def setup_solution_storage(self):
        self.step_index_store = []
        self.uc_store = []
        self.us_store = []
        self.x_store = []
        self.yex_store = []
        self.ysp_store = []
        self.curtail_store = []
        self.de_store = []
        self.dco_store = []


    def store_solution(self, step_index, uc, us, x, yex, ysp,  curtail, dex, dco):
        self.step_index_store.append(step_index)
        self.uc_store.append(uc)
        self.us_store.append(us)
        self.x_store.append(x)
        self.yex_store.append(yex)
        self.ysp_store.append(ysp)
        self.curtail_store.append(curtail)
        self.de_store.append(dex)
        self.dco_store.append(dco)

    def build_control_model(self, traversal_order, simulation_graph):

        G = simulation_graph

        A_list = []
        B_list = []
        C_list = []
        D_list = []
        E_list = []
        F_list = []

        Bc_list = []
        Bs_list = []

        Cs_list = []
        Co_list = []

        Dsc_list = []
        Dss_list = []
        Doc_list = []
        Dos_list = []

        E_list = []

        Fs_list = []
        Fo_list = []

        n_list = []
        mc_list = []
        ms_list = []
        p_list = []
        pzero_list = []
        o_list = []

        x_list = []
        uc_list = []
        us_list = []
        d_list = []
        ys_list = []
        yo_list = []
        e_list = []

        u_lb_list = []
        u_ub_list = []
        x_lb_list = []
        x_ub_list = []
        y_lb_list = []
        y_ub_list = []

        constraint_dict = {}

        for node in traversal_order:
            assert hasattr(
                G.nodes[node]["ionode"].model, "control_model"
            ), f"Node {node} has no control model available"

            cm = G.nodes[node]["ionode"].model.control_model

            in_degree = G.nodes[node]["ionode"].in_degree
            out_degree = G.nodes[node]["ionode"].out_degree

            if out_degree > 1:
                us_degree = out_degree
            else:
                us_degree = 0

            n = cm.A.shape[0]

            mc = cm.B.shape[1]
            ms = us_degree
            o = cm.E.shape[1]

            if us_degree == 0:
                ps = cm.C.shape[0]
                po = 0
            else:
                ps = us_degree
                po = cm.C.shape[0]

            A_list.append(cm.A)
            B_list.append(cm.B)
            C_list.append(cm.C)
            D_list.append(cm.D)
            # E_list.append(cm.E)
            F_list.append(np.tile(cm.F, in_degree))

            # First row
            # Already have A matrix
            Bc_list.append(cm.B)
            Bs_list.append(np.zeros((n, ms)))
            E_list.append(np.tile(cm.E, in_degree))

            # Second row
            if ms == 0:  # y_split is just normal y
                Cs_list.append(cm.C)
                Dsc_list.append(cm.D)
                Dss_list.append(np.zeros((ps, ms)))
                Fs_list.append(np.tile(cm.F, in_degree))
            else:  # no dimension here
                Cs_list.append(np.zeros((ps, n)))
                Dsc_list.append(np.zeros((ps, mc)))
                Dss_list.append(np.eye(ms))
                Fs_list.append(np.zeros((ps, in_degree * cm.F.shape[1])))

            # Third row
            if ms == 0:  # y_split is Ds us
                Co_list.append(np.zeros((po, n)))
                Doc_list.append(np.zeros((po, mc)))
                Dos_list.append(np.zeros((po, ms)))
                Fo_list.append(np.zeros((po, in_degree * cm.F.shape[1])))
            else:  # yzero  is constrained here
                Co_list.append(cm.C)
                Doc_list.append(cm.D)
                Dos_list.append(-np.tile(np.eye(po), ms) @ Dss_list[-1])
                Fo_list.append(np.tile(cm.F, in_degree))

            # Should break if the control model has inconsistent dimensions
            subsystem_block = np.block([[cm.A, cm.B, cm.E], [cm.C, cm.D, cm.F]])

            # Dimensions
            n_list.append(n)
            mc_list.append(mc)
            ms_list.append(ms)
            p_list.append(ps + po)
            o_list.append(o)
            pzero_list.append(cm.C.shape[0])

            u_lb_list.append(cm.u_lb)
            u_ub_list.append(cm.u_ub)
            x_lb_list.append(cm.x_lb)
            x_ub_list.append(cm.x_ub)
            y_lb_list.append(cm.y_lb)
            y_ub_list.append(cm.y_ub)

            # sort nodes inputs and outputs into
            # u_c, u_s,
            # d_ex, d_co,
            # y_ex, y_co, y_z

            xl = []
            for i in range(cm.A.shape[0]):
                xl.append(f"x {i} {node}")
            x_list.append(xl)

            ucl = []
            for i in range(cm.B.shape[1]):
                ucl.append(f"uc {i} {node}")
            uc_list.append(ucl)

            dsl = []
            if G.nodes[node]["is_source"]:  # d = d_ex
                for i in range(cm.E.shape[1] * in_degree):
                    dsl.append(f"de {i} {node} (from source)")
            else:  # d = d_co
                for i in range(in_degree):
                    in_edges = list(G.in_edges(node))
                    dsl.append(f"dc {i} {node} (from {in_edges[i][1]})")

            d_list.append(dsl)

            usl = []
            ysl = []
            if G.nodes[node]["is_sink"]:  # ys = y_ex
                for i in range(us_degree):
                    usl.append(f"us {i} {node} (to sink)")
                for i in range(out_degree):
                    ysl.append(f"yse {i} {node} (to sink)")
            else:  # ys = y_co
                for i in range(us_degree):
                    out_edges = list(G.out_edges(node))
                    usl.append(f"us {i} {node} (to {out_edges[i][1]})")

                for i in range(out_degree):
                    out_edges = list(G.out_edges(node))
                    ysl.append(f"ysc {i} {node} (to {out_edges[i][1]})")
            us_list.append(usl)
            ys_list.append(ysl)

            yol = []
            for i in range(po):
                yol.append(f"yo {i} {node}")
            yo_list.append(yol)

        self.x_list = [x for xs in x_list for x in xs]
        self.uc_list = [x for xs in uc_list for x in xs]
        self.us_list = [x for xs in us_list for x in xs]
        self.d_list = [x for xs in d_list for x in xs]
        self.ys_list = [x for xs in ys_list for x in xs]
        self.yo_list = [x for xs in yo_list for x in xs]

        # First row
        A = scipy.linalg.block_diag(*A_list)
        Bc = scipy.linalg.block_diag(*Bc_list)
        Bs = scipy.linalg.block_diag(*Bs_list)
        E = scipy.linalg.block_diag(*E_list)

        # Second row
        Cs = scipy.linalg.block_diag(*Cs_list)
        Dsc = scipy.linalg.block_diag(*Dsc_list)
        Dss = scipy.linalg.block_diag(*Dss_list)
        Fs = scipy.linalg.block_diag(*Fs_list)

        # Third row
        Co = scipy.linalg.block_diag(*Co_list)
        Doc = scipy.linalg.block_diag(*Doc_list)
        Dos = scipy.linalg.block_diag(*Dos_list)
        Fo = scipy.linalg.block_diag(*Fo_list)

        # others
        B = scipy.linalg.block_diag(*(b for b in B_list))
        D = scipy.linalg.block_diag(*(d for d in D_list))
        E = scipy.linalg.block_diag(*(e for e in E_list))
        F = scipy.linalg.block_diag(*(f for f in F_list))

        self.system_row1 = np.block([[A, Bc, Bs, E]])
        self.system_row2 = np.block([[Cs, Dsc, Dss, Fs]])
        self.system_row3 = np.block([[Co, Doc, Dos, Fo]])

        self.system = np.block(
            [[self.system_row1], [self.system_row2], [self.system_row3]]
        )

        # cols = {}
        # rows = {}

        # count = 0
        # col_type = ["x", "uc", "us", "d"]
        # for i, col_list in enumerate([n_list, mc_list, ms_list, o_list]):
        #     for j, node in enumerate(traversal_order):
        #         for k in range(col_list[j]):
        #             cols.update({count: f"{col_type[i]} {k}, {node}"})
        #             count += 1

        # count = 0
        # row_type = ["xdot", "ysp", "split"]
        # for i, row_list in enumerate([n_list, p_list, pzero_list]):
        #     for j, node in enumerate(traversal_order):
        #         for k in range(row_list[j]):
        #             rows.update({count: f"{row_type[i]} {k}, {node}"})
        #             count += 1

        # self.cols = cols
        # self.rows = rows

        # # TODO Do this for the whole decision vector too

        # Save control index order for sorting outputs
        # uc_order = {}
        # us_order = {}

        # uc_count = 0
        # us_count = 0

        # for i in range(len(cols)):
        #     node = cols[i].split(" ")[-1]
        #     if cols[i].startswith("uc"):

        #         # if node in uc_order:
        #         #     uc_order[node].append(count)
        #         # else:

        #         if node not in uc_order:
        #             uc_order.update({node: []})

        #         uc_order[node].append(uc_count)

        #         uc_count += 1

        #     elif cols[i].startswith("us"):
        #         if node not in us_order:
        #             us_order.update({node: []})

        #         us_order[node].append(us_count)
        #         us_count += 1

        # self.uc_order = uc_order
        # self.us_order = us_order

        # ys_order = {}
        # ys_count = 0

        # for i in range(len(rows)):
        #     node = rows[i].split(" ")[-1]

        #     if rows[i].startswith("ysp"):

        #         if node not in ys_order:
        #             ys_order.update({node: []})

        #         ys_order[node].append(ys_count)
        #         ys_count += 1

        # self.ys_order = ys_order

        uc_order = {}

        for i in range(len(self.uc_list)):
            node = self.uc_list[i].split(" ")[-1]
            if node not in uc_order:
                uc_order.update({node: []})

            uc_order[node].append(i)

        self.uc_order = uc_order

        us_order = {}
        for i in range(len(self.us_list)):
            node = self.us_list[i].split(" ")[-3]
            if node not in us_order:
                us_order.update({node: []})

            us_order[node].append(i)

        self.us_order = us_order

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
        self.y_lb_list = np.where(self.y_lb_list == None, -np.inf, self.y_lb_list)
        self.y_ub_list = np.concatenate(y_ub_list)
        self.y_ub_list = np.where(self.y_ub_list == None, np.inf, self.y_ub_list)

        # assert the shapes are correct

        self.q = len(G.edges)
        self.n = len(self.x_list)
        self.mc = len(self.uc_list)
        self.ms = len(self.us_list)
        self.ps = len(self.ys_list)
        self.pse = len([ys for ys in self.ys_list if ys.startswith("yse")])
        self.po = len(self.yo_list)
        self.o = len(self.d_list)
        self.ox = len([d for d in self.d_list if d.startswith("de")])

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

        A_block = A
        B_block = np.block([Bc, Bs])
        E_block = E

        C_block = np.block([[Cs], [Co]])
        D_block = np.block([[Dsc, Dss], [Doc, Dos]])
        F_block = np.block([[Fs], [Fo]])

        u_ctrl = np.array(
            [
                [
                    i
                    for i in range(len(self.uc_list + self.us_list))
                    if (self.uc_list + self.us_list)[i].startswith("uc")
                ]
            ]
        )
        u_split = np.array(
            [
                [
                    i
                    for i in range(len(self.uc_list + self.us_list))
                    if (self.uc_list + self.us_list)[i].startswith("us")
                ]
            ]
        )

        d_ex = np.array(
            [[i for i in range(len(self.d_list)) if self.d_list[i].startswith("de")]]
        )
        d_co = np.array(
            [[i for i in range(len(self.d_list)) if self.d_list[i].startswith("dc")]]
        )

        y_spex = np.array(
            [[i for i in range(len(self.ys_list)) if self.ys_list[i].startswith("yse")]]
        )
        y_co = np.array(
            [[i for i in range(len(self.ys_list)) if self.ys_list[i].startswith("ysc")]]
        )
        y_zero = np.array(
            [
                [
                    i
                    for i in range(len(self.ys_list + self.yo_list))
                    if (self.ys_list + self.yo_list)[i].startswith("yo")
                ]
            ]
        )

        e_co = np.arange(1, len(G.edges) + 1, 1)[None, :]

        # P in calculations still feel fragile. Relies on assumption about the order of edges in the graph. Probably fine but may throw an error some time.

        # TODO: look for a graph configuration where P_out and P_in are not almost Identity then see how weird the M matrix can get.
        M_yco_dco = P_out[y_co.T, e_co] @ np.linalg.inv(P_in[d_co.T, e_co])
        # y_co  = M_yco_dco @ d_co
        self.M_yco_dco = M_yco_dco

        M_dco_yco = P_in[d_co.T, e_co] @ np.linalg.inv(P_out[y_co.T, e_co])
        # d_co = M_dco_yco @ yco
        self.M_dco_yco = M_dco_yco

        # M = P_out[y_co.T, d_co] @ P_in[d_co.T, d_co]

        combined_block, combined_mat = self.reorder_coupling(
            A=A_block,
            B=B_block,
            C=C_block,
            D=D_block,
            E=E_block,
            F=F_block,
            M=M_yco_dco,
            uc=u_ctrl,
            us=u_split,
            de=d_ex,
            dc=d_co,
            ye=y_spex,
            yz=y_zero,
            yc=y_co,
        )

        self.combined_block = combined_block

        self.A, self.Bc, self.Bs, self.E = combined_mat[0]
        self.Cs, self.Dsc, self.Dss, self.Fs = combined_mat[1]
        self.Co, self.Doc, self.Dos, self.Fo = combined_mat[2]
        self.Csp, self.Dspc, self.Dsps, self.Fsp = combined_mat[3]

        # print("\n\n")
        # self.print_block_matrices(mats, in_labels, out_labels)

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

    def reorder_coupling(self, A, B, C, D, E, F, M, uc, us, de, dc, ye, yz, yc):

        x = np.arange(0, A.shape[0], 1)[None, :]

        A = A
        B_c = B[x.T, uc]
        B_s = B[x.T, us]
        E_e = E[x.T, de]
        E_c = E[x.T, dc]

        C_e = C[ye.T, x]
        D_ec = D[ye.T, uc]
        D_es = D[ye.T, us]
        F_ee = F[ye.T, de]
        F_ec = F[ye.T, dc]

        C_c = C[yc.T, x]
        D_cc = D[yc.T, uc]
        D_cs = D[yc.T, us]
        F_ce = F[yc.T, de]
        F_cc = F[yc.T, dc]

        C_z = C[yz.T, x]
        D_zc = D[yz.T, uc]
        D_zs = D[yz.T, us]
        F_ze = F[yz.T, de]
        F_zc = F[yz.T, dc]

        FMi = np.linalg.inv(M - F_cc)
        # FMi = np.linalg.inv(F_cc - M)

        ss_mat = [
            [A, B_c, B_s, E_e],
            [C_e, D_ec, D_es, F_ee],
            [C_z, D_zc, D_zs, F_ze],
            [C_c, D_cc, D_cs, F_ce],
        ]

        coupling_mat = [
            [E_c @ FMi @ C_c, E_c @ FMi @ D_cc, E_c @ FMi @ D_cs, E_c @ FMi @ F_ce],
            [F_ec @ FMi @ C_c, F_ec @ FMi @ D_cc, F_ec @ FMi @ D_cs, F_ec @ FMi @ F_ce],
            [F_zc @ FMi @ C_c, F_zc @ FMi @ D_cc, F_zc @ FMi @ D_cs, F_zc @ FMi @ F_ce],
            [F_cc @ FMi @ C_c, F_cc @ FMi @ D_cc, F_cc @ FMi @ D_cs, F_cc @ FMi @ F_ce],
        ]

        combined_mat = [
            [ss_mat[i][j] + coupling_mat[i][j] for j in range(len(ss_mat[i]))]
            for i in range(len(ss_mat))
        ]

        self.print_block_matrices(
            combined_mat,
            in_labels=["x", "uct", "usp", "dex"],
            out_labels=["x+", "yex", "yze", "yco"],
        )

        state_space = np.block(ss_mat)
        coupling = np.block(coupling_mat)

        combined = state_space + coupling

        return combined, combined_mat

    def setup_optimization(self):

        opti: ca.Opti = ca.Opti()
        p_opts = {"print_time": False, "verbose": False}
        s_opts = {
            "print_level": 0,
            # "linear_solver": "ma27",
            "max_iter":1000,
        }
        opti.solver(
            "ipopt",
            p_opts,
            s_opts,
        )

        # Variables
        uc_var = opti.variable(self.mc, self.horizon)
        us_var = opti.variable(self.ms, self.horizon)
        x_var = opti.variable(self.n, self.horizon + 1)
        ys_var = opti.variable(self.pse, self.horizon)
        # e_var = opti.variable(self.q + 2, self.horizon)
        if self.allow_curtail_forecast:
            curtail = opti.variable(self.ox, self.horizon)
        else:
            curtail = opti.parameter(self.ox, self.horizon)

        # Parameters
        de_param = opti.parameter(self.ox, self.horizon)
        # e_src_param = opti.parameter(1, self.horizon)
        x0_param = opti.parameter(self.n, 1)

        # Initial conditions and forecasted disturbance
        opti.subject_to(x_var[:, 0] == x0_param)
        # opti.subject_to(e_var[0, :] == e_src_param)

        # Dynamics constraint
        for i in range(self.horizon):

            xkp1 = (
                self.A @ x_var[:, i]
                + self.Bc @ uc_var[:, i]
                + self.Bs @ us_var[:, i]
                + self.E @ (de_param[:, i] - curtail[:, i])
            )
            ysk = (
                self.Cs @ x_var[:, i]
                + self.Dsc @ uc_var[:, i]
                + self.Dss @ us_var[:, i]
                + self.Fs @ (de_param[:, i] - curtail[:, i])
            )
            opti.subject_to(
                np.zeros((self.po, 1))
                == self.Co @ x_var[:, i]
                + self.Doc @ uc_var[:, i]
                + self.Dos @ us_var[:, i]
                + self.Fo @ (de_param[:, i] - curtail[:, i])
            )

            opti.subject_to(x_var[:, i + 1] == xkp1)
            opti.subject_to(ys_var[:, i] == ysk)

            if self.use_sparsity_constraint:
                opti.subject_to(
                    us_var[2, i] == ca.if_else(uc_var[1, i] >= 0, uc_var[1, i], 0)
                )
                opti.subject_to(
                    us_var[0, i] == ca.if_else(uc_var[0, i] >= 0, uc_var[0, i], 0)
                )
            # opti.subject_to(us_var[2, i] <= ca.fabs(uc_var[1,i]))

            opti.subject_to(us_var[:, i] >= np.zeros(self.ms))
            # opti.subject_to(e_var[:, i] >= np.zeros(self.q + 2))

            opti.subject_to(uc_var[:, i] >= self.u_lb_list)
            opti.subject_to(uc_var[:, i] <= self.u_ub_list)

            opti.subject_to(x_var[:, i] >= self.x_lb_list)
            opti.subject_to(x_var[:, i] <= self.x_ub_list)

            # opti.subject_to((self.Ds_flat @ ys_var[:, i]) >= self.y_lb_list)
            # opti.subject_to((self.Ds_flat @ ys_var[:, i]) <= self.y_ub_list)

            # trickier constraints - maybe wont work

        if self.allow_curtail_forecast:
            opti.subject_to(curtail <= de_param)
            opti.subject_to(curtail >= np.zeros(curtail.shape))

        # Bounds
        # Add constraint
        # Objective
        opti.minimize(self.objective(x_var, uc_var, us_var, ys_var, curtail))

        self.opti = opti
        self.opt_vars = {
            "uc": uc_var,
            "us": us_var,
            "x": x_var,
            "ys": ys_var,
            # "e": e_var,
        }
        self.opt_params = {"d_ex": de_param, "x0": x0_param}

        if self.allow_curtail_forecast:
            self.opt_vars.update({"curtail": curtail})
        else:
            self.opt_params.update({"curtail": curtail})

    def objective(self, x, uc, us, ys, curtail):

        ref_steel = 45.48e3
        # ref_steel = 37.92e3
        # ref_steel = 20e3
        self.reference = ref_steel

        bes_state_reference = 1200000
        h2s_state_reference = 320467

        objective_value = 0
        for i in range(self.horizon):

            ysp = (
                self.Csp @ x[:, i] + self.Dspc @ uc[:, i] + self.Dsps @ us[:, i]
            )  # + self.Fsp @ de

            tracking_term = (ref_steel - ys[0, i]) ** 2

            # h2s_sparsity = 1e-5 * (ysp[3] ** 2 + ysp[5] ** 2) ** 2
            # h2s_sparsity = 1e-5 * (ysp[3] * ysp[5] ) **2
            # h2s_sparsity = 1e3 * ((ysp[3] +  ysp[5]) + ca.fabs(uc[1,i]) )
            # h2s_sparsity = us[3,i] ** 2 - uc[1, i]**2
            h2s_sparsity = 1e3 * ca.if_else(
                uc[1, i] >= 0, (uc[1, i] - us[3, i]) ** 2, 0
            )
            # h2s_sparsity = (2 * us[3, i] - uc[1, i] - ca.fabs(uc[1, i]))
            # h2s_sparsity = 1e3 * (ysp[3] -  uc[1,i] ) **2
            # h2s_sparsity = 1e3 * (ysp[3] + ysp[5]) **2
            # bes_sparsity = 1e-5 * (ysp[0] ** 2 + ysp[2] ** 2) ** 2
            bes_sparsity = 1e-5 * (ysp[0] * ysp[2]) ** 2

            no_h2s_charge = 1e3 * uc[1, i] ** 2

            bes_state = 1e-5 * (x[0, i] - bes_state_reference) ** 2
            h2s_state = 1e-3 * (x[1, i] - h2s_state_reference) ** 2

            curtail_penalty = 1e-3 * curtail[0, i] ** 2
            storage_agreement = 1e-3 * (0.0218 * uc[0, i] - uc[1, i]) ** 2

            objective_value += (
                tracking_term
                # + bes_state
                # + h2s_state
                # + storage_agreement
                # + h2s_sparsity
                # + bes_sparsity
                # + curtail_penalty
            )

        return objective_value

    def update_optimization_parameters(self, x0, src_forecast):
        self.opti.set_value(self.opt_params["d_ex"], src_forecast)
        self.opti.set_value(self.opt_params["x0"], x0)
        if not self.allow_curtail_forecast:
            self.opti.set_value(
                self.opt_params["curtail"], np.zeros(self.opt_params["curtail"].shape)
            )

    def update_optimization_constraints(self):
        pass

    def compute_trajectory(self, x0, forecast, step_index=0):
        self.update_optimization_parameters(x0, forecast)

        if hasattr(self, "x_init"): # then try to update initial guess
            self.opti.set_initial(self.opt_vars["uc"], self.uc_init)
            self.opti.set_initial(self.opt_vars["us"], self.us_init)
            self.opti.set_initial(self.opt_vars["x"], self.x_init)
            self.opti.set_initial(self.opt_vars["ys"], self.ys_init)
            


        try:

            sol = self.opti.solve()

        except:

            with Capturing() as output:
                self.opti.debug.show_infeasibilities()

            output2 = []

            i = 0
            while i < len(output):
                if output[i].startswith("------- i = "):  # new constraint description
                    num_description = output[i + 1]
                    line_number = output[i + 2]
                    code_description = output[i + 3]
                    at_description = ""
                    # at_description = output[i + 4]

                    violation = float(num_description.split("viol ")[1].split(")")[0])

                    if violation >= 1e-9:

                        print_line = (
                            str(num_description).ljust(45)
                            + code_description.split("opti.subject_to(")[1][:-1].ljust(
                                130
                            )
                            + at_description
                        )
                        pprint.pprint(print_line, width=200)

                        # print(code_description)
                    i += 4
                i += 1

            x_db = self.opti.debug.value(self.opt_vars["x"])
            uc_db = self.opti.debug.value(self.opt_vars["uc"])
            us_db = self.opti.debug.value(self.opt_vars["us"])
            ys_db = self.opti.debug.value(self.opt_vars["ys"])[None, :]

            fig, ax = plt.subplots(
                np.max([uc_db.shape[0], us_db.shape[0], x_db.shape[0], ys_db.shape[0]]),
                4,
                sharex="all",
                layout="constrained",
            )

            to_plot = [x_db, uc_db, us_db, ys_db]
            titles = ["x", "uc", "us", "ys"  ]
            for i in range(len(to_plot)):
                ax[0,i].set_title(titles[i])
                for j in range(len(to_plot[i])):
                    ax[j, i].plot(to_plot[i][j, :])



            np.set_printoptions(linewidth=200, suppress=True, precision=4)

            # print("\nDebug x:")
            # print(x_db)
            # print("\nDebug uc:")
            # print(uc_db)
            # print("\nDebug us:")
            # print(us_db)
            # print("\nDebug ys:")
            # print(ys_db)
            # print(self.opti.debug)  

            assert False, self.opti.debug

        jac = sol.value(ca.jacobian(sol.opti.f, sol.opti.x)).toarray()[0]
        # jac = self.opti.debug.value(ca.jacobian(self.opti.debug.f, self.opti.debug.x)).toarray()[0]
        try:
            assert (np.abs(jac) < 1).any()
            # True
        except:
            np.set_printoptions(linewidth=200, suppress=True, precision=4)

            uc_slice = slice(0, self.mc * self.horizon)
            us_slice = slice(self.mc * self.horizon, (self.mc + self.ms) * self.horizon)
            x_slice = slice(
                (self.mc + self.ms) * self.horizon,
                (self.mc + self.ms) * self.horizon + self.n * (self.horizon + 1),
            )
            ys_slice = slice(
                (self.mc + self.ms) * self.horizon + self.n * (self.horizon + 1), (self.mc + self.ms) * self.horizon + self.n * (self.horizon + 1) + self.pse * self.horizon
            )

            jac_uc = np.reshape(jac[uc_slice], (self.horizon, self.mc))
            jac_us = np.reshape(jac[us_slice], (self.horizon, self.ms))
            jac_x = np.reshape(jac[x_slice], (self.horizon + 1, self.n))
            jac_ys = np.reshape(jac[ys_slice], (self.horizon, self.pse))

            self.print_block_matrices(
                mat=[[jac_uc, jac_us, jac_x[0 : self.horizon, :], jac_ys]],
                in_labels=["jac uc", "jac us", "jac x", "jac yex"],
                out_labels=[f"step {i}" for i in range(self.horizon)],
            )

            []

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
        # e = sol.value(self.opt_vars["e"])
        de = sol.value(self.opt_params["d_ex"])[None, :]
        if self.allow_curtail_forecast:
            curtail = sol.value(self.opt_vars["curtail"])
        else:
            curtail = sol.value(self.opt_params["curtail"])

        self.curtail_storage[step_index : step_index + self.horizon] = curtail

        self.uc_init = uc
        self.us_init = us
        self.x_init = x
        self.ys_init = ys
        self.curtail_init = curtail


        ysp = self.Csp @ x[:, :-1] + self.Dspc @ uc + self.Dsps @ us + self.Fsp @ de
        # coupling disturbances
        dco = self.M_dco_yco @ ysp


        ysp = np.concatenate([ysp, ys[None, :]])



        self.store_solution(step_index=step_index, uc=uc, us=us, x=x, yex=ys, ysp=ysp, curtail=curtail, dex=de, dco = dco)


        if False:

            mat_co = (
                self.Csp @ x[:, 0 : self.horizon]
                + self.Dspc @ uc
                + self.Dsps @ us
                + self.Fsp @ de
            )
            mat_ex = (
                self.Cs @ x[:, 0 : self.horizon]
                + self.Dsc @ uc
                + self.Dss @ us
                + self.Fs @ de
            )
            self.print_block_matrices(
                [[mat_co[None, i, :]] for i in range(mat_co.shape[0])]
                + [list(mat_ex[None, :])],
                in_labels=[""],
                out_labels=self.ys_list,
                no_space=True,
            )

        if self.mc == 1:
            u_ctrl = uc[None, 0]
        else:
            u_ctrl = uc[:, 0]

        if curtail.ndim < 2:
            curtail = curtail[None, :]

        # self.plot_solution(sol, forecast)

        u_split = us[:, 0]
        # return u_ctrl, u_split
        # return u[:, 0]
        return uc, us, curtail

    def plot_solution(self, sol, forecast):

        uc = sol.value(self.opt_vars["uc"])
        us = sol.value(self.opt_vars["us"])
        x = sol.value(self.opt_vars["x"])
        ys = sol.value(self.opt_vars["ys"])[None, :]
        # e = sol.value(self.opt_vars["e"])

        fig, ax = plt.subplots(
            np.max([len(uc), len(us), len(x), len(ys)]),
            4,
            sharex="all",
            layout="constrained",
        )

        to_plot = [x, uc, us, ys]
        for i in range(len(to_plot)):
            for j in range(len(to_plot[i])):
                ax[j, i].plot(to_plot[i][j, :])

        fig, ax = plt.subplots(
            4, 2, figsize=(10, 10), sharex="all", layout="constrained"
        )

        ax[0, 0].fill_between(
            np.arange(0, len(forecast), 1),
            np.zeros(len(forecast)),
            forecast,
            alpha=0.25,
            edgecolor=None,
            color="yellow",
            label="forecast",
        )
        ax[0, 0].plot(us[0:2, :].T, label="generation")

        ax[1, 0].plot(uc[0, :], label="battery charge")
        # ax[1, 0].plot(-e[3, :], label="battery discharge")

        ax[2, 0].plot(forecast - uc[0, :], label="gen + bes")

        ax[2, 1].plot(us[3, :] - us[2, :] - uc[1, :], label="H2 to steel")
        ax[1, 1].plot(uc[1, :], label="H2S charge")
        # ax[1, 1].plot(-e[6, :], label="H2S discharge")

        ax[0, 1].fill_between(
            np.arange(0, uc.shape[1], 1),
            np.zeros(uc.shape[1]),
            np.sum(us[2:4, :], axis=0),
            alpha=0.25,
            edgecolor=None,
            color="blue",
            label="H2 gen",
        )

        ax[0, 1].plot(us[2:4, :].T, label="H2 gen")

        ax[3, 1].plot(ys[0, :], label="Steel")

        for i in range(ax.shape[0]):
            for j in range(ax.shape[1]):
                ax[i, j].legend()

        # uc = sol.value(self.opt_vars["uc"])
        # us = sol.value(self.opt_vars["us"])
        # x = sol.value(self.opt_vars["x"])
        # ys = sol.value(self.opt_vars["ys"])
        # # e = sol.value(self.opt_vars["e"])

        # fig, ax = plt.subplots(
        #     4, 2, figsize=(10, 10), sharex="all", layout="constrained"
        # )

        # ax[0, 0].fill_between(
        #     np.arange(0, len(forecast), 1),
        #     np.zeros(len(forecast)),
        #     forecast,
        #     alpha=0.25,
        #     edgecolor=None,
        #     color="yellow",
        #     label="forecast",
        # )
        # ax[0, 0].plot(e[1:3, :].T, label="generation")

        # ax[1, 0].plot(e[1, :], label="battery charge")
        # ax[1, 0].plot(-e[3, :], label="battery discharge")

        # ax[2, 0].plot(forecast + e[3, :], label="gen + bes")

        # ax[1, 1].plot(e[4, :], label="H2S charge")
        # ax[1, 1].plot(-e[6, :], label="H2S discharge")

        # ax[0, 1].plot(np.sum(e[4:6, :], axis=0), label="H2 gen")

        # ax[3, 1].plot(e[7, :], label="Steel")

        # for i in range(ax.shape[0]):
        #     for j in range(ax.shape[1]):
        #         ax[i, j].legend()

        []

    def print_matrix(self, matrix, rows, cols):
        pass

    def print_block_matrix(self, A, B, C, D, E, F):

        precision = 4

        n = A.shape[1]
        m = B.shape[1]
        o = E.shape[1]

        p = C.shape[0]

        for i in range(n):
            line = ""

            line += "["

            for j in range(n):
                line += f" {A[i,j] : .2g}, "

            line = line[0:-2]
            line += "]  ["

            for j in range(m):
                line += f" {B[i,j] : .2g}, "

            line = line[0:-2]
            line += "]  ["

            for j in range(o):
                line += f" {E[i,j] : .2g}, "

            line = line[0:-2]
            line += "]"

            print(line)

        print("")

        for i in range(p):
            line = ""

            line += "["

            for j in range(n):
                line += f" {C[i,j] : .2g}, "

            line = line[0:-2]
            line += "]  ["

            for j in range(m):
                line += f" {D[i,j] : .2g}, "

            line = line[0:-2]
            line += "]  ["

            for j in range(o):
                line += f" {F[i,j] : .2g}, "

            line = line[0:-2]
            line += "]"

            print(line)

        []

    def print_block_matrices(self, mat, in_labels, out_labels, no_space=False):

        try:
            np.block(mat)
        except:
            AssertionError("bad matrix")

        block_mat = np.block(mat)
        # block_cols = block_mat.shape[1]
        # block_rows = block_mat.shape[0]

        out_label_width = int(np.max([len(label) for label in out_labels]))
        num_col_width = 10


        for row_num, row_mat in enumerate(mat):
            if not no_space:
                print("")

            row_mat_lens = [matr.shape[1] for matr in row_mat]
            if row_num == 0:
                line = " " * (out_label_width + 5)
                for coli, col_label in enumerate(in_labels):
                    line += f"{col_label}".ljust(row_mat_lens[coli] * num_col_width + 3)

                print(line)
            n_rows = row_mat[0].shape[0]
            for i in range(n_rows):
                line = f"{out_labels[row_num]}".ljust(out_label_width + 3)
                line += "["
                for col_mat in row_mat:
                    for j in range(col_mat.shape[1]):
                        line += f"{col_mat[i,j] :.4g}, ".rjust(num_col_width)

                    line = line[0:-2]
                    line += " ]  ["
                line = line[0:-2]
                print(line)

        []


class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio  # free up some memory
        sys.stdout = self._stdout
