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
    uct_store: list
    usp_store: list
    dco_store: list

    def __init__(self, config, simulation_graph, node_order = None, edge_order = None):

        system_graph = load_yaml(
            config.greenheart_config["system"]["system_graph_config"]
        )

        nodes = system_graph["traversal_order"]
        traversal_order = system_graph["traversal_order"]

        self.node_order = node_order
        self.edge_order = edge_order


        self.horizon = 48
        self.G = simulation_graph
        self.traversal_order = traversal_order

        self.allow_curtail_forecast = True
        self.include_edges = True
        self.use_sparsity_constraint = False

        self.curtail_storage = np.zeros(8760 + self.horizon)

        # self.build_control_model(traversal_order, simulation_graph)
        self.collect_system_matrices(traversal_order, simulation_graph)
        self.setup_optimization()
        self.setup_solution_storage()

    def setup_solution_storage(self):
        self.step_index_store = []
        self.uct_store = []
        self.usp_store = []
        self.x_store = []
        self.yex_store = []
        self.ysp_store = []
        self.curtail_store = []
        self.de_store = []
        self.dco_store = []

    def store_solution(self, step_index, uc, us, x, yex, ysp, curtail, dex, dco):
        self.step_index_store.append(step_index)
        self.uct_store.append(np.atleast_2d(uc))
        self.usp_store.append(np.atleast_2d(us))
        self.x_store.append(np.atleast_2d(x))
        self.yex_store.append(np.atleast_2d(yex))
        self.ysp_store.append(np.atleast_2d(ysp))
        self.curtail_store.append(np.atleast_2d(curtail))
        self.de_store.append(np.atleast_2d(dex))
        self.dco_store.append(np.atleast_2d(dco))

    def collect_system_matrices(self, traversal_order, G):

        dims = {
            "dims": {
                "n": [],  # number of states
                "mct": [],  # number of control inputs
                "msp": [],  # number of splitting inputs
                "m": [],  # total number of inputs
                "oex": [],  # number of external disturbances
                "oco": [],  # number of coupling disturbances
                "o": [],  # total number of disturbances
                "pex": [],  # number of external outputs
                "pco": [],  # number of coupling outputs
                "pze": [],  # number of zero output constraints (splitting)
                "pet": [],  # number of equal to zero output contraints (from cm)
                "pgt": [],  # number of greater than zero output constraints (from cm)
                "p": [],  # total number of outputs
                "pcons": [],  # total number of output constraints
            },
            "labels": {
                "n": [],
                "mct": [],
                "msp": [],
                "m": [],
                "oex": [],
                "oco": [],
                "o": [],
                "pex": [],
                "pco": [],
                "pze": [],
                "pet": [],
                "pgt": [],
                "p": [],
                "pcons": [],
            },
        }

        bounds = {
            "u_lb": [],
            "u_ub": [],
            "x_lb": [],
            "x_ub": [],
            "y_lb": [],
            "y_ub": [],
        }

        mats1 = {"A": [], "Bct": [], "Bsp": [], "Eco": [], "Eex": []}
        mats2 = {"Cex": [], "Dexct": [], "Dexsp": [], "Fexco": [], "Fexex": []}
        mats3 = {"Cco": [], "Dcoct": [], "Dcosp": [], "Fcoco": [], "Fcoex": []}
        mats4 = {"Cze": [], "Dzect": [], "Dzesp": [], "Fzeco": [], "Fzeex": []}
        mats5 = {"Cgt": [], "Dgtct": [], "Dgtsp": [], "Fgtco": [], "Fgtex": []}
        mats6 = {"Cet": [], "Detct": [], "Detsp": [], "Fetco": [], "Fetex": []}

        uct_order = {}
        usp_order = {}

        for node in traversal_order:

            cm = G.nodes[node]["ionode"].model.control_model

            in_degree = G.nodes[node]["ionode"].in_degree
            out_degree = G.nodes[node]["ionode"].out_degree

            if out_degree > 1:
                usp_degree = out_degree
            else:
                usp_degree = 0

            # identify the component model dimensions
            n = cm.A.shape[0]

            # create state labels
            x_labels = []
            for i in range(n):
                x_labels.append(f"x {i} {node}")

            mct = cm.B.shape[1]
            msp = usp_degree
            m = mct + msp




            # create controllable input label lists

            uct_indices = []

            uct_labels = []
            for i in range(mct):
                uct_indices.append(int(len(uct_labels) + np.sum(dims["dims"]["mct"])))
                uct_labels.append(f"uct {i} {node}")

            if len(uct_indices) > 0:
                uct_order.update({node:uct_indices})

            usp_indices = []
            usp_labels = []
            for i in range(usp_degree):
                usp_indices.append(int(len(usp_labels) + np.sum(dims["dims"]["msp"])))
                out_edges = list(G.out_edges(node))
                usp_labels.append(f"usp {i} {node} (to {out_edges[i][1]})")

            if len(usp_indices) > 0:
                usp_order.update({node:usp_indices})

            # create uncontrollable input label lists
            dex_labels = []
            dco_labels = []

            if G.nodes[node]["is_source"]:
                oex = cm.F.shape[1]
                assert in_degree == 1
                oco = 0

                for i in range(oex):
                    dex_labels.append(f"dex {i} {node}")

            else:
                oex = 0
                oco = cm.F.shape[1] * in_degree

                for i in range(in_degree):
                    in_edges = list(G.in_edges(node))
                    dco_labels.append(f"dco {i} {node} (from {in_edges[i][0]})")

            o = oex + oco

            # create output label lists

            yex_labels = []
            yco_labels = []
            yze_labels = []
            yet_labels = []
            ygt_labels = []

            if G.nodes[node]["is_sink"]:
                pex = cm.C.shape[0]
                for i in range(pex):
                    yex_labels.append(f"yex {i} {node}")
            else:
                pex = 0

            if usp_degree > 0:
                pze = cm.C.shape[0]
                assert pex == 0, "sink node should not be splitting"
                pco = usp_degree

                for i in range(pze):
                    yze_labels.append(f"yze {i} {node}")

            else:
                pze = 0
                pco = cm.C.shape[0] - pex

            if not G.nodes[node]["is_sink"]:
                for i in range(out_degree):
                    out_edges = list(G.out_edges(node))
                    yco_labels.append(f"yco {i} {node} (to {out_edges[i][1]})")

            pet = cm.C_et.shape[0]
            pgt = cm.C_gt.shape[0]

            for i in range(pet):
                yet_labels.append(f"yet {i} {node}")

            for i in range(pgt):
                ygt_labels.append(f"ygt {i} {node}")

            p = pex + pco
            pcons = pze + pet + pgt

            # Check the incoming edges for domain agreement
            in_edges = list(G.in_edges(node))
            disturbance_index = []
            for in_edge in in_edges:
                up_node = in_edge[0]
                up_cm = G.nodes[up_node]["ionode"].model.control_model
                up_node_output_domain = up_cm.output_domain
                disturbance_index.append(
                    np.where(cm.disturbance_domain @ up_node_output_domain == 1)[0]
                )

            oco = len(disturbance_index)

            # assert oco == len(disturbance_index)

            dim_list = [n, mct, msp, m, oex, oco, o, pex, pco, pze, pet, pgt, p, pcons]
            labels_list = [
                x_labels,
                uct_labels,
                usp_labels,
                [],
                dex_labels,
                dco_labels,
                [],
                yex_labels,
                yco_labels,
                yze_labels,
                yet_labels,
                ygt_labels,
                [],
                [],
            ]

            for i, key in enumerate(dims["dims"].keys()):
                dims["dims"][key].append(dim_list[i])
                dims["labels"][key].append(labels_list[i])

            # store bounds from the cm

            bounds["u_lb"].append(cm.u_lb)
            bounds["u_ub"].append(cm.u_ub)
            bounds["x_lb"].append(cm.x_lb)
            bounds["x_ub"].append(cm.x_ub)
            bounds["y_lb"].append(cm.y_lb)
            bounds["y_ub"].append(cm.y_ub)

            # Collect the relevant matrices

            # state transition row
            A = cm.A
            Bct = cm.B
            Bsp = np.zeros((n, usp_degree))
            if G.nodes[node]["is_source"]:
                Eco = np.zeros((n, 0))
                Eex = cm.E
            else:
                Eco = np.concatenate(
                    [cm.E[:, di[0], None] for di in disturbance_index], axis=1
                )
                # Eco = np.tile(cm.E, in_degree)
                Eex = np.zeros((n, 0))

            m1 = [A, Bct, Bsp, Eco, Eex]
            for i, key in enumerate(mats1.keys()):
                mats1[key].append(m1[i])

            # external output row
            if G.nodes[node]["is_sink"]:
                Cex = cm.C
                Dexct = cm.D
                Dexsp = np.zeros((pex, msp))

                assert not G.nodes[node][
                    "is_source"
                ], "source should not be the same as sink"

                Fexco = np.concatenate(
                    [cm.F[:, di[0], None] for di in disturbance_index], axis=1
                )
                # Fexco = np.tile(cm.F, in_degree)
                Fexex = np.zeros((pex, oex))
            else:
                Cex = np.zeros((0, n))
                Dexct = np.zeros((0, mct))
                Dexsp = np.zeros((0, msp))
                Fexco = np.zeros((0, oco))
                Fexex = np.zeros((0, oex))

            m2 = [Cex, Dexct, Dexsp, Fexco, Fexex]
            for i, key in enumerate(mats2.keys()):
                mats2[key].append(m2[i])

            # coupling output row
            if G.nodes[node]["is_sink"]:
                # if it is the sink node then there should be no coupling outputs
                Cco = np.zeros((pco, n))
                Dcoct = np.zeros((pco, mct))
                Dcosp = np.zeros((pco, msp))
                Fcoco = np.zeros((pco, oco))
                Fcoex = np.zeros((pco, oex))

                # and if it is a sink node then there will be no splitting constraints

                # splitting zero constraint row
                Cze = np.zeros((pze, n))
                Dzect = np.zeros((pze, mct))
                Dzesp = np.zeros((pze, msp))
                Fzeco = np.zeros((pze, oco))
                Fzeex = np.zeros((pze, oex))

            else:

                # splitting zero constraint row
                if usp_degree > 1:
                    # not sink node but is splitting node

                    Cco = np.zeros((pco, n))
                    Dcoct = np.zeros((pco, mct))
                    Dcosp = np.eye(msp)
                    Fcoco = np.zeros((pco, oco))
                    Fcoex = np.zeros((pco, oex))

                    Cze = cm.C
                    Dzect = cm.D
                    Dzesp = -np.tile(
                        np.eye(cm.C.shape[0]), msp
                    )  # Dsp matrix is not in here but it should be okay because the splitting matrix will always be identity
                    if G.nodes[node]["is_source"]:
                        Fzeco = np.zeros((pze, oco))
                        Fzeex = cm.F
                    else:
                        Fzeco = np.tile(cm.F, in_degree)
                        Fzeex = np.zeros((pze, oex))

                else:
                    # not sink node and not splitting node

                    Cco = cm.C
                    Dcoct = cm.D
                    Dcosp = np.zeros((pco, msp))
                    if G.nodes[node]["is_source"]:
                        Fcoco = np.zeros((pco, 0))
                        Fcoex = cm.F
                    else:
                        Fcoco = np.concatenate(
                            [cm.F[:, di[0], None] for di in disturbance_index], axis=1
                        )
                        # Fcoco = np.tile(cm.F, in_degree)
                        Fcoex = np.zeros((pco, 0))

                    Cze = np.zeros((pze, n))
                    Dzect = np.zeros((pze, mct))
                    Dzesp = np.zeros((pze, msp))
                    Fzeco = np.zeros((pze, oco))
                    Fzeex = np.zeros((pze, oex))

            m3 = [Cco, Dcoct, Dcosp, Fcoco, Fcoex]
            for i, key in enumerate(mats3.keys()):
                mats3[key].append(m3[i])

            m4 = [Cze, Dzect, Dzesp, Fzeco, Fzeex]
            for i, key in enumerate(mats4.keys()):
                mats4[key].append(m4[i])

            # greater than zero constraint row
            Cgt = cm.C_gt
            Dgtct = cm.D_gt
            Dgtsp = np.zeros((pgt, msp))
            if G.nodes[node]["is_source"]:
                Fgtco = np.zeros((pgt, oco))
                Fgtex = cm.F_gt
            else:
                Fgtco = np.concatenate(
                    [cm.F_gt[:, di[0], None] for di in disturbance_index], axis=1
                )
                # Fgtco = np.tile(cm.F_gt, in_degree)
                Fgtex = np.zeros((pgt, oex))

            m5 = [Cgt, Dgtct, Dgtsp, Fgtco, Fgtex]
            for i, key in enumerate(mats5.keys()):
                mats5[key].append(m5[i])

            # equal to zero contraint row
            Cet = cm.C_et
            Detct = cm.D_et
            Detsp = np.zeros((pet, msp))
            if G.nodes[node]["is_source"]:
                Fetco = np.zeros((pet, oco))
                Fetex = cm.F_et
            else:
                Fetco = np.concatenate(
                    [cm.F_et[:, di[0], None] for di in disturbance_index], axis=1
                )
                # Fetco = np.tile(cm.F_et, in_degree)
                Fetex = np.zeros((pet, oex))

            m6 = [Cet, Detct, Detsp, Fetco, Fetex]
            for i, key in enumerate(mats6.keys()):
                mats6[key].append(m6[i])

        A, Bct, Bsp, Eco, Eex = (
            scipy.linalg.block_diag(*mats1[key]) for key in mats1.keys()
        )
        Cex, Dexct, Dexsp, Fexco, Fexex = (
            scipy.linalg.block_diag(*mats2[key]) for key in mats2.keys()
        )
        Cco, Dcoct, Dcosp, Fcoco, Fcoex = (
            scipy.linalg.block_diag(*mats3[key]) for key in mats3.keys()
        )
        Cze, Dzect, Dzesp, Fzeco, Fzeex = (
            scipy.linalg.block_diag(*mats4[key]) for key in mats4.keys()
        )
        Cgt, Dgtct, Dgtsp, Fgtco, Fgtex = (
            scipy.linalg.block_diag(*mats5[key]) for key in mats5.keys()
        )
        Cet, Detct, Detsp, Fetco, Fetex = (
            scipy.linalg.block_diag(*mats6[key]) for key in mats6.keys()
        )

        np.block(
            [
                [A, Bct, Bsp, Eco, Eex],
                [Cex, Dexct, Dexsp, Fexco, Fexex],
                [Cco, Dcoct, Dcosp, Fcoco, Fcoex],
                [Cze, Dzect, Dzesp, Fzeco, Fzeex],
                [Cgt, Dgtct, Dgtsp, Fgtco, Fgtex],
                [Cet, Detct, Detsp, Fetco, Fetex],
            ]
        )

        labels = dims["labels"]
        dims = dims["dims"]

        for key in dims.keys():
            setattr(self, key, np.sum(dims[key]))

        for key in labels.keys():
            labels[key] = [x for xs in labels[key] for x in xs]

        labels["m"] = labels["mct"] + labels["msp"]

        # this order comes from assumption baked into the node order
        labels["o"] = labels["oex"] + labels["oco"]
        labels["p"] = labels["pco"] + labels["pex"]
        labels["pcons"] = labels["pze"] + labels["pet"] + labels["pgt"]

        for key in labels.keys():
            setattr(self, f"{key}_label", labels[key])


        # Make indices and reduce the order of the verbose statespace

        # extended incidence matrix
        E_inc = np.concatenate(
            [
                np.array([[1] + [0] * (len(G.nodes) - 1)]).T,
                nx.incidence_matrix(G, oriented=True, nodelist=self.node_order, edgelist=self.edge_order).toarray(),
                np.array([[0] * (len(G.nodes) - 1) + [-1]]).T,
            ],
            axis=1,
        )
        E_inc_in = np.where(E_inc > 0, E_inc, 0)
        E_inc_out = np.where(E_inc < 0, -E_inc, 0)

        p_ins = []
        p_outs = []

        for i, node in enumerate(traversal_order):

            p_in = np.zeros((int(np.sum(E_inc_in[i, :])), E_inc.shape[1]))
            in_inds = np.where(E_inc_in[i, :] == 1)[0]
            for j in range(len(in_inds)):
                p_in[j, in_inds[j]] = 1
            p_ins.append(p_in)

            p_out = np.zeros((int(np.sum(E_inc_out[i, :])), E_inc.shape[1]))
            out_inds = np.where(E_inc_out[i, :] == 1)[0]
            for j in range(len(out_inds)):
                p_out[j, out_inds[j]] = 1
            p_outs.append(p_out)

        P_in = np.concatenate(p_ins, axis=0)
        P_out = np.concatenate(p_outs, axis=0)

        def get_index(label_list, substring):
            return np.array(
                [
                    [
                        i
                        for i in range(len(label_list))
                        if label_list[i].startswith(substring)
                    ]
                ]
            )

        # coupling outputs
        yco_index = get_index(labels["p"], "yco")

        # coupling disturbances
        dco_index = get_index(labels["o"], "dco")

        # coupling edges
        e_co = np.arange(1, len(G.edges) + 1, 1)[None, :]

        M_yco_dco = P_out[yco_index.T, e_co] @ np.linalg.inv(P_in[dco_index.T, e_co])
        # y_co  = M_yco_dco @ d_co
        self.M_yco_dco = M_yco_dco

        M_dco_yco = P_in[dco_index.T, e_co] @ np.linalg.inv(P_out[yco_index.T, e_co])
        # d_co = M_dco_yco @ yco
        self.M_dco_yco = M_dco_yco

        if False:
            fig, ax = plt.subplots(1, 2, layout="constrained")
            ax[0].imshow(Fcoco)
            ax[1].imshow(M_yco_dco)




        MFi = np.linalg.inv(M_yco_dco - Fcoco)

        uncoupled_mat = [
            [A, Bct, Bsp, Eex],
            [Cco, Dcoct, Dcosp, Fcoex],
            [Cex, Dexct, Dexsp, Fexex],
            [Cze, Dzect, Dzesp, Fzeex],
            [Cgt, Dgtct, Dgtsp, Fgtex],
            [Cet, Detct, Detsp, Fetex],
        ]

        coupling_mat = [
            [Eco @ MFi @ Cco, Eco @ MFi @ Dcoct, Eco @ MFi @ Dcosp, Eco @ MFi @ Fcoex],
            [
                Fcoco @ MFi @ Cco,
                Fcoco @ MFi @ Dcoct,
                Fcoco @ MFi @ Dcosp,
                Fcoco @ MFi @ Fcoex,
            ],
            [
                Fexco @ MFi @ Cco,
                Fexco @ MFi @ Dcoct,
                Fexco @ MFi @ Dcosp,
                Fexco @ MFi @ Fcoex,
            ],
            [
                Fzeco @ MFi @ Cco,
                Fzeco @ MFi @ Dcoct,
                Fzeco @ MFi @ Dcosp,
                Fzeco @ MFi @ Fcoex,
            ],
            [
                Fgtco @ MFi @ Cco,
                Fgtco @ MFi @ Dcoct,
                Fgtco @ MFi @ Dcosp,
                Fgtco @ MFi @ Fcoex,
            ],
            [
                Fetco @ MFi @ Cco,
                Fetco @ MFi @ Dcoct,
                Fetco @ MFi @ Dcosp,
                Fetco @ MFi @ Fcoex,
            ],
        ]

        combined_mat = [
            [
                uncoupled_mat[i][j] + coupling_mat[i][j]
                for j in range(len(uncoupled_mat[i]))
            ]
            for i in range(len(uncoupled_mat))
        ]

        self.print_block_matrices(
            combined_mat,
            in_labels=["x", "uct", "usp", "dex"],
            out_labels=["x+", "yco", "yex", "yze", "ygt", "yet"],
        )

        self.A, self.Bct, self.Bsp, self.Eex = combined_mat[0]
        self.Cco, self.Dcoct, self.Dcosp, self.Fcoex = combined_mat[1]
        self.Cex, self.Dexct, self.Dexsp, self.Fexex = combined_mat[2]
        self.Cze, self.Dzect, self.Dzesp, self.Fzeex = combined_mat[3]
        self.Cgt, self.Dgtct, self.Dgtsp, self.Fgtex = combined_mat[4]
        self.Cet, self.Detct, self.Detsp, self.Fetex = combined_mat[5]

        self.E_inc = E_inc
        self.P_in = P_in
        self.P_out = P_out


        for key in bounds.keys():
            bounds[key] = np.concatenate(bounds[key])

        self.bounds = bounds
        self.uct_order = uct_order
        self.usp_order = usp_order

        []

    def build_control_model(self, traversal_order, simulation_graph):

        G = simulation_graph

        self.collect_system_matrices(traversal_order, G)

        A_list = []
        B_list = []
        C_list = []
        D_list = []
        E_list = []
        F_list = []

        Bct_list = []
        Bsp_list = []

        Csp_list = []
        Cze_list = []

        Dspct_list = []
        Dspsp_list = []
        Dzect_list = []
        Dzesp_list = []

        Fsp_list = []
        Fze_list = []

        n_list = []
        mct_list = []
        msp_list = []
        p_list = []
        pex_list = []
        pco_list = []
        pze_list = []
        o_list = []
        oco_list = []
        oex_list = []

        x_list = []
        uct_list = []
        usp_list = []

        uct_order = {}
        uct_index = 0
        usp_order = {}
        usp_index = 0

        d_list = []
        dex_list = []
        dco_list = []

        y_list = []
        yex_list = []
        yco_list = []
        yze_list = []

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
                usp_degree = out_degree
            else:
                usp_degree = 0

            n = cm.A.shape[0]

            mct = cm.B.shape[1]
            msp = usp_degree
            o = cm.E.shape[1]

            if usp_degree == 0:
                psp = cm.C.shape[0]
                pze = 0
            else:
                psp = usp_degree
                pze = cm.C.shape[0]

            A_list.append(cm.A)
            B_list.append(cm.B)
            C_list.append(cm.C)
            D_list.append(cm.D)
            # E_list.append(cm.E)
            F_list.append(np.tile(cm.F, in_degree))

            # First row
            # Already have A matrix
            Bct_list.append(cm.B)
            Bsp_list.append(np.zeros((n, msp)))
            E_list.append(np.tile(cm.E, in_degree))

            # Second row
            if msp == 0:  # y_split is just normal y
                Csp_list.append(cm.C)
                Dspct_list.append(cm.D)
                Dspsp_list.append(np.zeros((psp, msp)))

                # NOTE this is sketchy

                if node == "heat_exchanger":

                    outputs_heat = ["thermal_energy_storage"]
                    outputs_power = ["generation", "battery"]
                    outputs_h2 = ["electrolyzer", "hydrogen_storage"]

                    F_temp = []

                    for edge in list(G.in_edges(node)):

                        if (edge[0] in outputs_heat) or ((edge[0] in outputs_power)):
                            F_temp.append(np.array([[0, 1]]).T)
                        elif edge[0] in outputs_h2:
                            F_temp.append(np.array([[1, 0]]).T)

                    Fsp_list.append(np.concatenate(F_temp, axis=1))

                else:
                    Fsp_list.append(np.tile(cm.F, in_degree))

            else:  # no dimension here
                Csp_list.append(np.zeros((psp, n)))
                Dspct_list.append(np.zeros((psp, mct)))
                Dspsp_list.append(np.eye(msp))
                Fsp_list.append(np.zeros((psp, in_degree * cm.F.shape[1])))

            # Third row
            if msp == 0:  # y_split is Ds us
                Cze_list.append(np.zeros((pze, n)))
                Dzect_list.append(np.zeros((pze, mct)))
                Dzesp_list.append(np.zeros((pze, msp)))
                # if node == "heat_exchanger":
                #     Fze_list.append(np.zeros((pze, in_degree)))
                # else:
                Fze_list.append(np.zeros((pze, in_degree * cm.F.shape[1])))
            else:  # yzero  is constrained here
                Cze_list.append(cm.C)
                Dzect_list.append(cm.D)
                Dzesp_list.append(-np.tile(np.eye(pze), msp) @ Dspsp_list[-1])
                Fze_list.append(np.tile(cm.F, in_degree))

            # Should break if the control model has inconsistent dimensions
            subsystem_block = np.block([[cm.A, cm.B, cm.E], [cm.C, cm.D, cm.F]])

            # Dimensions
            n_list.append(n)
            mct_list.append(mct)
            msp_list.append(msp)
            p_list.append(psp + pze)
            o_list.append(o)
            pze_list.append(cm.C.shape[0])

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

            uctl = []
            for i in range(cm.B.shape[1]):
                uctl.append(f"uct {i} {node}")

            uct_list.append(uctl)

            uct_order.update({node: np.arange(uct_index, uct_index + mct, 1)})
            uct_index += mct

            usp_order.update({node: np.arange(usp_index, usp_index + msp, 1)})
            usp_index += msp

            dsl = []
            if G.nodes[node]["is_source"]:  # d = d_ex
                for i in range(cm.E.shape[1] * in_degree):
                    dsl.append(f"dex {i} {node} (from source)")
                dex_list.append(dsl)
            else:  # d = d_co
                for i in range(in_degree):
                    in_edges = list(G.in_edges(node))
                    dsl.append(f"dco {i} {node} (from {in_edges[i][0]})")
                dco_list.append(dsl)

            d_list.append(dsl)

            uspl = []
            yspl = []
            if G.nodes[node]["is_sink"]:  # ys = y_ex
                for i in range(usp_degree):
                    uspl.append(f"usp {i} {node} (to sink)")

                for i in range(out_degree):
                    yspl.append(f"yex {i} {node} (to sink)")

                yex_list.append(yspl)

            else:  # ys = y_co
                for i in range(usp_degree):
                    out_edges = list(G.out_edges(node))
                    uspl.append(f"usp {i} {node} (to {out_edges[i][1]})")

                for i in range(out_degree):
                    out_edges = list(G.out_edges(node))
                    yspl.append(f"yco {i} {node} (to {out_edges[i][1]})")

                if node == "heat_exchanger":
                    yspl.append(f"yex {i} {node} (to constraint)")
                    yex_list.append([f"yex {i} {node} (to constraint)"])

                yco_list.append(yspl)
            usp_list.append(uspl)
            y_list.append(yspl)

            yzel = []
            for i in range(pze):
                yzel.append(f"yze {i} {node}")
            yze_list.append(yzel)

        self.x_list = [x for xs in x_list for x in xs]
        self.uct_list = [x for xs in uct_list for x in xs]
        self.usp_list = [x for xs in usp_list for x in xs]
        self.d_list = [x for xs in d_list for x in xs]
        self.dco_list = [x for xs in dco_list for x in xs]
        self.dex_list = [x for xs in dex_list for x in xs]

        self.y_list = [x for xs in y_list + yze_list for x in xs]
        self.yex_list = [x for xs in yex_list for x in xs]
        self.yco_list = [x for xs in yco_list for x in xs]
        self.yze_list = [x for xs in yze_list for x in xs]

        self.uct_order = uct_order
        self.usp_order = usp_order

        # First row
        A = scipy.linalg.block_diag(*A_list)
        Bct = scipy.linalg.block_diag(*Bct_list)
        Bsp = scipy.linalg.block_diag(*Bsp_list)
        E = scipy.linalg.block_diag(*E_list)

        # Second row
        Csp = scipy.linalg.block_diag(*Csp_list)
        Dspct = scipy.linalg.block_diag(*Dspct_list)
        Dspsp = scipy.linalg.block_diag(*Dspsp_list)
        Fsp = scipy.linalg.block_diag(*Fsp_list)

        # Third row
        Cze = scipy.linalg.block_diag(*Cze_list)
        Dzect = scipy.linalg.block_diag(*Dzect_list)
        Dzesp = scipy.linalg.block_diag(*Dzesp_list)
        Fze = scipy.linalg.block_diag(*Fze_list)

        # others
        B = scipy.linalg.block_diag(*(b for b in B_list))
        D = scipy.linalg.block_diag(*(d for d in D_list))
        E = scipy.linalg.block_diag(*(e for e in E_list))
        F = scipy.linalg.block_diag(*(f for f in F_list))

        assert len(self.d_list) == Fsp.shape[1]

        # self.system_row1 = np.block([[A, Bc, Bs, E]])
        # self.system_row2 = np.block([[Cs, Dsc, Dss, Fs]])
        # self.system_row3 = np.block([[Co, Doc, Dos, Fo]])

        # self.system = np.block(
        #     [[self.system_row1], [self.system_row2], [self.system_row3]]
        # )

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

        # uc_order = {}

        # for i in range(len(self.uc_list)):
        #     node = self.uc_list[i].split(" ")[-1]
        #     if node not in uc_order:
        #         uc_order.update({node: []})

        #     uc_order[node].append(i)

        # self.uc_order = uc_order

        # us_order = {}
        # for i in range(len(self.us_list)):
        #     node = self.us_list[i].split(" ")[-3]
        #     if node not in us_order:
        #         us_order.update({node: []})

        #     us_order[node].append(i)

        # self.us_order = us_order

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

        self.mct = len(self.uct_list)
        self.msp = len(self.usp_list)

        self.p = len(self.y_list)
        self.pco = len(self.yco_list)
        self.pex = len(self.yex_list)
        # self.pex = len([ys for ys in self.ys_list if ys.startswith("yse")])
        self.pze = len(self.yze_list)
        self.o = len(self.d_list)
        self.oco = len(self.dco_list)
        self.oex = len(self.dex_list)
        # self.oex = len([d for d in self.d_list if d.startswith("de")])

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
        B_block = np.block([Bct, Bsp])
        E_block = E

        C_block = np.block([[Csp], [Cze]])
        D_block = np.block([[Dspct, Dspsp], [Dzect, Dzesp]])
        F_block = np.block([[Fsp], [Fze]])

        def get_index(label_list, substring):
            return np.array(
                [
                    [
                        i
                        for i in range(len(label_list))
                        if label_list[i].startswith(substring)
                    ]
                ]
            )

        # uct_index = np.array(
        #     [[i for i in range(len(u_list)) if (u_list)[i].startswith("uc")]]
        # )
        uct_index = get_index(self.uct_list + self.usp_list, "uct")
        usp_index = get_index(self.uct_list + self.usp_list, "usp")

        dex_index = get_index(self.d_list, "dex")
        dco_index = get_index(self.d_list, "dco")

        yex_index = get_index(self.y_list, "yex")
        yco_index = get_index(self.y_list, "yco")
        yze_index = get_index(self.y_list, "yze")

        # usp_index = np.array(
        #     [
        #         [
        #             i
        #             for i in range(len(self.uc_list + self.us_list))
        #             if (self.uc_list + self.us_list)[i].startswith("us")
        #         ]
        #     ]
        # )

        # d_ex = np.array(
        #     [[i for i in range(len(self.d_list)) if self.d_list[i].startswith("de")]]
        # )
        # d_co = np.array(
        #     [[i for i in range(len(self.d_list)) if self.d_list[i].startswith("dc")]]
        # )

        # y_spex = np.array(
        #     [[i for i in range(len(self.ys_list)) if self.ys_list[i].startswith("yse")]]
        # )
        # y_co = np.array(
        #     [[i for i in range(len(self.ys_list)) if self.ys_list[i].startswith("ysc")]]
        # )
        # y_zero = np.array(
        #     [
        #         [
        #             i
        #             for i in range(len(self.ys_list + self.yo_list))
        #             if (self.ys_list + self.yo_list)[i].startswith("yo")
        #         ]
        #     ]
        # )

        e_co = np.arange(1, len(G.edges) + 1, 1)[None, :]

        # P in calculations still feel fragile. Relies on assumption about the order of edges in the graph. Probably fine but may throw an error some time.

        # TODO: look for a graph configuration where P_out and P_in are not almost Identity then see how weird the M matrix can get.
        M_yco_dco = P_out[yco_index.T, e_co] @ np.linalg.inv(P_in[dco_index.T, e_co])
        # y_co  = M_yco_dco @ d_co
        self.M_yco_dco = M_yco_dco

        M_dco_yco = P_in[dco_index.T, e_co] @ np.linalg.inv(P_out[yco_index.T, e_co])
        # d_co = M_dco_yco @ yco
        self.M_dco_yco = M_dco_yco

        # M = P_out[y_co.T, d_co] @ P_in[d_co.T, d_co]

        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(M_yco_dco)
        # ax[1].imshow(M_dco_yco)

        combined_block, combined_mat = self.reorder_coupling(
            A=A_block,
            B=B_block,
            C=C_block,
            D=D_block,
            E=E_block,
            F=F_block,
            M=M_yco_dco,
            uc=uct_index,
            us=usp_index,
            de=dex_index,
            dc=dco_index,
            ye=yex_index,
            yz=yze_index,
            yc=yco_index,
        )

        self.combined_block = combined_block

        self.A, self.Bct, self.Bsp, self.E = combined_mat[0]
        self.Cex, self.Dexct, self.Dexsp, self.Fex = combined_mat[1]
        self.Cze, self.Dzect, self.Dzesp, self.Fze = combined_mat[2]
        self.Cco, self.Dcoct, self.Dcosp, self.Fco = combined_mat[3]

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
            "max_iter": 1000,
        }
        opti.solver(
            "ipopt",
            p_opts,
            s_opts,
        )

        # Variables
        uct_var = opti.variable(self.mct, self.horizon)
        usp_var = opti.variable(self.msp, self.horizon)
        x_var = opti.variable(self.n, self.horizon + 1)
        yex_var = opti.variable(self.pex, self.horizon)
        # e_var = opti.variable(self.q + 2, self.horizon)
        if self.allow_curtail_forecast:
            curtail = opti.variable(self.oex, self.horizon)
        else:
            curtail = opti.parameter(self.oex, self.horizon)

        # Parameters
        dex_param = opti.parameter(self.oex, self.horizon)
        # e_src_param = opti.parameter(1, self.horizon)
        x0_param = opti.parameter(self.n, 1)

        # Initial conditions and forecasted disturbance
        opti.subject_to(x_var[:, 0] == x0_param)
        # opti.subject_to(e_var[0, :] == e_src_param)

        objective = 0

        # Dynamics constraint
        for i in range(self.horizon):

            xkp1 = (
                self.A @ x_var[:, i]
                + self.Bct @ uct_var[:, i]
                + self.Bsp @ usp_var[:, i]
                + self.Eex @ (dex_param[:, i] - curtail[:, i])
            )
            # external outputs
            yexk = (
                self.Cex @ x_var[:, i]
                + self.Dexct @ uct_var[:, i]
                + self.Dexsp @ usp_var[:, i]
                + self.Fexex @ (dex_param[:, i] - curtail[:, i])
            )

            # coupling outputs
            yco = (
                self.Cco @ x_var[:, i]
                + self.Dcoct @ uct_var[:, i]
                + self.Dcosp @ usp_var[:, i]
                + self.Fcoex @ (dex_param[:, i] - curtail[:, i])
            )

            # Splitting constraint zero outputs
            yze = (
                self.Cze @ x_var[:, i]
                + self.Dzect @ uct_var[:, i]
                + self.Dzesp @ usp_var[:, i]
                + self.Fzeex @ (dex_param[:, i] - curtail[:, i])
            )

            # greater than 0 constraint outputs
            ygt = (
                self.Cgt @ x_var[:, i]
                + self.Dgtct @ uct_var[:, i]
                + self.Dgtsp @ usp_var[:, i]
                + self.Fgtex @ (dex_param[:, i] - curtail[:, i])
            )         

            # equal to 0 constraint outputs
            yet = (
                self.Cet @ x_var[:, i]
                + self.Detct @ uct_var[:, i]
                + self.Detsp @ usp_var[:, i]
                + self.Fetex @ (dex_param[:, i] - curtail[:, i])
            )            

            # opti.subject_to(
            #     np.zeros((self.pze, 1))
            #     == self.Cze @ x_var[:, i]
            #     + self.Dzect @ uct_var[:, i]
            #     + self.Dzesp @ usp_var[:, i]
            #     + self.Fzeex @ (dex_param[:, i] - curtail[:, i])
            # )

            # NOTE Sketchy to have these values hard coded
            # opti.subject_to(yexk[1] == 2 * yco[6])

            opti.subject_to(x_var[:, i + 1] == xkp1)
            opti.subject_to(yex_var[:, i] == yexk[0])
            if self.pze > 0:
                opti.subject_to(yze == np.zeros((self.pze, 1)))
            if self.pgt > 0:
                opti.subject_to(ygt >= np.zeros((self.pgt, 1)))
            if self.pet > 0:
                opti.subject_to(yet == np.zeros((self.pet, 1)))

            if self.use_sparsity_constraint:
                opti.subject_to(
                    usp_var[2, i] == ca.if_else(uct_var[1, i] >= 0, uct_var[1, i], 0)
                )
                opti.subject_to(
                    usp_var[0, i] == ca.if_else(uct_var[0, i] >= 0, uct_var[0, i], 0)
                )



            objective += self.objective_step(x_var[:, i], uct_var[:, i], usp_var[:, i], yco, yexk, curtail[:, i])


            # opti.subject_to(us_var[2, i] <= ca.fabs(uc_var[1,i]))

            opti.subject_to(usp_var[:, i] >= np.zeros(self.msp))
            # opti.subject_to(e_var[:, i] >= np.zeros(self.q + 2))

            opti.subject_to(uct_var[:, i] >= self.bounds["u_lb"])
            opti.subject_to(uct_var[:, i] <= self.bounds["u_ub"])

            opti.subject_to(x_var[:, i] >= self.bounds["x_lb"])
            opti.subject_to(x_var[:, i] <= self.bounds["x_ub"])
            # opti.subject_to(uct_var[:, i] >= self.u_lb_list)
            # opti.subject_to(uct_var[:, i] <= self.u_ub_list)

            # opti.subject_to(x_var[:, i] >= self.x_lb_list)
            # opti.subject_to(x_var[:, i] <= self.x_ub_list)

            # opti.subject_to((self.Ds_flat @ ys_var[:, i]) >= self.y_lb_list)
            # opti.subject_to((self.Ds_flat @ ys_var[:, i]) <= self.y_ub_list)

            # trickier constraints - maybe wont work

        if self.allow_curtail_forecast:
            opti.subject_to(curtail <= dex_param)
            opti.subject_to(curtail >= np.zeros(curtail.shape))

        # Bounds
        # Add constraint
        # Objective
        # opti.minimize(self.objective(x_var, uct_var, usp_var, yex_var, curtail))
        opti.minimize(objective)

        self.opti = opti
        self.opt_vars = {
            "uct": uct_var,
            "usp": usp_var,
            "x": x_var,
            "yex": yex_var,
            # "e": e_var,
        }
        self.opt_params = {"dex": dex_param, "x0": x0_param}

        if self.allow_curtail_forecast:
            self.opt_vars.update({"curtail": curtail})
        else:
            self.opt_params.update({"curtail": curtail})

    def objective_step(self, x, uct, usp, yco, yex, curtail):
        ref_steel = 45.48e3
        self.reference = ref_steel
        tracking_term = (ref_steel - yex) ** 2

        BES_local_curtail = (yco[0]) - uct[0]

        BES_simultaneous = uct[0] * uct[1]


        obj_value = tracking_term + BES_simultaneous + BES_local_curtail
        return obj_value

    def objective(self, x, uc, us, ys, curtail):

        ref_steel = 45.48e3
        # ref_steel = 37.92e3
        # ref_steel = 20e3
        self.reference = ref_steel

        bes_state_reference = 1200000
        h2s_state_reference = 320467

        objective_value = 0
        for i in range(self.horizon):

            # ysp = (
            #     self.Csp @ x[:, i] + self.Dspc @ uc[:, i] + self.Dsps @ us[:, i]
            # )  # + self.Fsp @ de

            tracking_term = (ref_steel - ys[0, i]) ** 2

            # h2s_sparsity = 1e-5 * (ysp[3] ** 2 + ysp[5] ** 2) ** 2
            # h2s_sparsity = 1e-5 * (ysp[3] * ysp[5] ) **2
            # h2s_sparsity = 1e3 * ((ysp[3] +  ysp[5]) + ca.fabs(uc[1,i]) )
            # h2s_sparsity = us[3,i] ** 2 - uc[1, i]**2
            # h2s_sparsity = 1e3 * ca.if_else(
            #     uc[1, i] >= 0, (uc[1, i] - us[3, i]) ** 2, 0
            # )
            # h2s_sparsity = (2 * us[3, i] - uc[1, i] - ca.fabs(uc[1, i]))
            # h2s_sparsity = 1e3 * (ysp[3] -  uc[1,i] ) **2
            # h2s_sparsity = 1e3 * (ysp[3] + ysp[5]) **2
            # bes_sparsity = 1e-5 * (ysp[0] ** 2 + ysp[2] ** 2) ** 2
            # bes_sparsity = 1e-5 * (ysp[0] * ysp[2]) ** 2

            # no_h2s_charge = 1e3 * uc[1, i] ** 2

            # bes_state = 1e-5 * (x[0, i] - bes_state_reference) ** 2
            # h2s_state = 1e-3 * (x[1, i] - h2s_state_reference) ** 2

            # curtail_penalty = 1e-3 * curtail[0, i] ** 2
            # storage_agreement = 1e-3 * (0.0218 * uc[0, i] - uc[1, i]) ** 2

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
        self.opti.set_value(self.opt_params["dex"], src_forecast)
        self.opti.set_value(self.opt_params["x0"], x0)
        if not self.allow_curtail_forecast:
            self.opti.set_value(
                self.opt_params["curtail"], np.zeros(self.opt_params["curtail"].shape)
            )

    def update_optimization_constraints(self):
        pass

    def compute_trajectory(self, x0, forecast, step_index=0):
        self.update_optimization_parameters(x0, forecast)

        if hasattr(self, "x_init"):  # then try to update initial guess
            self.opti.set_initial(self.opt_vars["uct"], self.uc_init)
            self.opti.set_initial(self.opt_vars["usp"], self.us_init)
            self.opti.set_initial(self.opt_vars["x"], self.x_init)
            self.opti.set_initial(self.opt_vars["yex"], self.ys_init)

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

            x_db = self.opti.debug.value(self.opt_vars["x"]) # [None, :]
            uc_db = self.opti.debug.value(self.opt_vars["uct"]) #[None, :]
            us_db = self.opti.debug.value(self.opt_vars["usp"])
            ys_db = self.opti.debug.value(self.opt_vars["yex"])  [None, :]

            fig, ax = plt.subplots(
                np.max([uc_db.shape[0], us_db.shape[0], x_db.shape[0], ys_db.shape[0]]),
                4,
                sharex="all",
                layout="constrained",
            )

            to_plot = [x_db, uc_db, us_db, ys_db]
            titles = ["x", "uct", "usp", "yex"]
            titles = [self.n_label, self.mct_label, self.msp_label, self.pex_label]
            for i in range(len(to_plot)):
                # ax[0, i].set_title(titles[i])
                for j in range(len(to_plot[i])):
                    ax[j, i].plot(to_plot[i][j, :])
                    ax[j, i].set_title(titles[i][j])

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

            uc_slice = slice(0, self.mct * self.horizon)
            us_slice = slice(self.mct * self.horizon, (self.mct + self.msp) * self.horizon)
            x_slice = slice(
                (self.mct + self.msp) * self.horizon,
                (self.mct + self.msp) * self.horizon + self.n * (self.horizon + 1),
            )
            ys_slice = slice(
                (self.mct + self.msp) * self.horizon + self.n * (self.horizon + 1),
                (self.mct + self.msp) * self.horizon
                + self.n * (self.horizon + 1)
                + self.pse * self.horizon,
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

        uct = sol.value(self.opt_vars["uct"])
        usp = sol.value(self.opt_vars["usp"])
        x = sol.value(self.opt_vars["x"])
        yex = sol.value(self.opt_vars["yex"])
        # e = sol.value(self.opt_vars["e"])
        dex = sol.value(self.opt_params["dex"])[None, :]
        if self.allow_curtail_forecast:
            curtail = sol.value(self.opt_vars["curtail"])
        else:
            curtail = sol.value(self.opt_params["curtail"])

        self.curtail_storage[step_index : step_index + self.horizon] = curtail

        self.uc_init = uct
        self.us_init = usp
        self.x_init = x
        self.ys_init = yex
        self.curtail_init = curtail

        ysp = (
            self.Cco @ np.atleast_2d(x)[:, :-1]
            + self.Dcoct @ np.atleast_2d(uct)
            + self.Dcosp @ usp
            + self.Fcoex @ (dex - curtail)
        )
        # coupling disturbances
        dco = self.M_dco_yco @ ysp

        ysp = np.concatenate([ysp, np.atleast_2d(yex)])
        # ysp = np.concatenate([ysp, ys[None, :]])

        self.store_solution(
            step_index=step_index,
            uc=uct,
            us=usp,
            x=x,
            yex=yex,
            ysp=ysp,
            curtail=curtail,
            dex=dex,
            dco=dco,
        )

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

        if self.mct == 1:
            u_ctrl = uct[None, 0]
        else:
            u_ctrl = uct[:, 0]

        if curtail.ndim < 2:
            curtail = curtail[None, :]

        # self.plot_solution(sol, forecast)

        u_split = usp[:, 0]

        return uct, usp, curtail

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
                line2 = " " * (out_label_width + 5 + 4)
                for coli, col_label in enumerate(in_labels):
                    line += f"{col_label}".ljust(row_mat_lens[coli] * num_col_width + 3)
                    for j in range(row_mat_lens[coli]):
                        line2 += f"{j}".ljust(num_col_width)
                    line2 += " " * 3


                print(line)
                print(line2)
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
