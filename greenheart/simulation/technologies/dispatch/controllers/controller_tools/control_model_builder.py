import numpy as np
import casadi as ca
import scipy
import networkx as nx

class ControlModelBuilder:
    def __init__(self, mpc):
        self.mpc = mpc

  
    def compute_feasible_initial_values(self, x0, forecast, opti=None, start_index=0):

        partial_traj = opti is not None

        


        r_y = self.mpc.reference
        m_h2_ptls = self.Dexct[0, 5]  # 0.01516 tls per kg h2
        q_pkgh2 = self.Dgtct[3, 5]    # 3.847 kWh heat per kg h2
        p_ptls = self.Dgtct[4, 5] / m_h2_ptls


        eta_el = self.Dzesp[2, 2]



        m_h2_total = r_y / m_h2_ptls
        q_total = m_h2_total * -q_pkgh2
        p_steel_total = p_ptls * -r_y
        p_h2_total = m_h2_total / eta_el


        p_total = np.sum([q_total, p_steel_total, p_h2_total])
        p_total_ptls = np.sum([-q_pkgh2/m_h2_ptls, -p_ptls, 1/m_h2_ptls / eta_el])


        p_available = np.where(forecast >= p_total, p_total, forecast)
        u_curtail = p_available - p_total
        u_curtail = np.where(u_curtail >= 0, u_curtail, 0)


        available_reference = p_available / p_total_ptls

        if partial_traj:

            x_var = ca.DM(np.atleast_2d(opti.value(self.mpc.opt_vars["x"], opti.initial())))
            uct_var = ca.DM(np.atleast_2d(opti.value(self.mpc.opt_vars["uct"], opti.initial())))
            usp_var = ca.DM(np.atleast_2d(opti.value(self.mpc.opt_vars["usp"], opti.initial())))
            yex_var = ca.DM(np.atleast_2d(opti.value(self.mpc.opt_vars["yex"], opti.initial())))
            yco_var = ca.DM(np.atleast_2d(opti.value(self.mpc.opt_vars["yco"], opti.initial())))
            ucur_var = ca.DM(np.atleast_2d(opti.value(self.mpc.opt_vars["gridcurtail"], opti.initial())))


            x_var[:, 0] = x0

        else:

            x_var = ca.DM(np.atleast_2d(np.zeros(self.mpc.opt_vars["x"].shape)))
            uct_var = ca.DM(np.atleast_2d(np.zeros(self.mpc.opt_vars["uct"].shape)))
            usp_var = ca.DM(np.atleast_2d(np.zeros(self.mpc.opt_vars["usp"].shape)))
            yex_var = ca.DM(np.atleast_2d(np.zeros(self.mpc.opt_vars["yex"].shape)))
            yco_var = ca.DM(np.atleast_2d(np.zeros(self.mpc.opt_vars["yco"].shape)))
            ucur_var = ca.DM(np.atleast_2d(np.zeros(self.mpc.opt_vars["gridcurtail"].shape)))

            x_var[:, 0] = x0


        tol = 1e-6

        for k in range(self.mpc.horizon):
            if k < start_index-1:
                continue

            xk = x_var[ :, k]
            dk = np.atleast_2d(forecast[k])
            uctk = np.zeros((self.mct, 1))
            uspk = np.zeros((self.msp, 1))
            ucurk = u_curtail[k]


            # Set control inputs to meet the heuristic reference
            r_yk = available_reference[k]


            # generation
            p_gen2steel = r_yk * -p_ptls
            p_gen2el = r_yk /m_h2_ptls / eta_el
            p_gen2tes = r_yk * -q_pkgh2/m_h2_ptls

            # electrolyzer
            m_el2hx = p_gen2el * eta_el

            # tes
            u_charge_tes = p_gen2tes
            u_discharge_tes = p_gen2tes

            # Apply inputs
            uspk[1] = p_gen2tes
            uspk[2] = p_gen2el
            uspk[3] = p_gen2steel

            uspk[8] = m_el2hx

            uctk[2] = u_charge_tes
            uctk[3] = u_discharge_tes



            xkp1, yexk, yco, yze, ygt, yet = self.step_control_model(xk, uctk, uspk, dk, ucurk)

            # assert np.all(np.abs(yze) <= tol)
            # assert np.all(np.abs(ygt) <= tol)


            x_var[:, k+1] = ca.evalf(xkp1)
            uct_var[:, k] = ca.evalf(uctk)
            usp_var[:, k] = ca.evalf(uspk)
            yex_var[:, k] = ca.evalf(yexk)
            yco_var[:, k] = ca.evalf(yco[self.mpc.yco_ub_ind])
            ucur_var[:, k] = ca.evalf(ucurk)
            # x_var[:, k+1, None] = xkp1
            # uct_var[:, k, None] = uctk
            # usp_var[:, k, None] = uspk
            # yex_var[:, k, None] = yexk
            # yco_var[:, k, None] = yco[self.mpc.yco_ub_ind]
            # ucur_var[:, k, None] = ucurk


        return uct_var, usp_var, x_var, yex_var, yco_var, ucur_var
        # return ca.evalf(uct_var), ca.evalf(usp_var), ca.evalf(x_var), ca.evalf(yex_var), ca.evalf(yco_var), ca.evalf(ucur_var)




    def step_control_model(self, x_var, uct_var, usp_var, dex_param, grid_curtail):

        if self.mpc.use_NL_electrolzyer:
            return self.step_control_model_NL(
                x_var, uct_var, usp_var, dex_param, grid_curtail
            )
        xkp1 = (
            self.A @ x_var
            + self.Bct @ uct_var
            + self.Bsp @ usp_var
            + self.Eex @ (dex_param + grid_curtail)
        )
        # external outputs
        yexk = (
            self.Cex @ x_var
            + self.Dexct @ uct_var
            + self.Dexsp @ usp_var
            + self.Fexex @ (dex_param + grid_curtail)
        )

        # coupling outputs
        yco = (
            self.Cco @ x_var
            + self.Dcoct @ uct_var
            + self.Dcosp @ usp_var
            + self.Fcoex @ (dex_param + grid_curtail)
        )

        # Splitting constraint zero outputs
        yze = (
            self.Cze @ x_var
            + self.Dzect @ uct_var
            + self.Dzesp @ usp_var
            + self.Fzeex @ (dex_param + grid_curtail)
        )

        # greater than 0 constraint outputs
        ygt = (
            self.Cgt @ x_var
            + self.Dgtct @ uct_var
            + self.Dgtsp @ usp_var
            + self.Fgtex @ (dex_param + grid_curtail)
        )

        # equal to 0 constraint outputs
        yet = (
            self.Cet @ x_var
            + self.Detct @ uct_var
            + self.Detsp @ usp_var
            + self.Fetex @ (dex_param + grid_curtail)
        )

        return xkp1, yexk, yco, yze, ygt, yet

    def nonlinear_block(self, X):

        P_el = np.ones((1, 2)) @ X

        # Hacky for electrolyzer only right now
        if self.mpc.NL_EL_order == 1:
            # 1st order fit
            popt = ca.MX(np.array([0.01885931]))

            Y = popt[0] * P_el

        elif self.mpc.NL_EL_order == 2:
            # 2nd order fit
            popt = np.array([-2.28481418e-09, 2.08294629e-02])
            Y = popt[0] * P_el**2 + popt[1] * P_el
        elif self.mpc.NL_EL_order == 3:
            # 3rd order fit
            popt = np.array([1.28840632e-15, -4.33591254e-09, 2.15782895e-02])
            Y = popt[0] * P_el**3 + popt[1] * P_el**2 + popt[2] * P_el

        return Y

    def step_control_model_NL(self, x_var, uct_var, usp_var, dex_param, grid_curtail):

        X_block = ca.vertcat(x_var, uct_var, usp_var, dex_param + grid_curtail)
        X_block_li = X_block[self.cols_li]
        X_block_nl = X_block[self.cols_nl]

        ss_lili = ca.MX(self.block_ss[self.rows_li, self.cols_li])
        ss_linl = ca.MX(self.block_ss[self.rows_li, self.cols_nl])
        ss_nlli = ca.MX(self.block_ss[self.rows_nl, self.cols_li])
        ss_nlnl = ca.MX(self.block_ss[self.rows_nl, self.cols_nl])

        Y_block_li = ss_lili @ X_block_li + ss_linl @ X_block_nl
        if self.mpc.use_NL_electrolzyer:
            Y_block_nl = ss_nlli @ X_block_li + self.nonlinear_block(X_block_nl)
        else:
            Y_block_nl = ss_nlli @ X_block_li + ss_nlnl @ X_block_nl

        Y_block = ca.MX(len(self.rows_li) + len(self.rows_nl), 1)
        Y_block[self.rows_li, :] = Y_block_li
        Y_block[self.rows_nl, :] = Y_block_nl

        # Y_block = self.block_ss @ ca.vertcat(x_var, uct_var, usp_var, dex_param+ grid_curtail)

        row_inds = [self.n, self.pco, self.pex, self.pze, self.pgt, self.pet]
        previous = 0
        y_parts = []
        for rows in row_inds:
            y_parts.append(Y_block[previous : previous + rows])
            previous += rows

        xkp1, yco, yexk, yze, ygt, yet = (
            y_parts[0],
            y_parts[1],
            y_parts[2],
            y_parts[3],
            y_parts[4],
            y_parts[5],
        )
        return xkp1, yexk, yco, yze, ygt, yet

    # def collect_system_matrices(self, traversal_order, G):
    def build_control_model(self, traversal_order, G):
        # =============================================================================
        # ==                                                                         ==
        # ==                     Construct control model                             ==
        # ==                                                                         ==
        # =============================================================================

        self.node_order = self.mpc.node_order
        self.edge_order = self.mpc.edge_order
        

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

        verbose_bounds = {}

        mats1 = {"A": [], "Bct": [], "Bsp": [], "Eco": [], "Eex": []}
        mats2 = {"Cex": [], "Dexct": [], "Dexsp": [], "Fexco": [], "Fexex": []}
        mats3 = {"Cco": [], "Dcoct": [], "Dcosp": [], "Fcoco": [], "Fcoex": []}
        mats4 = {"Cze": [], "Dzect": [], "Dzesp": [], "Fzeco": [], "Fzeex": []}
        mats5 = {"Cgt": [], "Dgtct": [], "Dgtsp": [], "Fgtco": [], "Fgtex": []}
        mats6 = {"Cet": [], "Detct": [], "Detsp": [], "Fetco": [], "Fetex": []}

        uct_order = {}
        usp_order = {}

        # TODO linear and nonlinear columns

        linear_cols = {"x": [], "uct": [], "usp": [], "dco": [], "dex": []}
        nonlinear_cols = {"x": [], "uct": [], "usp": [], "dco": [], "dex": []}
        linear_rows = {"x": [], "yex": [], "yco": [], "yze": [], "ygt": [], "yet": []}
        nonlinear_rows = {
            "x": [],
            "yex": [],
            "yco": [],
            "yze": [],
            "ygt": [],
            "yet": [],
        }

        linear_vars = {}
        nonlinear_vars = {}

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
            x_col_linear = []
            x_row_linear = []
            for i in range(n):

                if cm.x_linear[i]:
                    linear_str = "linear"
                else:
                    linear_str = "nonlinear"
                x_labels.append(f"x {i} {node} {linear_str}")

                x_col_linear.append(cm.x_linear[i])
                x_row_linear.append(cm.x_linear[i])

            mct = cm.B.shape[1]
            msp = usp_degree
            m = mct + msp

            # create controllable input label lists

            uct_indices = []

            uct_labels = []
            uct_col_linear = []
            for i in range(mct):
                uct_indices.append(int(len(uct_labels) + np.sum(dims["dims"]["mct"])))
                uct_labels.append(f"uct {i} {node}")

                uct_col_linear.append(cm.u_linear[i])

            if len(uct_indices) > 0:
                uct_order.update({node: uct_indices})

            usp_indices = []
            usp_labels = []
            usp_col_linear = []
            for i in range(usp_degree):
                usp_indices.append(int(len(usp_labels) + np.sum(dims["dims"]["msp"])))
                out_edges = list(G.out_edges(node))
                usp_labels.append(f"usp {i} {node} (to {out_edges[i][1]})")
                usp_col_linear.append(True)

            if len(usp_indices) > 0:
                usp_order.update({node: usp_indices})

            # create uncontrollable input label lists
            dex_labels = []
            dco_labels = []
            dex_col_linear = []
            dco_col_linear = []

            if G.nodes[node]["is_source"]:
                oex = cm.F.shape[1]
                assert in_degree == 1
                oco = 0

                for i in range(oex):
                    dex_labels.append(f"dex {i} {node}")
                    dex_col_linear.append(cm.d_linear[i])

            else:
                oex = 0
                oco = cm.F.shape[1] * in_degree

                for i in range(in_degree):
                    in_edges = list(G.in_edges(node))
                    dco_labels.append(f"dco {i} {node} (from {in_edges[i][0]})")
                    dco_col_linear.append(cm.d_linear[0])
                    # for j in range(cm.o):
                    #     dco_col_linear.append(cm.d_linear[j])

            o = oex + oco

            # create output label lists

            yex_labels = []
            yco_labels = []
            yze_labels = []
            yet_labels = []
            ygt_labels = []

            yex_row_linear = []
            yco_row_linear = []
            yze_row_linear = []
            yet_row_linear = []
            ygt_row_linear = []

            if G.nodes[node]["is_sink"]:
                pex = cm.C.shape[0]
                for i in range(pex):
                    yex_labels.append(f"yex {i} {node}")
                    yex_row_linear.append(cm.y_linear[i])
            else:
                pex = 0

            if usp_degree > 0:
                # Splitting node so yze constraints are needed
                pze = cm.C.shape[0]
                assert pex == 0, "sink node should not be splitting"
                pco = usp_degree

                for i in range(pze):
                    yze_labels.append(f"yze {i} {node}")
                    yze_row_linear.append(cm.y_linear[i])
            else:
                pze = 0
                pco = cm.C.shape[0] - pex

            if not G.nodes[node]["is_sink"]:
                for i in range(out_degree):
                    out_edges = list(G.out_edges(node))
                    yco_labels.append(f"yco {i} {node} (to {out_edges[i][1]})")
                    # Splitting modification means these output rows are linear
                    yco_row_linear.append(True)

            pet = cm.C_et.shape[0]
            pgt = cm.C_gt.shape[0]

            for i in range(pet):
                yet_labels.append(f"yet {i} {node}")
                # TODO double check if this is always true
                yet_row_linear.append(True)

            for i in range(pgt):
                ygt_labels.append(f"ygt {i} {node}")
                # TODO double check if this is always true
                ygt_row_linear.append(True)

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
                    np.where(
                        np.multiply(   cm.disturbance_permutation, (cm.disturbance_domain * up_node_output_domain))
                        # cm.disturbance_permutation
                        # @ (cm.disturbance_domain * up_node_output_domain)
                        == 1
                    )[0]
                )
                # disturbance_index.append(
                #     np.where(cm.disturbance_domain @ up_node_output_domain == 1)[0]
                # )

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

            linear_col_list = [
                x_col_linear,
                uct_col_linear,
                usp_col_linear,
                dco_col_linear,
                dex_col_linear,
            ]
            linear_row_list = [
                x_row_linear,
                yex_row_linear,
                yco_row_linear,
                yze_row_linear,
                ygt_row_linear,
                yet_row_linear,
            ]

            for i, key in enumerate(linear_cols.keys()):
                linear_cols[key].append(linear_col_list[i])

            for i, key in enumerate(linear_rows.keys()):
                linear_rows[key].append(linear_row_list[i])

            # store bounds from the cm

            bounds["u_lb"].append(cm.u_lb)
            bounds["u_ub"].append(cm.u_ub)
            bounds["x_lb"].append(cm.x_lb)
            bounds["x_ub"].append(cm.x_ub)
            bounds["y_lb"].append(cm.y_lb)
            bounds["y_ub"].append(cm.y_ub)

            verbose_bounds.update(
                {
                    node: {
                        "u_lb": cm.u_lb,
                        "u_ub": cm.u_ub,
                        "x_lb": cm.x_lb,
                        "x_ub": cm.x_ub,
                        "y_lb": cm.y_lb,
                        "y_ub": cm.y_ub,
                    }
                }
            )

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

            []

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

        ss_verbose = np.block(
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
            setattr(self.mpc, key, np.sum(dims[key]))


        for key in labels.keys():
            labels[key] = [x for xs in labels[key] for x in xs]

        labels["m"] = labels["mct"] + labels["msp"]

        # this order comes from assumption baked into the node order
        labels["o"] = labels["oex"] + labels["oco"]
        labels["p"] = labels["pco"] + labels["pex"]
        labels["pcons"] = labels["pze"] + labels["pet"] + labels["pgt"]

        for key in labels.keys():
            setattr(self, f"{key}_label", labels[key])
            setattr(self.mpc, f"{key}_label", labels[key])

        self.labels = labels
        self.dims = dims

        linear_cols_verbose = {}
        linear_rows_verbose = {}

        for key in linear_cols.keys():
            linear_cols_verbose.update(
                {key: [boo for bool_list in linear_cols[key] for boo in bool_list]}
            )

        for key in linear_rows.keys():
            linear_rows_verbose.update(
                {key: [boo for bool_list in linear_rows[key] for boo in bool_list]}
            )

        np.sum([len(linear_cols_verbose[key]) for key in linear_cols_verbose.keys()])
        np.sum([len(linear_rows_verbose[key]) for key in linear_rows_verbose.keys()])

        # Make indices and reduce the order of the verbose statespace

        # extended incidence matrix
        E_inc = np.concatenate(
            [
                np.array([[1] + [0] * (len(G.nodes) - 1)]).T,
                nx.incidence_matrix(
                    G, oriented=True, nodelist=self.mpc.node_order, edgelist=self.mpc.edge_order
                ).toarray(),
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
            [
                Eco @ MFi @ Cco,
                Eco @ MFi @ Dcoct,
                Eco @ MFi @ Dcosp,
                Eco @ MFi @ Fcoex,
            ],
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

        linear_cols_coupled = {}

        # Find new nonlinear columns
        coupling_dict = {
            "x": MFi @ Cco,
            "uct": MFi @ Dcoct,
            "usp": MFi @ Dcosp,
            "dex": MFi @ Fcoex,
        }
        for key in coupling_dict.keys():
            coupled_nl = coupling_dict[key].T @ np.invert(linear_cols_verbose["dco"])
            coupled_linear = np.invert(coupled_nl.astype(bool))

            linear_cols_coupled.update(
                {
                    key: np.invert(
                        np.invert(linear_cols_verbose[key]) + np.invert(coupled_linear)
                    )
                }
            )

            []

        linear_rows_coupled = linear_rows_verbose

        combined_mat = [
            [
                uncoupled_mat[i][j] + coupling_mat[i][j]
                for j in range(len(uncoupled_mat[i]))
            ]
            for i in range(len(uncoupled_mat))
        ]

        self.print_block_matrices(
            [combined_mat[i] for i in [0, 1, 2, 3, 4, 5]],
            in_labels=["x", "uct", "usp", "dex"],
            out_labels=["x+", "yco", "yex", "yze", "ygt", "yet"],
            save_description=True,
        )

        # self.print_block_matrices(
        #     combined_mat,
        #     in_labels=["x", "uct", "usp", "dex"],
        #     out_labels=["x+", "yco", "yex", "yze", "ygt", "yet"]
        # )

        self.A, self.Bct, self.Bsp, self.Eex = combined_mat[0]
        self.Cco, self.Dcoct, self.Dcosp, self.Fcoex = combined_mat[1]
        self.Cex, self.Dexct, self.Dexsp, self.Fexex = combined_mat[2]
        self.Cze, self.Dzect, self.Dzesp, self.Fzeex = combined_mat[3]
        self.Cgt, self.Dgtct, self.Dgtsp, self.Fgtex = combined_mat[4]
        self.Cet, self.Detct, self.Detsp, self.Fetex = combined_mat[5]

        self.block_ss = np.block(combined_mat)

        mat_names = [
            ["A", "Bct", "Bsp", "Eex"],
            ["Cco", "Dcoct", "Dcosp", "Fcoex"],
            ["Cex", "Dexct", "Dexsp", "Fexex"],
            ["Cze", "Dzect", "Dzesp", "Fzeex"],
            ["Cgt", "Dgtct", "Dgtsp", "Fgtex"],
            ["Cet", "Detct", "Detsp", "Fetex"],
        ]

        # TODO apply scaling here

        # self.calculate_minimal_inputs()

        self.E_inc = E_inc
        self.P_in = P_in
        self.P_out = P_out

        for key in bounds.keys():
            bounds[key] = np.concatenate(bounds[key])

        self.bounds = bounds
        self.bounds_verbose = verbose_bounds

        self.mpc.bounds = bounds
        self.mpc.bounds_verbose = verbose_bounds

        # Separate out the yco indices that need to be there and the ones that dont
        yco_ub = []

        self.yco_ub_node_ind = np.where(self.bounds["y_ub"] != np.inf)[0]
        self.mpc.yco_ub_node_ind = np.where(self.bounds["y_ub"] != np.inf)[0]
        for y_ind in np.where(self.bounds["y_ub"] != np.inf)[0]:
            node = self.node_order[y_ind]
            for j in range(self.pco):
                if self.pco_label[j].split(" ")[2] == node:
                    yco_ub.append(j)

        self.yco_ub_ind = np.sort(yco_ub)
        self.mpc.yco_ub_ind = np.sort(yco_ub)
        


        self.uct_order = uct_order
        self.usp_order = usp_order

        self.mpc.uct_order = uct_order
        self.mpc.usp_order = usp_order

        self.linear_cols_dict = linear_cols_coupled
        self.linear_rows_dict = linear_rows_coupled

        cols_li = []
        cols_nl = []
        col_count = 0
        for key in ["x", "uct", "usp", "dex"]:
            for boo in linear_cols_coupled[key]:
                if boo:
                    cols_li.append(col_count)
                else:
                    cols_nl.append(col_count)
                col_count += 1
        self.cols_li = np.array(cols_li)[None, :]
        self.cols_nl = np.array(cols_nl)[None, :]

        rows_li = []
        rows_nl = []
        row_count = 0
        for key in ["x", "yex", "yco", "yze", "ygt", "yet"]:
            for boo in linear_rows_coupled[key]:
                if boo:
                    rows_li.append(row_count)
                else:
                    rows_nl.append(row_count)
                row_count += 1
        self.rows_li = np.array(rows_li)[:, None]
        self.rows_nl = np.array(rows_nl)[:, None]

    def calculate_minimal_inputs(self):

        Cze = np.block([[self.Cze], [self.Cgt], [self.Cet]])
        Dzect = np.block([[self.Dzect], [self.Dgtct], [self.Detct]])
        Dzesp = np.block([[self.Dzesp], [self.Dgtsp], [self.Detsp]])
        Fzeex = np.block([[self.Fzeex], [self.Fgtex], [self.Fetex]])

        n_constraints = Cze.shape[0]
        n_variables = Dzect.shape[1] + Dzesp.shape[1]
        n_inds = n_variables - n_constraints
        n_deps = n_constraints

        uct_inds = np.array([1, 2, 5])
        usp_inds = np.array([1, 3, 5, 6])

        uct_deps = np.array(
            [i for i in range(self.mct) if i not in uct_inds], dtype=int
        )
        usp_deps = np.array(
            [i for i in range(self.msp) if i not in usp_inds], dtype=int
        )

        Dze = np.block([Dzect, Dzesp])
        inds_desired = np.concatenate([uct_inds, usp_inds + self.mct])
        deps_desired = np.concatenate([uct_deps, usp_deps + self.mct])

        # [(i, np.linalg.matrix_rank(Dze[:, np.delete(inds_desired, i)])) for i in range(len(inds_desired))]
        # [(i, np.linalg.matrix_rank(Dze[:, np.delete(deps_desired, i)])) for i in range(len(deps_desired))]

        inds = inds_desired
        # inds = np.array([0, 1, 2, 4, 6, 8, 9, 10, 11, 14, 16, 18])
        # inds = np.array([1,  3,  5,  6,  7, 8, 11, 14])
        deps = np.array(
            [i for i in range(self.mct + self.msp) if i not in inds], dtype=int
        )

        Dze_inv = np.linalg.inv(Dze[:, deps])

        Dze_ind = Dze[:, inds]

        ind_ct = [i for i in inds if i < self.mct]
        ind_sp = [
            i - self.mct for i in inds if (i >= self.mct) and (i < self.mct + self.msp)
        ]

        dep_ct = [i for i in inds if i < self.mct]
        dep_sp = [
            i - self.mct for i in deps if (i >= self.mct) and (i < self.mct + self.msp)
        ]

        ct_labels = [self.mct_label[i] for i in ind_ct]
        sp_labels = [self.msp_label[i] for i in ind_sp]

        ct_labels_dep = [self.mct_label[i] for i in dep_ct]
        sp_labels_dep = [self.msp_label[i] for i in dep_sp]

        Anew = self.A - np.block([self.Bct, self.Bsp])[:, deps] @ Dze_inv @ Cze
        Bnew = (
            np.block([self.Bct, self.Bsp])[:, inds]
            - np.block([self.Bct, self.Bsp])[:, deps] @ Dze_inv @ Dze_ind
        )
        Enew = self.Eex - np.block([self.Bct, self.Bsp])[:, deps] @ Dze_inv @ Fzeex

        Cnew = self.Cex - np.block([self.Dexct, self.Dexsp])[:, deps] @ Dze_inv @ Cze
        Dnew = (
            np.block([self.Dexct, self.Dexsp])[:, inds]
            - np.block([self.Dexct, self.Dexsp])[:, deps] @ Dze_inv @ Dze_ind
        )
        Fnew = (
            self.Fexex - np.block([self.Dexct, self.Dexsp])[:, deps] @ Dze_inv @ Fzeex
        )

        Cdep = Dze_inv @ Cze
        Ddep = Dze_inv @ Dze_ind
        Fdep = Dze_inv @ Fzeex

        mat = [[Anew, Bnew, Enew], [Cnew, Dnew, Fnew], [Cdep, Ddep, Fdep]]

        self.print_block_matrices(
            mat,
            in_labels=["x", "u", "dex"],
            out_labels=["x+", "yex", "udep"],
            save_description=False,
        )
        pprint.pprint(list(zip(range(n_inds), ct_labels + sp_labels)))
        pprint.pprint(list(zip(range(n_deps), ct_labels_dep + sp_labels_dep)))

        []

    def print_block_matrices(
        self, mat, in_labels, out_labels, no_space=False, save_description=False
    ):

        try:
            np.block(mat)
        except:
            AssertionError("bad matrix")

        rounding_tol = -9
        rounded_flag = False

        block_mat = np.block(mat)

        col_widths = np.zeros(block_mat.shape[1], dtype=int)
        for i in range(block_mat.shape[1]):
            col_widths[i] = int(
                np.max(
                    [len(f"{block_mat[j,i]:.4g}") for j in range(block_mat.shape[0])]
                )
                + 2
            )

        # block_cols = block_mat.shape[1]
        # block_rows = block_mat.shape[0]

        out_label_width = int(np.max([len(label) for label in out_labels]))
        num_col_width = 10

        print_str = ""

        for row_num, row_mat in enumerate(mat):
            if not no_space:
                # print("")
                print_str += "\n"

            row_mat_lens = [matr.shape[1] for matr in row_mat]
            if row_num == 0:
                line = " " * (out_label_width + 5)
                # line2 = " " * (out_label_width + 5 + 4)
                line2 = " " * (out_label_width + 3)
                col_count = 0
                for coli, col_label in enumerate(in_labels):
                    label_pad = 0
                    for j in range(row_mat_lens[coli]):
                        # line2 += f"{j}".ljust(num_col_width)
                        line2 += f"{j}".rjust(col_widths[col_count])
                        label_pad += col_widths[col_count]
                        col_count += 1

                    line += f"{col_label}".ljust(label_pad + 2)
                    line2 += " " * 2

                # print(line)
                # print(line2)
                print_str += line + "\n"
                print_str += line2 + "\n"

            n_rows = row_mat[0].shape[0]
            for i in range(n_rows):
                line = f"{out_labels[row_num]}".ljust(out_label_width + 3)
                line += "[ "
                col_count = 0

                for col_mat in row_mat:
                    for j in range(col_mat.shape[1]):
                        if np.abs(col_mat[i, j]) < 10 ** (rounding_tol):
                            num = 0
                            rounded_flag = True
                        else:
                            num = np.round(col_mat[i, j], -rounding_tol)

                        # line += f"{col_mat[i,j] :.4g}, ".rjust(num_col_width)
                        # line += f"{num :.4g}, ".rjust(num_col_width)
                        line += f"{num :.4g}, ".rjust(col_widths[col_count])
                        col_count += 1

                    line = line[0:-2]
                    line += " ][ "
                line = line[0:-2]
                # print(line)
                print_str += line + "\n"

        if rounded_flag:
            print_str += (
                f"some values were lower than 1e{rounding_tol} so they were set to 0\n"
            )

        if save_description:
            self.state_space_string = print_str
        else:
            print(print_str)

        []