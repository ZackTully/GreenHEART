import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

class MPCPlotter:
    def __init__(self):
        pass

    def plot_saved_trajectories(self):

        n_nodes = len(self.node_order)
        fig, ax = plt.subplots(
            n_nodes, 4, sharex="all", layout="constrained", figsize=(15, 10), dpi=100
        )

        ax[0, 0].set_title("Disturbance")
        ax[0, 1].set_title("Control input")
        ax[0, 2].set_title("State")
        ax[0, 3].set_title("Output")
        # ax[0, 4].set_title("Split")

        # for i, node in enumerate(list(RTS.G.nodes)):
        for i, node in enumerate(self.node_order):

            ax[i, 0].set_ylabel("\n".join(node.split("_")))

            # 0 - disturbance, 1 - control input, 2 - states, 3- outputs total, 4 - outputs split

            # dex_inds = [
            #     i for i in range(len(self.oex_label)) if node in self.oex_label[i]
            # ]
            dco_inds = [
                i
                for i in range(len(self.oco_label))
                if node in self.oco_label[i].split(" ")[2]
            ]

            uct_inds = [
                i for i in range(len(self.mct_label)) if node in self.mct_label[i]
            ]
            # usp_inds = [
            #     i
            #     for i in range(len(self.msp_label))
            #     if node in self.msp_label[i].split(" ")[2]
            # ]

            x_inds = [i for i in range(len(self.n_label)) if node in self.n_label[i]]
            y_inds = [
                i
                for i in range(len(self.p_label))
                if node in self.p_label[i].split(" ")[2]
            ]

            colors = ["blue", "orange", "red", "brown", "cyan"]

            def plot_one(ax, stored, inds):
                for j in range(len(self.step_index_store)):
                    t = np.arange(
                        self.step_index_store[j],
                        self.step_index_store[j] + self.horizon,
                    )[None, :]
                    for k in range(len(inds)):
                        if self.horizon == 1:
                            ax.scatter(
                                t * np.ones(len(inds)),
                                stored[j][inds, :],
                                color=colors[k],
                            )
                        else:
                            ax.plot(t.T, stored[j][inds[k], :].T, color=colors[k])

            plot_one(ax[i, 0], self.dco_store, dco_inds)
            plot_one(ax[i, 1], self.uct_store, uct_inds)
            plot_one(
                ax[i, 2], [xst[:, 0 : self.horizon] for xst in self.x_store], x_inds
            )
            plot_one(
                ax[i, 3],
                [np.sum(ysp[y_inds, :], axis=0)[None, :] for ysp in self.ysp_store],
                [0],
            )
            # plot_one(ax[i, 4], self.ysp_store, y_inds)
            # ax[i, 4].set_ylim(ax[i, 3].get_ylim())

            if node == "generation":
                forecast_curtail_grid = [
                    np.concatenate(
                        [
                            self.forecast_store[i],
                            self.forecast_store[i] - self.curtail_store[i],
                            self.forecast_store[i] + self.grid_store[i],
                        ]
                    )
                    # [self.forecast_store[i], np.array(self.forecast_store[i]) - np.array(self.curtail_store[i]), np.array(self.forecast_store[i]) + np.array(self.grid_store[i])]
                    for i in range(len(self.step_index_store))
                ]

                plot_one(ax[i, 0], forecast_curtail_grid, np.array([0, 1, 2]))

            fig.align_ylabels()

        pass


    def plot_solution(self, uct, usp, x, ysp, forecast):

        # uc = sol.value(self.opt_vars["uct"])
        # us = sol.value(self.opt_vars["usp"])
        # x = sol.value(self.opt_vars["x"])
        # ys = sol.value(self.opt_vars["ysp"])[None, :]
        # e = sol.value(self.opt_vars["e"])

        uc = uct
        us = usp
        x = x
        ys = ysp

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


    # def plot_trajectory(self, step_index=None):

    #     idx = -1

    #     fig, ax = plt.subplots(2, 2, sharex="all", layout="constrained")

    #     ax[0, 0].plot(self.forecast_store[idx].T)
    #     ax[0, 0].fill_between(
    #         np.arange(0, self.horizon, 1),
    #         self.forecast_store[idx][0, :],
    #         self.forecast_store[idx][0, :] - self.curtail_store[idx][0, :],
    #     )

    #     gen_split_index = [
    #         i
    #         for i in range(self.msp)
    #         if self.msp_label[i].split(" ")[2] == "generation"
    #     ]
    #     start = np.zeros(self.horizon)
    #     time = np.arange(0, self.horizon, 1)
    #     for k in gen_split_index:
    #         stop = self.usp_store[idx][k, :]
    #         ax[0, 0].fill_between(
    #             time,
    #             start,
    #             stop,
    #             edgecolor=None,
    #             label=self.msp_label[gen_split_index[k]],
    #         )
    #         start += stop

    #     ax[0, 0].legend()

    #     pass

    def plot_trajectory_generic(self, prob: ca.Opti, forecast):

        def get_sol_value(prob: ca.Opti, var):
            val = prob.value(var)
            val = np.reshape(val, var.shape)
            return val

        x_db = get_sol_value(prob, self.opt_vars["x"])
        uc_db = get_sol_value(prob, self.opt_vars["uct"])
        us_db = get_sol_value(prob, self.opt_vars["usp"])
        ys_db = get_sol_value(prob, self.opt_vars["yex"])
        yco_db = get_sol_value(prob, self.opt_vars["yco"])

        gridcurtail = get_sol_value(prob, self.opt_vars["gridcurtail"])
        grid_db = np.where(gridcurtail >= 0, gridcurtail, 0)
        curtail_db = np.where(gridcurtail <= 0, -gridcurtail, 0)

        fig, ax = plt.subplots(
            np.max(
                [
                    uc_db.shape[0],
                    us_db.shape[0],
                    x_db.shape[0],
                    ys_db.shape[0],
                    yco_db.shape[0],
                ]
            ),
            5,
            sharex="all",
            layout="constrained",
        )

        to_plot = [x_db, uc_db, us_db, ys_db, yco_db]
        titles = [
            self.n_label,
            self.mct_label,
            self.msp_label,
            self.pex_label,
            self.pco_label,
        ]
        for i in range(len(to_plot)):
            # ax[0, i].set_title(titles[i])
            for j in range(len(to_plot[i])):
                ax[j, i].plot(to_plot[i][j, :])
                ax[j, i].set_title(titles[i][j])

                if (i == 0) or (i == 1):
                    if i == 0:
                        lb = self.bounds["x_lb"]
                        ub = self.bounds["x_ub"]
                    elif i == 1:
                        lb = self.bounds["u_lb"]
                        ub = self.bounds["u_ub"]

                    ylim = ax[j, i].get_ylim()
                    ax[j, i].axhline(lb[j], color="black", linewidth=0.75)
                    ax[j, i].axhline(ub[j], color="black", linewidth=0.75)
                    ax[j, i].set_ylim(ylim)

        ax[-1, 0].set_title("Forecast and curtail")
        ax[-1, 0].plot(forecast)
        ax[-1, 0].plot(forecast - curtail_db)
        ax[-1, 0].plot(forecast - curtail_db + grid_db)