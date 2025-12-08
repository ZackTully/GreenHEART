import numpy as np
import matplotlib.pyplot as plt
import casadi as ca

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.size": 10,
        "legend.fontsize": 8,
        "figure.titlesize": 12,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.borderpad": 0.2,
        "legend.labelspacing": 0.3,
        "legend.handlelength": 1.25,
        "legend.handletextpad": 0.3,
        "legend.columnspacing": 1.0,
        "axes.axisbelow": True,
    }
)


class MPCPlotter:
    def __init__(self, mpc):
        self.mpc = mpc

    def plot_saved_trajectories(self, n=100):

        n_nodes = len(self.mpc.node_order)
        fig, ax = plt.subplots(
            n_nodes, 4, sharex="all", layout="constrained", figsize=(15, 10), dpi=100
        )

        ax[0, 0].set_title("Disturbance")
        ax[0, 1].set_title("Control input")
        ax[0, 2].set_title("State")
        ax[0, 3].set_title("Output")
        # ax[0, 4].set_title("Split")

        o_lab = self.mpc.oco_label
        mct_lab = self.mpc.mct_label
        n_lab = self.mpc.n_label
        p_lab = self.mpc.p_label

        bound_kwargs = dict(linewidth=0.75, color="black")

        for i, node in enumerate(self.mpc.node_order):

            ax[i, 0].set_ylabel("\n".join(node.split("_")))

            dco_inds = [i for i in range(len(o_lab)) if node in o_lab[i].split(" ")[2]]
            uct_inds = [i for i in range(len(mct_lab)) if node in mct_lab[i]]
            x_inds = [i for i in range(len(n_lab)) if node in n_lab[i]]
            y_inds = [i for i in range(len(p_lab)) if node in p_lab[i].split(" ")[2]]

            colors = ["blue", "orange", "red", "brown", "cyan"]

            def plot_one(ax, stored, inds):
                n_traj = len(self.mpc.step_index_store)
                if n >= n_traj:
                    j_range = range(n_traj)
                else:
                    j_range = range(n_traj - n, n_traj)

                for j in j_range:
                    t = np.arange(
                        self.mpc.step_index_store[j],
                        self.mpc.step_index_store[j] + self.mpc.horizon,
                    )[None, :]
                    for k in range(len(inds)):
                        if self.mpc.horizon == 1:
                            ax.scatter(
                                t * np.ones(len(inds)),
                                stored[j][inds, :],
                                color=colors[k],
                            )
                        else:
                            ax.plot(t.T, stored[j][inds[k], :].T, color=colors[k])

            plot_one(ax[i, 0], self.mpc.dco_store, dco_inds)
            plot_one(ax[i, 1], self.mpc.uct_store, uct_inds)
            # Add in state bounds
            if len(self.mpc.bounds_verbose[node]["x_lb"]) > 0:
                for j in range(len(self.mpc.bounds_verbose[node]["x_lb"])):
                    ax[i, 2].axhline(
                        self.mpc.bounds_verbose[node]["x_lb"][j], **bound_kwargs
                    )
                    ax[i, 2].axhline(
                        self.mpc.bounds_verbose[node]["x_ub"][j], **bound_kwargs
                    )

            plot_one(
                ax[i, 2],
                [xst[:, 0 : self.mpc.horizon] for xst in self.mpc.x_store],
                x_inds,
            )
            plot_one(
                ax[i, 3],
                [np.sum(ysp[y_inds, :], axis=0)[None, :] for ysp in self.mpc.ysp_store],
                [0],
            )
            # plot_one(ax[i, 4], self.ysp_store, y_inds)
            # ax[i, 4].set_ylim(ax[i, 3].get_ylim())

            if node == "generation":
                forecast_curtail_grid = [
                    np.concatenate(
                        [
                            self.mpc.forecast_store[i],
                            self.mpc.forecast_store[i] - self.mpc.curtail_store[i],
                            self.mpc.forecast_store[i] + self.mpc.grid_store[i],
                        ]
                    )
                    # [self.forecast_store[i], np.array(self.forecast_store[i]) - np.array(self.curtail_store[i]), np.array(self.forecast_store[i]) + np.array(self.grid_store[i])]
                    for i in range(len(self.mpc.step_index_store))
                ]

                plot_one(ax[i, 0], forecast_curtail_grid, np.array([0, 1, 2]))

            fig.align_ylabels()

        pass

    def plot_saved_trajectories_paper(self, save_figure=False, save_data=True):

        n_rows = 5
        n_cols = 1
        fig, ax = plt.subplots(
            n_rows, n_cols, sharex="all", layout="constrained", figsize=(4, 5)
        )
        ax = np.atleast_2d(ax).T

        traj_kw = dict(alpha=0.25, linewidth=0.75)
        if self.mpc.horizon == 48:
            scat_kw = dict(marker="D", s=5, label="action")
            # scat_kw = dict(marker="D", s=11, label="action")
        elif self.mpc.horizon == 6:
            # scat_kw = dict(marker="s", s=11, label="action")
            scat_kw = dict(marker="s", s=5, label="action")
        else:
            scat_kw = dict(marker=".", label="action")

        d_kw = dict(color="gray")
        cur_kw = dict(color="orange")
        xb_kw = dict(color="black")
        xq_kw = dict(color="red")
        xh_kw = dict(color="blue")
        y_kw = dict(color="green")

        def get_node_inds(node):
            o_lab = self.mpc.oco_label
            mct_lab = self.mpc.mct_label
            n_lab = self.mpc.n_label
            p_lab = self.mpc.p_label
            dco_inds = [i for i in range(len(o_lab)) if node in o_lab[i].split(" ")[2]]
            uct_inds = [i for i in range(len(mct_lab)) if node in mct_lab[i]]
            x_inds = [i for i in range(len(n_lab)) if node in n_lab[i]]
            y_inds = [i for i in range(len(p_lab)) if node in p_lab[i].split(" ")[2]]

            return dco_inds, uct_inds, x_inds, y_inds

        def draw_bounds(ax, lb, ub, factor=1, omit=""):
            kw = dict(color="black", linewidth=0.5, linestyle="dashed")
            for i in range(len(lb)):
                if not omit == "lower":
                    ax.axhline(lb[i] / factor, **kw)
                if not omit == "upper":
                    ax.axhline(ub[i] / factor, **kw)

        def plot_timeseries(ax, t, y_data, kw={}):
            ax.fill_between(t, y_data, edgecolor="none", **kw)

        def plot_actions(ax, t, y_data, factor=1, kw={}):
            yd = np.array([y[0, 0] for y in y_data])
            ax.scatter(t, yd / factor, **kw)

        def plot_trajectories(ax, t, y_data, factor=1, kw={}):
            # TODO make the trajectory go beyond the last step (like zero order hold behavior)
            for i in range(len(y_data)):
                if i == 0:
                    label = "trajectory"
                else:
                    label = None
                ti = np.arange(t[i], t[i] + y_data[i].shape[1], 1)
                ax.plot(ti, y_data[i][0, :] / factor, **kw, label=label)

        t = np.arange(0, len(self.mpc.de_store), 1)

        disturbance = np.array([de[0, 0] for de in self.mpc.de_store])

        # Disturbance
        plot_timeseries(ax[0, 0], t, disturbance, kw=(d_kw | dict(alpha=0.25)))

        # Forecasted disturbance
        plot_trajectories(ax[0, 0], t, self.mpc.forecast_store, kw=(d_kw | traj_kw))

        # Measured disturbance
        plot_actions(ax[0, 0], t, self.mpc.forecast_store, kw=(d_kw | scat_kw))

        plot_trajectories(ax[0, 0], t, self.mpc.curtail_store, kw = (cur_kw | traj_kw))
        plot_actions(ax[0, 0], t, self.mpc.curtail_store, kw=(cur_kw | traj_kw))


        # Battery state
        dbes, ubes, xbes, ybes = get_node_inds("battery")
        ydat_bes = [xs[xbes, :] for xs in self.mpc.x_store]

        bds_bes = self.mpc.bounds_verbose["battery"]
        draw_bounds(
            ax[1, 0],
            bds_bes["x_lb"],
            bds_bes["x_ub"],
            factor=bds_bes["x_ub"],
            omit="upper",
        )
        plot_trajectories(
            ax[1, 0], t, ydat_bes, factor=bds_bes["x_ub"], kw=(xb_kw | traj_kw)
        )
        plot_actions(
            ax[1, 0], t, ydat_bes, factor=bds_bes["x_ub"], kw=(xb_kw | scat_kw)
        )

        # Thermal energy storage state
        dtes, utes, xtes, ytes = get_node_inds("thermal_energy_storage")
        ydat_tes = [xs[xtes, :] for xs in self.mpc.x_store]

        bds_tes = self.mpc.bounds_verbose["thermal_energy_storage"]
        draw_bounds(
            ax[2, 0],
            [bds_tes["x_lb"][0]],
            [bds_tes["x_ub"][1]],
            factor=bds_tes["x_ub"][0],
            omit="upper",
        )
        plot_trajectories(
            ax[2, 0], t, ydat_tes, factor=bds_tes["x_ub"][0], kw=(xq_kw | traj_kw)
        )
        plot_actions(
            ax[2, 0], t, ydat_tes, factor=bds_tes["x_ub"][0], kw=(xq_kw | scat_kw)
        )

        # Hydrogen storage state
        dh2s, uh2s, xh2s, yh2s = get_node_inds("hydrogen_storage")
        ydat_h2s = [xs[xh2s, :] for xs in self.mpc.x_store]

        bds_h2s = self.mpc.bounds_verbose["hydrogen_storage"]
        draw_bounds(
            ax[3, 0],
            bds_h2s["x_lb"],
            bds_h2s["x_ub"],
            factor=bds_h2s["x_ub"],
            omit="upper",
        )
        plot_trajectories(
            ax[3, 0], t, ydat_h2s, factor=bds_h2s["x_ub"], kw=(xh_kw | traj_kw)
        )
        plot_actions(
            ax[3, 0], t, ydat_h2s, factor=bds_h2s["x_ub"], kw=(xh_kw | scat_kw)
        )

        # Output
        yex_dat = self.mpc.yex_store
        draw_bounds(ax[4, 0], lb=[0], ub=[self.mpc.reference], omit="lower")
        plot_trajectories(ax[4, 0], t, yex_dat, kw=(y_kw | traj_kw))
        plot_actions(ax[4, 0], t, yex_dat, kw=(y_kw | scat_kw))

        ax[-1, 0].set_xlabel("Time [h]")

        ax[0, 0].set_ylabel("Disturbance\n[kWh]")
        ax[1, 0].set_ylabel("BES SOC")
        ax[2, 0].set_ylabel("TES SOC")
        ax[3, 0].set_ylabel("H2S SOC")
        # ax[1, 0].set_ylabel("BES state\n[kWh]")
        # ax[2, 0].set_ylabel("TES state 1\n[kWh]")
        # ax[3, 0].set_ylabel("H2S state\n[kg]")
        ax[4, 0].set_ylabel("Output\n[tonne]")

        ax[0, 0].set_xlim([-5, 105])

        ax[0, 0].ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
        ax[0, 0].set_ylim([0, ax[0, 0].get_ylim()[1]])

        ax[1, 0].set_ylim([-0.05, 1.05])
        ax[2, 0].set_ylim([-0.05, 0.5])
        ax[3, 0].set_ylim([-0.025, 0.25])
        ax[4, 0].set_ylim([0, 130])

        for i in range(ax.shape[0]):
            for j in range(ax.shape[1]):
                ax[i, j].legend()

        fig.align_labels()

        yex = np.array([y[0, 0] for y in yex_dat])
        mpc_RMSE = np.sqrt(np.mean((self.mpc.reference - yex) ** 3))

        fig.suptitle(f"Horizon: {self.mpc.horizon}, RMSE: {mpc_RMSE:.2f}")

        preamble = ""
        if "gridcurtail" in self.mpc.term_keys:
            preamble += "curtailment_"
        if "bes_soc_state" in self.mpc.term_keys:
            preamble += "storage_state_"

        fname = f'syn_data_{preamble}ry{str(self.mpc.reference).replace(".", "p")}_H{self.mpc.horizon}_alpha{str(self.mpc.config.greenheart_config["realtime_simulation"]["forecast"]["method_config"]["perfect_fraction"]).replace(".", "p")}.pdf'
        if save_figure:
            fig.savefig(
                f"/Users/ztully/Documents/hybrids_code/GH_scripts/greenheart_scripts/plots/synthetic_disturbance/{fname}",
                format="pdf",
            )

        data_dict = dict(
            de_store=self.mpc.de_store,
            curtail_store=self.mpc.curtail_store,
            forecast_store=self.mpc.forecast_store,
            x_store=self.mpc.x_store,
            yex_store=self.mpc.yex_store,
            objective_store = self.mpc.objective_store,
            obj_terms = list(self.mpc.obj_terms.keys()),
        )
        if save_data:
            np.savez(
                f"/Users/ztully/Documents/hybrids_code/GH_scripts/greenheart_scripts/minnesota_reference_design/configs/synthetic_generation_profile/weather/synthetic/plot_data/{fname.rstrip('.pdf')}",
                **data_dict,
            )

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

        x_db = get_sol_value(prob, self.mpc.opt_vars["x"])
        uc_db = get_sol_value(prob, self.mpc.opt_vars["uct"])
        us_db = get_sol_value(prob, self.mpc.opt_vars["usp"])
        ys_db = get_sol_value(prob, self.mpc.opt_vars["yex"])
        yco_db = get_sol_value(prob, self.mpc.opt_vars["yco"])

        gridcurtail = get_sol_value(prob, self.mpc.opt_vars["gridcurtail"])
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
            self.mpc.n_label,
            self.mpc.mct_label,
            self.mpc.msp_label,
            self.mpc.pex_label,
            self.mpc.pco_label,
        ]
        for i in range(len(to_plot)):
            # ax[0, i].set_title(titles[i])
            for j in range(len(to_plot[i])):
                ax[j, i].plot(to_plot[i][j, :])
                ax[j, i].set_title(titles[i][j])

                if (i == 0) or (i == 1):
                    if i == 0:
                        lb = self.mpc.bounds["x_lb"]
                        ub = self.mpc.bounds["x_ub"]
                    elif i == 1:
                        lb = self.mpc.bounds["u_lb"]
                        ub = self.mpc.bounds["u_ub"]

                    ylim = ax[j, i].get_ylim()
                    ax[j, i].axhline(lb[j], color="black", linewidth=0.75)
                    ax[j, i].axhline(ub[j], color="black", linewidth=0.75)
                    ax[j, i].set_ylim(ylim)

        ax[-1, 0].set_title("Forecast and curtail")
        ax[-1, 0].plot(forecast)
        ax[-1, 0].plot(forecast - curtail_db)
        ax[-1, 0].plot(forecast - curtail_db + grid_db)
