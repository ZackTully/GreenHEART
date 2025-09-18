import numpy as np
import matplotlib.pyplot as plt


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
