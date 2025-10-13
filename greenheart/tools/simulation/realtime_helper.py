import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import networkx as nx

class RealTimeSimulationHelper:
    def __init__(self, rts):
        self.rts = rts

    def setup_ctrl_sysid(self):

        # Make a bunch of dicts with 8760 length

        self.rts.sysid = {}

        for label in self.rts.dispatcher.controller.n_label:
            self.rts.sysid.update({label: np.zeros(8760)})

        for label in self.rts.dispatcher.controller.mct_label:
            self.rts.sysid.update({label: np.zeros(8760)})

        for label in self.rts.dispatcher.controller.msp_label:
            self.rts.sysid.update({label: np.zeros(8760)})

        for label in self.rts.dispatcher.controller.oex_label:
            self.rts.sysid.update({label: np.zeros(8760)})

        for label in self.rts.dispatcher.controller.pco_label:
            self.rts.sysid.update({label: np.zeros(8760)})

        for label in self.rts.dispatcher.controller.pex_label:
            self.rts.sysid.update({label: np.zeros(8760)})


    def save_ctrl_for_sysid(self, step_index=None):

        # Think about doing this with the un-simplified system model rather than the simplified one with coupling

        # save x uct usp dex inputs
        # save yco yex yze ygt outputs

        # states
        for label in self.rts.dispatcher.controller.n_label:
            node_name = label.split(" ")[2]
            if node_name == "battery":
                state = self.rts.G.nodes["battery"]["ionode"].model.storage_state
            elif node_name == "thermal_energy_storage":
                state = (
                    self.rts.G.nodes["thermal_energy_storage"]["ionode"].model._SOC()
                    * self.rts.G.nodes["thermal_energy_storage"][
                        "ionode"
                    ].model.H_capacity_kWh
                )
            elif node_name == "hydrogen_storage":
                state = self.rts.G.nodes["hydrogen_storage"]["ionode"].model.storage_state

            self.rts.sysid[label][step_index] = state

        # control inputs

        for label in self.rts.dispatcher.controller.mct_label:
            node_name = label.split(" ")[2]
            uct_index = int(label.split(" ")[1])
            self.rts.sysid[label][step_index] = self.rts.G.nodes[node_name]["dispatch_ctrl"][
                uct_index
            ]

        # splitting inputs

        for label in self.rts.dispatcher.controller.msp_label:
            source_node = label.split(" ")[2]
            sink_node = label[label.find("(") + 1 : label.find(")")].split(" ")[1]

            usp_index = int(label.split(" ")[1])

            self.rts.sysid[label][step_index] = self.rts.G.nodes[source_node]["dispatch_split"][
                usp_index
            ]

        # disturbance

        # Output stuff

        # yco

        for label in self.rts.dispatcher.controller.pco_label:
            source_node = label.split(" ")[2]
            sink_node = label[label.find("(") + 1 : label.find(")")].split(" ")[1]

            output_domain = self.rts.G.nodes[source_node]["ionode"].output_list[0:-1]
            output_index = np.where(output_domain)[0]

            for i, edge in enumerate(self.rts.edge_order):
                if (edge[0] == source_node) and (edge[1] == sink_node):
                    # This is probably the right case then
                    self.rts.sysid[label][step_index] = self.rts.system_states[
                        i, step_index, output_index
                    ]

        # yex
        yex_label = self.rts.dispatcher.controller.pex_label[0]
        yex_node = yex_label.split(" ")[-1]
        self.rts.sysid[yex_label][step_index] = self.rts.G.nodes[yex_node][
            "ionode"
        ].model.steel_store_tonne[step_index]

        # dex
        dex_label = self.rts.dispatcher.controller.oex_label[0]
        self.rts.sysid[dex_label][step_index] = (
            self.rts.hybrid_profile[step_index]
            + self.rts.G.nodes["generation"]["grid_purchase"]
        )



    def plot_system_graph(self):

        fig, ax = plt.subplots(1, 1, layout="constrained")
        ax.set_axis_off()
        G = self.rts.G

        # Make multipartite layout for plotting
        indeg = G.in_degree
        root_node = [node for node in indeg if node[1] == 0]
        node_layers = []

        node_layers.append([])
        if len(root_node) > 1:
            shortest_path_list = []
            for rn in root_node:
                node_layers[0].append(rn[0])
                shortest_path_list.append(
                    nx.single_source_shortest_path(G, source=rn[0])
                )
        else:
            node_layers[0].append(root_node[0][0])

            shortest_paths = nx.single_source_shortest_path(G, source=root_node[0][0])
            shortest_path_list = [shortest_paths]

        for spl in shortest_path_list:
            for key in spl.keys():
                path_length = len(spl[key])
                if len(node_layers) <= path_length:
                    node_layers.append([])

                node_layers[path_length].append(key)
        G_comp = nx.DiGraph()
        for i in range(len(node_layers)):
            G_comp.add_nodes_from(node_layers[i], layer=i)

        G_comp.add_edges_from(G.edges)
        layout = nx.multipartite_layout(
            G_comp, subset_key="layer", align="vertical", scale=1
        )

        for key in layout.keys():
            coords = layout[key]
            coords[0] += np.random.randn(1) * 0.05
            layout[key] = coords

        if hasattr(self, "print_locs"):
            layout = self.rts.print_locs

        nodes = nx.draw_networkx_nodes(G, pos=layout, ax=ax)
        nodes.set_edgecolor("white")
        nodes.set_facecolor("white")
        labels = nx.draw_networkx_labels(G, pos=layout, ax=ax)
        edges = nx.draw_networkx_edges(G, pos=layout, ax=ax)

        # latex_graph = nx.to_latex(G, pos=nx.rescale_layout_dict(layout, scale=3))
        # latex_graph = nx.to_latex(G, pos=nx.rescale_layout(np.array(list({key:np.array(value) for key, value in layout.items()}.values()), dtype=float), scale=3))
        # latex_graph = nx.to_latex(G, pos={key: nx.rescale_layout(np.array(value, dtype=float), scale=3) for key, value in layout.items()})
        latex_graph = nx.to_latex(
            G,
            pos={
                key: np.array(value, dtype=float) * 2 for key, value in layout.items()
            },
        )

        
    def plot_edges(self):
        fig, ax = plt.subplots(
            len(self.edge_order), 1, sharex="col", layout="constrained"
        )
        colors = ["black", "red", "blue"]
        labels = ["power", "heat", "hydrogen"]
        for i in range(len(self.edge_order)):
            for j in range(3):
                ax[i].fill_between(
                    np.arange(0, self.system_states.shape[1], 1),
                    np.zeros(self.system_states.shape[1]),
                    self.system_states[i, :, j],
                    step="post",
                    color=colors[j],
                    label=labels[j],
                )
                ylabel = f"{self.edge_order[i][0][0:4]} to {self.edge_order[i][1][0:4]}"
                ax[i].set_ylabel(ylabel)

            ax[i].set_ylim([0, ax[i].get_ylim()[1]])
            ax[i].legend()

        if self.stop_index < 8760:
            ax[0].set_xlim([0, self.stop_index])

        fig.align_ylabels()

    def plot_nodes(
        self, data="edges", figsize=(15, 8), hide_yaxis=True, fname=None, save=False
    ):
        fig, ax = plt.subplots(
            len(self.rts.node_order),
            1,
            sharex="all",
            sharey="all",
            layout="constrained",
            figsize=figsize,
        )

        ax = ax[:, None]

        if data == "edges":

            # normalized_states = np.zeros(self.rts.system_states.shape)
            # # for i in range(self.rts.system_states.shape[2]):
            # #     normalized_states[:, :, i] = np.nan_to_num(
            # #         self.rts.system_states[:, :, i] / np.max(self.rts.system_states[:, :, i])
            # #     )

            # normalized_states[:, :, 0] = np.nan_to_num(self.rts.system_states[:, :, 0] / np.max(self.rts.system_states[:, :, [0, 1]]))
            # normalized_states[:, :, 1] = np.nan_to_num(self.rts.system_states[:, :, 1] / np.max(self.rts.system_states[:, :, [0, 1]]))
            # normalized_states[:, :, 2] = np.nan_to_num(self.rts.system_states[:, :, 2] / np.max(self.rts.system_states[:, :, 2]))

            normalized_states = np.copy(self.rts.system_states)
            normalized_states[:, :, 2] *= 54

            plot_states = normalized_states
        elif data == "error":
            normalized_states = np.zeros(self.rts.system_states.shape)

            normalized_states[:, :, 0] = np.nan_to_num(
                self.rts.system_states[:, :, 0] / np.max(self.rts.system_states[:, :, [0, 1]])
            )
            normalized_states[:, :, 1] = np.nan_to_num(
                self.rts.system_states[:, :, 1] / np.max(self.rts.system_states[:, :, [0, 1]])
            )
            normalized_states[:, :, 2] = np.nan_to_num(
                self.rts.system_states[:, :, 2] / np.max(self.rts.system_states[:, :, 2])
            )

            plot_states = normalized_states

        colors = ["black", "red", "blue"]
        cmaps = ["Greys", "Reds", "Blues"]
        edgecolors = ["orange", "yellow", "cyan", "magenta", "green"]
        labels = ["P", "Q", "H2"]
        for i in range(len(self.rts.node_order)):
            node = self.rts.node_order[i]

            ylabel = "\n".join(node.split("_"))

            ax[i, 0].set_ylabel(ylabel)

            incoming = [
                self.rts.edge_order[k][0]
                for k in range(len(self.rts.edge_order))
                if node == self.rts.edge_order[k][1]
            ]
            outgoing = [
                self.rts.edge_order[k][1]
                for k in range(len(self.rts.edge_order))
                if node == self.rts.edge_order[k][0]
            ]
            in_index = np.array(
                [
                    k
                    for k in range(len(self.rts.edge_order))
                    if node == self.rts.edge_order[k][1]
                ]
            )
            out_index = np.array(
                [
                    k
                    for k in range(len(self.rts.edge_order))
                    if node == self.rts.edge_order[k][0]
                ]
            )
            for j in range(3):
                # for j in [0]

                j_ax = 0

                n_fills = len(in_index) + len(out_index)
                cmap_level = 0.25

                start = np.zeros(plot_states.shape[1])

                if (node == "generation") and (j == 0):
                    # ax[i, j_ax].step(
                    #     np.arange(0, self.rts.system_states.shape[1], 1),
                    #     -self.rts.hybrid_profile / np.max(self.rts.system_states[:, :, 0]),
                    #     color="black",
                    #     linewidth=1,
                    #     where="post",
                    #     label="Hybrid gen.",
                    # )
                    ax[i, j_ax].fill_between(
                        np.arange(0, self.rts.system_states.shape[1], 1),
                        np.zeros(len(self.rts.hybrid_profile)),
                        self.rts.hybrid_profile,  # / np.max(self.rts.system_states[:, :, 0]),
                        color=mpl.colormaps[cmaps[j]](0.8),
                        linewidth=0,
                        step="post",
                        label="Hybrid gen.",
                    )
                    curtail = self.rts.G.nodes[node]["ionode"].u_curtail_store
                    grid = self.rts.grid_power_store[0, :]
                    # curtail = self.rts.G.nodes[node]["ionode"].u_curtail_store / np.max(
                    #     self.rts.system_states[:, :, 0]
                    # )
                    # grid = (
                    #     self.rts.grid_power_store / np.max(self.rts.system_states[:, :, 0])
                    # )[0, :]
                    ax[i, j_ax].fill_between(
                        np.arange(0, plot_states.shape[1], 1),
                        self.rts.hybrid_profile,  # / np.max(self.rts.system_states[:, :, 0]),
                        self.rts.hybrid_profile
                        + grid,  # / np.max(self.rts.system_states[:, :, 0])
                        # + grid,
                        step="post",
                        alpha=1,
                        linewidth=0,
                        label="grid",
                        color="darkviolet",
                        # color=mpl.colormaps[cmaps[j]](cmap_level + 0.35),
                    )

                    stop = -curtail[:, 0]
                    ax[i, j_ax].fill_between(
                        np.arange(0, plot_states.shape[1], 1),
                        start,
                        start + stop,
                        step="post",
                        alpha=1,
                        linewidth=0,
                        label=f"curtail",
                        # color=mpl.colormaps[cmaps[j]](cmap_level - 0.15),
                        color="orange",
                    )
                    start += stop

                if node == "battery":

                    axt = ax[i, 0].twinx()

                    axt.plot(
                        self.rts.G.nodes["battery"]["ionode"].model.store_storage_state
                        / self.rts.G.nodes["battery"]["ionode"].model.max_capacity_kWh,
                        color="black",
                        linewidth=1,
                        label="BES SOC",
                    )
                    axt.set_ylim([0, 1])
                    axt.set_yticks([])
                    bes_soc_legend = axt.get_legend_handles_labels()

                if node == "hydrogen_storage":

                    axt = ax[i, 0].twinx()

                    axt.plot(
                        self.rts.G.nodes["hydrogen_storage"][
                            "ionode"
                        ].model.store_storage_state
                        / self.rts.G.nodes["hydrogen_storage"][
                            "ionode"
                        ].model.max_capacity_kg,
                        color="blue",
                        linewidth=1,
                        label="H2S SOC",
                    )
                    axt.set_ylim([0, 1])
                    axt.set_yticks([])
                    h2s_soc_legend = axt.get_legend_handles_labels()

                if node == "thermal_energy_storage":

                    axt = ax[i, 0].twinx()

                    axt.plot(
                        self.rts.G.nodes["thermal_energy_storage"][
                            "ionode"
                        ].model.SOC_store,
                        color="red",
                        linewidth=1,
                        label="TES SOC",
                    )
                    axt.set_ylim([0, 1])
                    axt.set_yticks([])
                    tes_soc_legend = axt.get_legend_handles_labels()

                for k in range(len(in_index)):
                    stop = plot_states[in_index[k], :, j]
                    if np.sum(stop) != 0:
                        ax[i, j_ax].fill_between(
                            np.arange(0, plot_states.shape[1], 1),
                            start,
                            start + stop,
                            step="post",
                            alpha=1,
                            linewidth=0,
                            label=f"from {incoming[k][0:4]}",
                            color=mpl.colormaps[cmaps[j]](cmap_level),
                        )

                    cmap_level += 0.15
                    start += stop

                start = np.zeros(plot_states.shape[1])
                if node == "generation":
                    start = -curtail[:, 0]
                for k in range(len(out_index)):
                    stop = -plot_states[out_index[k], :, j]
                    if np.sum(stop) != 0:
                        ax[i, j_ax].fill_between(
                            np.arange(0, plot_states.shape[1], 1),
                            start,
                            start + stop,
                            step="post",
                            alpha=1,
                            linewidth=0,
                            label=f"to {outgoing[k][0:4]}",
                            color=mpl.colormaps[cmaps[j]](cmap_level),
                        )

                    start += stop
                    cmap_level += 0.15

                if (node == "steel") and (j == 2):
                    steel_output = self.rts.G.nodes["steel"][
                        "ionode"
                    ].model.steel_store_tonne
                    # steel_output = steel_output / np.max(steel_output)
                    steel_output = steel_output * 4300
                    ax[i, j_ax].fill_between(
                        np.arange(0, plot_states.shape[1], 1),
                        np.zeros(len(steel_output)),
                        -steel_output,
                        step="post",
                        alpha=1,
                        linewidth=0,
                        label=f"Steel output",
                        color="darkgreen",
                    )

                []

        # ax[0, 0].set_title("Power")
        # ax[0, 1].set_title("Heat")
        # ax[0, 2].set_title("Hydrogen")

        # if self.rts.stop_index / self.rts.dispatcher.update_period <= 50:
        #     xtick_locs = np.arange(0, self.rts.stop_index, self.rts.dispatcher.update_period)
        #     ax[-1, 0].set_xticks(xtick_locs, xtick_locs, rotation=90)
        #     # ax[-1, j].tick_params(axis="x", direction="in")
        # else:
        #     update_locs = np.arange(0, self.rts.stop_index, self.rts.dispatcher.update_period)
        #     xtick_locs = np.arange(
        #         0,
        #         self.rts.stop_index,
        #         int(
        #             np.round(self.rts.stop_index / 50 / self.rts.dispatcher.update_period)
        #             * self.rts.dispatcher.update_period
        #         ),
        #     )
        #     ax[-1, 0].set_xticks(xtick_locs, xtick_locs, rotation=90)
        #     []

        legend_kwargs = {
            "fontsize": 10,
            "borderpad": 0.2,
            "handlelength": 1.2,
            "handleheight": 0.6,
            "handletextpad": 0.25,
            # "loc": "upper right",
            "loc": "upper center",
            "ncols": 8,
        }

        for i in range(ax.shape[0]):
            for j in range(ax.shape[1]):
                # ax[i,j].set_ylim([0, ax[i,j].get_ylim()[1]])
                ax[i, j].set_ylim(
                    [
                        -np.max(np.abs(ax[i, j].get_ylim())),
                        np.max(np.abs(ax[i, j].get_ylim())),
                    ]
                )
                ax[i, j].yaxis.tick_right()
                t = ax[i, j].yaxis.get_offset_text()
                t.set_x(1.01)
                # if ax.shape[1] == 1:
                #     ax[i, j].set_yticks([])
                ax[i, j].axhline(0, linewidth=0.5, color="black", alpha=0.5, zorder=0.5)
                ax[i, j].tick_params(axis="x", direction="in")

                if self.rts.node_order[i] == "battery":
                    handles, labels = ax[i, j].get_legend_handles_labels()
                    handles.append(bes_soc_legend[0][0])
                    labels.append(bes_soc_legend[1][0])
                    ax[i, j].legend(handles, labels, **legend_kwargs)

                elif self.rts.node_order[i] == "hydrogen_storage":
                    handles, labels = ax[i, j].get_legend_handles_labels()
                    handles.append(h2s_soc_legend[0][0])
                    labels.append(h2s_soc_legend[1][0])
                    ax[i, j].legend(handles, labels, **legend_kwargs)
                elif self.rts.node_order[i] == "thermal_energy_storage":
                    handles, labels = ax[i, j].get_legend_handles_labels()
                    handles.append(tes_soc_legend[0][0])
                    labels.append(tes_soc_legend[1][0])
                    ax[i, j].legend(handles, labels, **legend_kwargs)
                else:
                    handles, labels = ax[i, j].get_legend_handles_labels()

                ax[i, j].legend(handles, labels, **legend_kwargs)

        # if self.rts.stop_index <= 8760:
        #     ax[0, 0].set_xlim([0, self.rts.stop_index])
        # else:
        #     ax[0, 0].set_xlim([0, 8760])

        ax[0, 0].set_xlim(
            [np.max([0, self.rts.start_index]), np.min([8760, self.rts.stop_index])]
        )

        if save:

            fig.savefig(f"{fname}{'_8760.pdf'}", format="pdf")
            # ax[0,0].set_xlim([2800, 3150])
            # fig.savefig(f"{fname}{'_zoom.pdf'}", format="pdf")

        []
