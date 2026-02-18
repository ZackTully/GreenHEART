import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path
import sys
import yaml
import pprint
import logging
from logging import handlers
import tqdm
from multiprocessing import current_process
import traceback


import time

from greenheart.simulation.technologies.dispatch.dispatch import GreenheartDispatch
from greenheart.simulation.technologies.dispatch.forecast import Forecast

from greenheart.simulation.realtime_node import (
    Node,
    setup_generation_node,
    setup_electricity_output_node,
    setup_battery_node,
    setup_electrolyzer_node,
    setup_hydrogen_storage_node,
    setup_thermal_energy_storage_node,
    setup_heat_exchanger_node,
    setup_steel_node,
)
from greenheart.tools.simulation.realtime_helper import RealTimeSimulationHelper
from hopp.utilities import load_yaml


class RealTimeSimulation:
    def __init__(self, config, hopp_interface, case_description=None):

        self.case_description = case_description

        self.config = config
        self.rts_config = self.config.greenheart_config["realtime_simulation"]

        self.forecast_config = self.rts_config.get(
            "forecast",
            None,
            # dict(
            #     horizon=self.dispatcher.controller.horizon,
            #     method="perfect_method",
            #     method_config={},
            # ),
        )

        options = self.rts_config.get("options", {})

        self.tqdm_progress = options.get("tqdm_progress", False)
        self.verbose = options.get("verbose", True)
        self.save_sysid = options.get("save_sysid", False)

        self.stop_index = self.rts_config.get("stop_index", 8760)
        self.start_index = self.rts_config.get("start_index", 0)

        if isinstance(self.rts_config["component_config"], str):
            self.component_config = yaml.safe_load(
                open(self.rts_config["component_config"], "r")
            )
        else:
            self.component_config = self.rts_config["component_config"]

        self.hi = hopp_interface

        self.rts_helper = RealTimeSimulationHelper(self)

        # For setting the line height of the progress bar when running in parallel
        self.worker_id = 0
        if (self.case_description is not None) and ("case" in case_description):
            case_num = case_description.split("case_")[1].split(".")[0].split("_")[0]
            self.worker_id = int(case_num)
        elif current_process().name != "MainProcess":
            self.worker_id = current_process()._identity[0] - 1

        if "logging" in self.rts_config:
            self.setup_logging(self.rts_config.pop("logging"))
        else:
            self.logger = logging.getLogger()
            self.logger.setLevel(logging.DEBUG)
            terminal_handler = logging.StreamHandler(sys.stdout)
            terminal_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(terminal_handler)

        self.setup_simulation_model(config)
        self.setup_record_keeping()
        if self.save_sysid:
            self.rts_helper.setup_ctrl_sysid()

        self.edge_error_count = 0

    def setup_logging(self, log_config):

        if log_config["queue"] is None:
            self.logger = log_config["logger"]

        else:
            logger_name = f"SIMULATION {log_config['case_description']}"
            self.logger = logging.getLogger(logger_name)
            self.logger.setLevel(logging.DEBUG)

            queue_handler = handlers.QueueHandler(log_config["queue"])
            queue_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(queue_handler)

        self.logger.info("Simulation logger initialized")

    def setup_simulation_model(self, config):

        GH_tech_options = [
            "generation",
            "curtail",
            "battery",
            "electrolyzer",
            "joule_heater",
            "hydrogen_storage",
            "heat_exchanger",
            "thermal_energy_storage",
            "steel",
            "output",
        ]

        GH_techs = []
        for key in config.greenheart_config.keys():
            if key in GH_tech_options:
                GH_techs.append(key)

        # graph_config_fpath = config.greenheart_config["realtime_simulation"]["system"][
        #     "system_graph_config"
        # ]
        # graph_config = yaml.safe_load(open(graph_config_fpath, "r"))

        graph_config = self.rts_config["system"]["system_graph_config"]
        if isinstance(graph_config, str):
            graph_config = yaml.safe_load(open(graph_config, "r"))

        network_config = graph_config["network"]

        edges = network_config
        nodes = []
        for edge in network_config:
            for node in edge:
                if node not in nodes:
                    nodes.append(node)

        GH_techs = nodes

        if "traversal_order" in graph_config.keys():
            self.node_order = graph_config["traversal_order"]

        if "print_locs" in graph_config.keys():
            self.print_locs = graph_config["print_locs"]

        self.edge_order = edges

        G = nx.DiGraph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)

        self.G = G

        # Instantiate the individual steppable models of each technology

        RT_techs = {}

        subsystem_args = (self.G, self.config, self.hi, self.component_config)

        for GH_tech in GH_techs:

            if GH_tech == "generation":
                RT_techs.update(setup_generation_node(*subsystem_args))
            elif GH_tech == "electricity_output":
                RT_techs.update(setup_electricity_output_node(*subsystem_args))
            elif GH_tech == "battery":
                RT_techs.update(setup_battery_node(*subsystem_args))
            elif GH_tech == "electrolyzer":
                RT_techs.update(setup_electrolyzer_node(*subsystem_args))
            elif GH_tech == "hydrogen_storage":
                RT_techs.update(setup_hydrogen_storage_node(*subsystem_args))
            elif GH_tech == "thermal_energy_storage":
                RT_techs.update(setup_thermal_energy_storage_node(*subsystem_args))
            elif GH_tech == "heat_exchanger":
                RT_techs.update(setup_heat_exchanger_node(*subsystem_args))
            elif GH_tech == "steel":
                RT_techs.update(setup_steel_node(*subsystem_args))

        # Build the connections with a graph network

        for node in self.G.nodes:
            model = None

            ionode = Node(
                name=node,
                model=RT_techs[node]["model"],
                expected_inputs=RT_techs[node]["model_inputs"],
                expected_outputs=RT_techs[node]["model_outputs"],
                in_degree=self.G.in_degree[node],
                out_degree=self.G.out_degree[node],
            )

            # Add an assertion that there is one source and one sink
            is_source = False
            is_sink = False
            if graph_config["source_node"] == node:
                is_source = True
            if graph_config["sink_node"] == node:
                is_sink = True

            self.G.nodes[node].update(
                {"ionode": ionode, "is_source": is_source, "is_sink": is_sink}
            )

        for node in list(self.G.nodes):
            assert (
                not self.G.nodes[node]["ionode"].model == None
            ), f"no model for node: {node}"

        # self.plot_system_graph()

        self.technologies = RT_techs
        # Print or log the control input format

        # find which nodes are splitting nodes by which ones have out degree > 1

        []

    def step_system_state_function(self, G_dispatch, generation_available, step_index):

        if hasattr(self, "node_order"):
            node_order = self.node_order

        simulated_edges = list(self.G.edges)
        simulated_IO = {}
        for edge in simulated_edges:
            simulated_IO.update({edge: {"simulated": None}})

        for node in list(self.G.nodes):
            self.G.nodes[node].update({"wasted_output": np.zeros(4)})

        for edge in list(self.G.edges):
            self.G.edges[edge].update({"simulated": None})

        for node in node_order:

            if step_index == self.start_index:
                self.logger.info(f"Step: {step_index}, simulating node: {node}")

            if node == "generation":
                this_node_input = generation_available
                # self.G.nodes[node]["model"].set_output(generation_available)
                self.G.nodes[node]["ionode"].model.set_output(generation_available)

            sim_in_edges = self.G.in_edges(node)

            this_node_input = []

            for in_edge in sim_in_edges:
                # Check that upstream values have been simulated
                # edge_data = G_simulated.get_edge_data(in_edge[0], in_edge[1])["value"]
                edge_data = self.G.get_edge_data(in_edge[0], in_edge[1])["simulated"]
                # assert edge_data is not None, "edge data is none, needs to be run first"
                # assert not np.isnan(edge_data)

                this_node_input.append(edge_data)

            if len(this_node_input) > 0:
                this_node_input = np.stack(this_node_input)
            else:
                this_node_input = np.zeros((1, 4))

            node_dispatch_split = np.array(self.G.nodes[node]["dispatch_split"])
            node_dispatch_ctrl = np.array(self.G.nodes[node]["dispatch_ctrl"])

            node_output = self.G.nodes[node]["ionode"].step(
                this_node_input,
                node_dispatch_ctrl,
                node_dispatch_split,
                step_index,
            )

            sim_out_edges = self.G.out_edges(node)
            wasted_output = np.zeros_like(node_output)

            self.G.nodes[node].update({"wasted_output": wasted_output})
            for i, out_edge in enumerate(list(sim_out_edges)):
                simulated_IO[out_edge]["simulated"] = node_output[:, i]

            nx.set_edge_attributes(self.G, simulated_IO)

            # To check that the edge attributes are being updated
            # nx.get_edge_attributes(G_simulated, "value")

        return self.G

    def get_time_iterable(self, hybrid_profile):
        total = np.min([self.stop_index, 8760]) - np.max([0, self.start_index])
        if self.tqdm_progress:
            # Print progress using a tqdm progress bar
            colors = [
                "#32a852",
                "#a010b0",
                "#c81b3b",
                "#ddac2e",
                "#c95517",
                "#2926da",
                "#22dcb3",
                "#c2b1cf",
            ]
            tqdm_color = colors[self.worker_id % len(colors)]
            time_iterable = tqdm.tqdm(
                range(len(hybrid_profile)),
                desc=self.case_description,
                position=self.worker_id + 1,
                leave=False,
                colour=tqdm_color,
                total=total,
                bar_format="{l_bar}{bar:40}|{postfix}"
                # bar_format="{l_bar}{bar:40}{r_bar}"
            )

        else:
            time_iterable = range(len(hybrid_profile))

        return time_iterable

    def get_hybrid_profile(self, hopp_results):

        # Check if a path to load a generation profile was given in the config file
        gen_profile_path = self.rts_config.get("generation_profile", False)

        
        if gen_profile_path:

            hybrid_profile = np.load(gen_profile_path)

            pass
        else:
            # If no path to load profile or path is not specified in config, check hopp
            gen_profiles = {}
            technologies = hopp_results["annual_energies"]["technologies"]

            for hopp_tech in technologies.keys():

                if hopp_tech in ["pv", "wind"]:
                    tech_profile = technologies[hopp_tech].generation_profile
                    gen_profiles.update({hopp_tech: tech_profile})

            hybrid_profile = np.array(gen_profiles["pv"]) + np.array(gen_profiles["wind"])

            assert not np.any(
                np.isnan(hybrid_profile)
            ), "Nans found in hopp_results hybrid profile"

        return hybrid_profile

    def simulate(self, dispatcher: GreenheartDispatch, hopp_results):

        if self.verbose:
            self.logger.info(f"{self.case_description}, Simulation started ")

        # Get generation signals
        hybrid_profile = self.get_hybrid_profile(hopp_results)
        self.hybrid_profile = hybrid_profile

        self.dispatcher = dispatcher
        self.forecaster = Forecast(self.forecast_config, hybrid_profile, self.config)

        if self.verbose:
            self.logger.info("Forecaster initialized")


        # For timing
        t0 = time.time()
        t_log_last = time.time()

        time_iterable = self.get_time_iterable(hybrid_profile)
        for i in time_iterable:

            if i > self.stop_index:
                print("stopping at realtime simulator stop index")
                self.logger.info("stopping at realtime simulator stop index")
                break

            if i < self.start_index:
                continue

            if self.verbose and i == self.start_index:
                self.logger.info("First time step")

            try:
                forecast = self.forecaster.get_forecast(hybrid_profile[i], i)
            except Exception as e:
                self.logger.error(f"Forecasting calculation error at step: {i}")
                self.logger.error(traceback.format_exc())
            
            if self.verbose and i == self.start_index:
                self.logger.info("First time step, ran forecast")

            x0 = self.get_state_measurement(step_index=i)

            if self.verbose and i == self.start_index:
                self.logger.info("First time step, got state measurement")


            self.G = dispatcher.step(
                self.G,
                hybrid_profile[i],
                forecast=forecast,
                x_measured=x0,
                step_index=i,
            )

            if self.verbose and i == self.start_index:
                self.logger.info("First time step, got dispatch input")

            if "generation" in self.G.nodes:
                if "grid_purchase" in self.G.nodes["generation"]:
                    grid_power = self.G.nodes["generation"]["grid_purchase"]
                else:
                    grid_power = np.array([0])

            self.G = self.step_system_state_function(
                self.G, hybrid_profile[i] + grid_power, i
            )

            if self.verbose and i == self.start_index:
                self.logger.info("First time step, simulated simulation graph")


            stop_idx = np.min([self.stop_index, len(hybrid_profile)])


            prog_step = f"{i}".rjust(4) + f"/{stop_idx}".rjust(5)
            prog_percent = f"{(i / (stop_idx)* 100) :.1f}%".rjust(6)
            el_sec = time.time() - t0
            prog_elapsed = f"{np.floor(el_sec/3600):.0f}".zfill(2) + ":" + f"{(np.floor(el_sec/60))%60:.0f}".zfill(2) + ":" + f"{el_sec%60:.0f}".zfill(2)
            rem_sec = ((1 - i/stop_idx) * (time.time() - t0) / ((i+1) / stop_idx))
            prog_remaining = f"{np.floor(rem_sec/3600):.0f}".zfill(2) + ":" + f"{(np.floor(rem_sec/60))%60:.0f}".zfill(2) + ":" + f"{rem_sec%60:.0f}".zfill(2)
            prog_str = f" {prog_step}, {prog_percent}, ela.: {prog_elapsed}, rem.: {prog_remaining}"
            if (time.time() - t_log_last > 60) or (i == self.stop_index):
                # Print and log progress every 60 seconds
                self.logger.info(prog_str)
                t_log_last = time.time()

            self.record_states(i, self.G, grid_power)

            if self.tqdm_progress:
                if "steel" in self.G.nodes:
                    time_iterable.set_postfix({"Prog" : prog_str, "yex" : f'{self.G.nodes["steel"]["ionode"].model.steel_store_tonne[i]:7.2f}'})
                # time_iterable.set_postfix({"x_bes" : f'{self.G.nodes["battery"]["ionode"].model.hopp_battery._system_model.StatePack.SOC:.2f}'})

            if self.save_sysid:
                self.rts_helper.save_ctrl_for_sysid(step_index=i)

            if self.dispatcher.use_MPC:
                sim_edges = self.get_sim_edges(i)
                mpc_edges = self.get_mpc_edges(i)

                err, sim_curt, sim_pass = self.check_edge_error(sim_edges, mpc_edges, i)
                self.record_error(
                    err,
                    sim_curt,
                    sim_pass,
                    sim_edges,
                    mpc_edges,
                    step_index=i,
                )
            else:
                sim_edges = self.get_sim_edges(i)
                heu_edges = self.get_heuristic_edges(i)

                err, sim_curt, sim_pass = self.check_edge_error(sim_edges, heu_edges, i)
                self.record_error(
                    err,
                    sim_curt,
                    sim_pass,
                    sim_edges,
                    heu_edges,
                    step_index=i,
                )


        t1 = time.time()
        self.simulation_elapsed_time = t1 - t0

        for node in self.G.nodes:
            if hasattr(self.G.nodes[node]["ionode"].model, "consolidate_sim_outcome"):
                self.G.nodes[node]["ionode"].model.consolidate_sim_outcome()

        self.models = {
            node: self.G.nodes[node]["ionode"].model for node in self.node_order
        }

        if self.rts_config.get("generation_profile", False) and self.dispatcher.use_MPC:
            self.dispatcher.controller.plotter.plot_saved_trajectories(n=self.stop_index)
            self.dispatcher.controller.plotter.plot_saved_trajectories_paper()
            # And maybe save too?
            []


        # self.logger.info(f"Simulation took: {self.simulation_elapsed_time/60:.2f} min or {self.simulation_elapsed_time/3600:.2f} hr")

    def get_sim_edges(self, i):
        sim_edges_full = self.system_states[:, i, :]
        sim_edges = np.zeros(sim_edges_full.shape[0])
        for j in range(sim_edges_full.shape[0]):
            sim_edges[j] = np.sum(sim_edges_full[j, 0:-1])
        return sim_edges

    def get_mpc_edge_permutation(self, mpc_edges):
        permutation = np.zeros(mpc_edges.shape, dtype=int)
        for k in range(len(self.edge_order)):
            index = [
                ind
                for ind in range(len(self.dispatcher.controller.pco_label))
                if (
                    (
                        self.dispatcher.controller.pco_label[ind].split(" ")[2]
                        == self.edge_order[k][0]
                    )
                    and (
                        self.dispatcher.controller.pco_label[ind]
                        .split(" ")[-1]
                        .split(")")[0]
                        == self.edge_order[k][1]
                    )
                )
            ]
            permutation[k] = index[0]

        self.mpc_permutation = permutation

    def get_mpc_edges(self, i):

        # Find the most recent mpc solve to query if mpc is not updated every step
        idx_store = np.array(self.dispatcher.controller.step_index_store)
        if not (i % self.dispatcher.update_period) or (i == 0):
            idx = np.where(idx_store == i)[0][0]
        else:
            idx = np.where(idx_store == self.dispatcher.previous_update)[0][0]

        mpc_edges = self.dispatcher.controller.ysp_store[idx]
        mpc_edges = mpc_edges[0:-1, i - self.dispatcher.previous_update]

        if not hasattr(self, "mpc_permutation"):
            self.get_mpc_edge_permutation(mpc_edges)

        return mpc_edges[self.mpc_permutation]
    
    def get_heuristic_edges(self, i):
        return self.dispatcher.controller.yco_ctrl


    def check_edge_error(self, sim_edges, mpc_edges, i):

        edge_error = sim_edges - mpc_edges

        edge_percent_error = (sim_edges - mpc_edges) / (0.5 * (sim_edges + mpc_edges))
        # tol = 0.15
        tol = 0.5
        ignore_tol = 300
        if np.any(np.abs(edge_percent_error) > tol):
            erronious_indices = np.where(np.abs(edge_percent_error) > tol)[0]
            erronious_edges = [self.edge_order[i] for i in erronious_indices]

            if np.all(np.abs(sim_edges[erronious_indices] < ignore_tol)) and np.all(
                np.abs(mpc_edges[erronious_indices] < ignore_tol)
            ):
                pass
            else:

                self.logger.warning(
                    f"Step {i}, MPC/sim. edge difference greater than tolerance ({tol * 100}%). Erronious edges: {erronious_edges}"
                )

                self.logger.warning(
                    f"Sim edges: {  {str(self.edge_order[k]): str(sim_edges[k]) for k in erronious_indices}    }"
                )
                self.logger.warning(
                    f"MPC edges: {  {str(self.edge_order[k]): str(mpc_edges[k]) for k in erronious_indices}    }"
                )
                self.logger.warning(
                    f"Percent difference: {  {str(self.edge_order[k]): str(edge_percent_error[k]*100) for k in erronious_indices}    }"
                )

                # assert self.edge_error_count < 50, f"Step {i}, MPC/sim. edge difference greater than tolerance ({tol * 100}%). Erronious edges: {erronious_edges}"
                self.edge_error_count += 1
                # raise AssertionError(f"Step {i}, MPC/sim. edge difference greater than tolerance ({tol * 100}%). Erronious edges: {erronious_edges}")

        error_dict = {
            str(self.edge_order[k]): edge_error[k] for k in range(len(self.edge_order))
        }

        sim_u_curtail = {
            node: self.G.nodes[node]["ionode"].u_curtail_store[i]
            for node in self.node_order
        }
        sim_u_passthrough = {
            node: self.G.nodes[node]["ionode"].u_passthrough_store[i]
            for node in self.node_order
        }

        return error_dict, sim_u_curtail, sim_u_passthrough

    def get_state_measurement(self, step_index):
        x0 = []
        for state_node in ["battery", "thermal_energy_storage", "hydrogen_storage"]:
            if state_node in self.G:
                model = self.G.nodes[state_node]["ionode"].model
                if state_node == "battery":
                    first_step = step_index == self.start_index
                    state = [
                        model.get_state_measurement(step_index, first_step=first_step)
                    ]
                elif state_node == "hydrogen_storage":
                    state = [model.get_state_measurement(step_index)]
                elif state_node == "thermal_energy_storage":
                    state_H, state_M = model.get_state_measurement(step_index)
                    state = [state_H, state_M]
                x0.append(state)
        x0 = np.concatenate(x0)
        return x0

    def setup_record_keeping(self):
        duration = 8760

        # index_dict = {}
        # for i, edge in enumerate(self.G.edges):
        #     index_dict.update({edge: i})

        index_dict = {tuple(self.edge_order[i]): i for i in range(len(self.edge_order))}

        self.index_dict = index_dict
        self.system_states = np.zeros((len(self.G.edges), duration, 4))
        # self.node_waste = np.zeros((len(self.G.nodes), duration, 4))

        self.grid_power_store = np.zeros((1, duration))

        self.sim_edge_store = np.zeros((len(self.G.edges), duration))
        self.mpc_edge_store = np.zeros((len(self.G.edges), duration))

        self.edge_error_store = np.zeros((len(self.G.edges), duration))
        # passthrough_dims = [self.G.nodes[node]["ionode"].u_passthrough_store.shape[1] for node in self.node_order]
        # curtail_dims = [self.G.nodes[node]["ionode"].u_curtail_store.shape[1] for node in self.node_order]
        # self.input_error = {self.node_order[i]: {curtail}}

        self.curtail_store = {}
        self.passthrough_store = {}
        for node in self.node_order:
            n_inputs = np.sum(self.G.nodes[node]["ionode"].input_list)
            if self.G.nodes[node]["ionode"].inputs["T"]:
                n_inputs -= 1

            self.curtail_store.update({node: np.zeros((duration, n_inputs))})
            self.passthrough_store.update({node: np.zeros((duration, n_inputs))})

        # self.curtail_store = np.zeros((len(self.G.nodes), duration))
        # self.passthrough_store = np.zeros((len(self.G.nodes), duration))

    def record_error(
        self,
        error_dict,
        node_curtail=None,
        node_passthrough=None,
        sim_edges=None,
        mpc_edges=None,
        step_index=None,
    ):

        self.edge_error_store[:, step_index] = list(error_dict.values())

        self.sim_edge_store[:, step_index] = sim_edges
        self.mpc_edge_store[:, step_index] = mpc_edges

        for node in self.node_order:
            self.curtail_store[node][step_index, :] = node_curtail[node]
            self.passthrough_store[node][step_index, :] = node_passthrough[node]

        # self.curtail_store[:, step_index] = np.concatenate(list(node_curtail.values()))
        # self.passthrough_store[:, step_index] = np.concatenate(list(node_passthrough.values()))

    def record_states(self, time_step, simulated_IO, grid_power):
        self.grid_power_store[0, time_step] = grid_power[0]

        values = nx.get_edge_attributes(simulated_IO, "simulated")
        for key in values.keys():
            self.system_states[self.index_dict[key], time_step, :] = values[key]


    def get_component(self, component_name):
        return self.G.nodes[component_name]["ionode"].model

    def unpack_component(self, component_name, make_plot=False):
        # get inputs outputs, component model

        # Do graph stuff

        # input edges
        # output edges

        rt_node = self.G.nodes[component_name]["ionode"]

        passthrough = rt_node.u_passthrough_store
        input_curtail = rt_node.u_curtail_store
        split_curtail = rt_node.u_curtail_split_store[
            :, np.where(rt_node.output_list)[0]
        ].T[0, :]

        if rt_node.inputs["T"]:

            disturbance = rt_node.disturbance_store[:, 0:-1]
        else:
            disturbance = rt_node.disturbance_store

        in_edges = self.G.in_edges(component_name)

        in_edge_index = []
        for in_edge in in_edges:
            for i, edge in enumerate(list(self.G.edges)):
                if in_edge == edge:
                    in_edge_index.append(i)

        # in_edge_data = self.system_states[
        #     in_edge_index, :, np.where(rt_node.input_list)[0]
        # ]
        in_edge_data = self.system_states[:, :, np.where(rt_node.input_list)[0]][
            in_edge_index, :, :
        ]

        out_edges = self.G.out_edges(component_name)

        out_edge_index = []
        for out_edge in out_edges:
            for i, edge in enumerate(list(self.G.edges)):
                if out_edge == edge:
                    out_edge_index.append(i)
        out_edge_data = self.system_states[:, :, np.where(rt_node.output_list)[0]][
            out_edge_index, :, :
        ]
        # out_edge_data = self.system_states[
        #     out_edge_index, :, np.where(rt_node.output_list)[0]
        # ]

        # In these get the states and LLC signals

        # comp_local_data should have u, x, y, control?

        if component_name == "generation":
            comp_local_data = self.unpack_generation()
        elif component_name == "battery":
            comp_local_data = self.unpack_battery()
        elif component_name == "thermal_energy_storage":
            comp_local_data = self.unpack_thermal_energy_storage()
        else:
            comp_local_data = {"uct": [], "x": [], "y": []}

        component_data = {
            "passthrough": passthrough,
            "input_curtail": input_curtail,
            "split_curtail": split_curtail,
            "disturbance": disturbance,
            "in_edges": list(in_edges),
            "in_data": in_edge_data,
            "out_edges": list(out_edges),
            "out_data": out_edge_data,
            "uct": comp_local_data["uct"],
            "x": comp_local_data["x"],
            "y": comp_local_data["y"],
        }

        # if make_plot:
        #     self.plot_component(component_name, component_data)

        return component_data

  

    def plot_system_graph(self):
        self.rts_helper.plot_system_graph()

    def plot_edges(self):
        self.rts_helper.plot_edges()

    def plot_nodes(
        self, data="edges", figsize=(15, 8), hide_yaxis=True, fname=None, save=False
    ):
        self.rts_helper.plot_nodes(
            data=data, figsize=figsize, hide_yaxis=hide_yaxis, fname=fname, save=save
        )
