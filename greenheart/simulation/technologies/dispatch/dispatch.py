import numpy as np
import networkx as nx
import pprint


class GreenheartDispatchConfig:
    def __init__(self):
        pass


class GreenheartDispatchOutput:
    def __init__(self):
        pass


class GreenheartDispatch:
    def __init__(self, hopp_interface, GHconfig, dispatch_config=None):

        self.setup_control_model(GHconfig)
        self.setup_constraints(GHconfig)

        self.setup_time_parameters(hopp_interface, GHconfig, dispatch_config)
        self.setup_objective(hopp_interface, GHconfig, dispatch_config)

        pass

        # extract the timeseries- and feedback-capabale component simulation models from hopp
        # hopp_simulation_models = hopp_interface.hopp.system.technologies

        # # extract or initialze the timeseries- and feedback-capabale component simulation models from greenheart
        # GH_simulation_model_names = ["X", "Y", "Z"]
        # GH_simulation_models = []
        # for GH_model in GH_simulation_model_names:
        #     GH_simulation_models.append(GH_model)

        # self.setup_simulation_model(hopp_simulation_models, GH_simulation_models)

    def setup_time_parameters(self, hopp_interface, GHconfig, dispatch_config):
        # Set up MPC time parameters for dispatch
        # Make sure the GH, hopp, and dispatch config files don't contradict
        self.dt = 1
        self.horizon = 1

    def setup_objective(self, hopp_interface, GHconfig, dispatch_config):
        # Set up the objective function
        self.objective = []

    def setup_control_model(self, GHconfig):
        # set up linear control model based on the parameters in hopp_config and in GHconfig
        hopp_config = GHconfig.hopp_config
        self.control_model = []

    def setup_constraints(self, GHconfig):
        # set up control model constraints based on the parameters in hopp_config and GHconfig
        hopp_config = GHconfig.hopp_config
        self.control_constraints = []

    def setup_simulation_model(self, hopp_simulation_models, GH_simulation_models):
        # Build graph network or something
        self.simulation_model = []

    def optimize(self, objective):
        optimal = []
        return optimal

    def step_control(self):
        # Called from within hopp hybrid_dispatch_builder_solver
        # Optimize the system trajectory for one horizon

        state_measurement = 0

        optimal_trajectory = self.optimize(self.objective)

        control_actions = optimal_trajectory
        control_actions_for_hopp = []

        return control_actions_for_hopp

    def get(self, data_name):

        return getattr(self, data_name)

    def step(self, G_dispatch, available_power, feedback_error=None):

        # G_dispatch = self.example_control_elec_storage_steel(G_dispatch, available_power)
        # G_dispatch = self.example_control_hydrogen_heat(G_dispatch, available_power)
        # G_dispatch = self.example_control_elec_heat_exchanger(G_dispatch, available_power)
        # G_dispatch = self.example_control_BES_TES_heat_exchanger(G_dispatch, available_power, feedback_error)
        G_dispatch = self.splitting_controller(
            G_dispatch, available_power, feedback_error
        )
        return G_dispatch

    def splitting_controller(self, G, available_power, feedback_error=None):

        mean_generation = 150000  # kWh / hr
        frac_electrolysis = 0.944
        frac_heating = 0.056
        mean_to_hydrogen = frac_electrolysis * mean_generation
        mean_to_heating = frac_heating * mean_generation

        power_to_hydrogen = frac_electrolysis * available_power

        gen_to_EL = np.min([mean_to_hydrogen, power_to_hydrogen])
        gen_to_BES = np.max([0, power_to_hydrogen - mean_to_hydrogen])
        BES_to_EL = np.max([0, mean_to_hydrogen - power_to_hydrogen])
        EL_to_HX = 1 / 55 * (gen_to_EL + BES_to_EL)

        # Heat path

        power_to_heating = frac_heating * available_power

        gen_to_HX = np.min([mean_to_heating, power_to_heating])
        gen_to_TES = np.max([0, power_to_heating - mean_to_heating])
        TES_to_HX = np.max([0, mean_to_heating - power_to_heating])

        HX_to_output = mean_to_hydrogen / 55

        gen_split = np.array([gen_to_EL, gen_to_BES, gen_to_HX, gen_to_TES])
        gen_split_normalized = np.nan_to_num(gen_split / np.sum(gen_split))


        # BES_charge = np.sign(gen_to_BES)
        # if gen_to_BES == 0:
        #     BES_charge = 0


        # TES_charge = np.sign(gen_to_TES)
        # if gen_to_TES == 0:
        #     TES_charge = 0 


        if gen_to_BES > 0:
            BES_charge = gen_to_BES
        elif BES_to_EL > 0:
            BES_charge = -BES_to_EL
        else:
            BES_charge = 0

        if gen_to_TES > 0:
            TES_charge = gen_to_TES
        elif TES_to_HX > 0:
            TES_charge = -TES_to_HX
        else:
            TES_charge = 0

        
        

        for node in list(G.nodes):
            G.nodes[node].update({"dispatch_split": np.array([1])})
            G.nodes[node].update({"dispatch_ctrl": None})

        G.nodes["generation"].update({"dispatch_split": gen_split_normalized})
        G.nodes["battery"].update({"dispatch_ctrl": BES_charge})
        G.nodes["thermal_energy_storage"].update({"dispatch_ctrl": TES_charge})


        dispatch_IO = {
            ("generation", "electrolyzer"): {"dispatch": [gen_to_EL, 0, 0, 0]},
            ("generation", "battery"): {"dispatch": [gen_to_BES, 0, 0, 0]},
            ("battery", "electrolyzer"): {"dispatch": [BES_to_EL, 0, 0, 0]},
            ("electrolyzer", "heat_exchanger"): {"dispatch": [0, 0, EL_to_HX, 80]},
            ("generation", "thermal_energy_storage"): {
                "dispatch": [gen_to_TES, 0, 0, 0]
            },
            ("generation", "heat_exchanger"): {"dispatch": [gen_to_HX, 0, 0, 0]},
            ("thermal_energy_storage", "heat_exchanger"): {
                "dispatch": [0, TES_to_HX, 0, 0]
            },
            ("heat_exchanger", "output"): {"dispatch": [0, 0, HX_to_output, 900]},
        }

        nx.set_edge_attributes(G, dispatch_IO)




        return G

    def example_control_BES_TES_heat_exchanger(
        self, G, available_power, feedback_error=None
    ):

        # Plan for CF = 0.3, 150 MW average generation

        mean_generation = 150000  # kWh / hr

        frac_electrolysis = 0.944
        frac_heating = 0.056

        # Hydrogen path
        mean_to_hydrogen = frac_electrolysis * mean_generation

        power_to_hydrogen = frac_electrolysis * available_power

        gen_to_EL = np.min([mean_to_hydrogen, power_to_hydrogen])
        gen_to_BES = np.max([0, power_to_hydrogen - mean_to_hydrogen])
        BES_to_EL = np.max([0, mean_to_hydrogen - power_to_hydrogen])
        EL_to_HX = 1 / 55 * (gen_to_EL + BES_to_EL)

        # Heat path
        mean_to_heating = frac_heating * mean_generation

        power_to_heating = frac_heating * available_power

        gen_to_HX = np.min([mean_to_heating, power_to_heating])
        gen_to_TES = np.max([0, power_to_heating - mean_to_heating])
        TES_to_HX = np.max([0, mean_to_heating - power_to_heating])

        HX_to_output = mean_to_hydrogen / 55

        dispatch_IO = {
            ("generation", "electrolyzer"): {"dispatch": [gen_to_EL, 0, 0, 0]},
            ("generation", "battery"): {"dispatch": [gen_to_BES, 0, 0, 0]},
            ("battery", "electrolyzer"): {"dispatch": [BES_to_EL, 0, 0, 0]},
            ("electrolyzer", "heat_exchanger"): {"dispatch": [0, 0, EL_to_HX, 80]},
            ("generation", "thermal_energy_storage"): {
                "dispatch": [gen_to_TES, 0, 0, 0]
            },
            ("generation", "heat_exchanger"): {"dispatch": [gen_to_HX, 0, 0, 0]},
            ("thermal_energy_storage", "heat_exchanger"): {
                "dispatch": [0, TES_to_HX, 0, 0]
            },
            ("heat_exchanger", "output"): {"dispatch": [0, 0, HX_to_output, 900]},
        }

        # TODO Update to get the error signal from G not from feedback_error
        if feedback_error is not None:
            error_IO = nx.get_edge_attributes(feedback_error, "error")

            for key in dispatch_IO.keys():
                self.dispatch_IO[key]["value"] = (
                    dispatch_IO[key]["value"] - error_IO[key]
                )
                dispatch_IO = self.dispatch_IO
        else:
            self.dispatch_IO = dispatch_IO

        nx.set_edge_attributes(G, dispatch_IO)

        # pprint.pprint(nx.get_edge_attributes(G_dispatch, "value"))

        return G

    def example_control_elec_heat_exchanger(self, G_dispatch, available_power):

        gen_to_curtail = [np.max([available_power - 50000, 0]), 0, 0, 0]
        gen_to_el = [50000 * 0.95, 0, 0, 0]
        el_to_hx = [0, 0, (1 / 55) * gen_to_el[0], 80]
        # el_to_hx = [0, 0, 500, 80]
        # gen_to_hx = [available_power / 2, 0, 0, 0]
        gen_to_hx = [50000 * 0.05, 0, 0, 0]

        nx.set_edge_attributes(
            G_dispatch,
            {
                ("generation", "curtail"): {"value": gen_to_curtail},
                ("generation", "electrolyzer"): {"value": gen_to_el},
                ("electrolyzer", "heat_exchanger"): {"value": el_to_hx},
                ("generation", "heat_exchanger"): {"value": gen_to_hx},
            },
        )

        return G_dispatch

    def example_control_hydrogen_heat(self, G_dispatch, available_power):

        for edge in list(G_dispatch.edges):
            G_dispatch.edges[edge].update({"value": available_power / 3})

        return G_dispatch

    def example_control_elec_storage_steel(self, G_dispatch, available_power):
        electrolyzer_eta = 1 / 55  # kg/kWh
        average_generation = (
            0.3 * (10 * 6000) + 0.3 * 100000
        )  # wind + solar rated * 0.3 capacity factor

        # average_generation = 50000

        # Always send all of the power through the electrolyzer
        # If available is greater than average, charge the storage by the difference
        # If available is less than average, discharge the storage by the difference

        available_more_than_average_difference = np.max(
            [0, available_power - average_generation]
        )
        available_less_than_average_difference = np.max(
            [0, average_generation - available_power]
        )

        generation_to_curtail = 0
        generation_to_electrolyzer = available_power
        electrolyzer_to_storage = (
            electrolyzer_eta * available_more_than_average_difference
        )
        # electrolyzer_to_storage = electrolyzer_eta * (available_power - average_generation)

        electrolyzer_to_steel = electrolyzer_eta * average_generation
        storage_to_steel = electrolyzer_eta * available_less_than_average_difference

        assert not np.isnan(storage_to_steel)

        dispatch_edges = list(G_dispatch.edges)
        dispatch_IO = {
            ("generation", "curtail"): {"value": generation_to_curtail},
            ("generation", "electrolyzer"): {"value": generation_to_electrolyzer},
            ("electrolyzer", "hydrogen_storage"): {"value": electrolyzer_to_storage},
            ("electrolyzer", "steel"): {"value": electrolyzer_to_steel},
            ("hydrogen_storage", "steel"): {"value": storage_to_steel},
        }

        nx.set_edge_attributes(G_dispatch, dispatch_IO)

        if False:
            layout = nx.random_layout(G_dispatch)
            nx.draw_networkx(G_dispatch, pos=layout, with_labels=True)
            nx.draw_networkx_edge_labels(
                G_dispatch, pos=layout, edge_labels=dispatch_IO
            )

        return G_dispatch
