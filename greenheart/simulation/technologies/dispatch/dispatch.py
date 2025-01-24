import numpy as np
import networkx as nx


class GreenheartDispatchConfig:
    def __init__(self):
        pass

class GreenheartDispatchOutput:
    def __init__(self):
        pass


class GreenheartDispatch:
    def __init__(self, hopp_interface, GHconfig, dispatch_config = None):
        
        
        self.setup_control_model(GHconfig)
        self.setup_constraints(GHconfig)
        
        self.setup_time_parameters(hopp_interface, GHconfig, dispatch_config)
        self.setup_objective(hopp_interface, GHconfig, dispatch_config)

        pass

        # extract the timeseries- and feedback-capabale component simulation models from hopp
        hopp_simulation_models = hopp_interface.hopp.system.technologies

        # extract or initialze the timeseries- and feedback-capabale component simulation models from greenheart
        GH_simulation_model_names = ["X", "Y", "Z"]
        GH_simulation_models = []
        for GH_model in GH_simulation_model_names:
            GH_simulation_models.append(GH_model)
        
        self.setup_simulation_model(hopp_simulation_models, GH_simulation_models)
        

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
        

    def step(self, G_dispatch, available_power):

        G_dispatch = self.example_control_elec_storage_steel(G_dispatch, available_power)
        # G_dispatch = self.example_control_hydrogen_heat(G_dispatch, available_power)
        return G_dispatch



    def example_control_hydrogen_heat(self, G_dispatch, available_power):

        for edge in list(G_dispatch.edges):
            G_dispatch.edges[edge].update({"value":available_power / 3})

        return G_dispatch

     

    def example_control_elec_storage_steel(self, G_dispatch, available_power):
        electrolyzer_eta = 1 / 55 # kg/kWh
        average_generation = 0.3 * (10 * 6000) + 0.3 * 100000 # wind + solar rated * 0.3 capacity factor

        # average_generation = 50000

        # Always send all of the power through the electrolyzer
        # If available is greater than average, charge the storage by the difference
        # If available is less than average, discharge the storage by the difference

        available_more_than_average_difference = np.max([0, available_power - average_generation])
        available_less_than_average_difference = np.max([0, average_generation - available_power])

        generation_to_curtail = 0
        generation_to_electrolyzer = available_power
        electrolyzer_to_storage = electrolyzer_eta * available_more_than_average_difference
        # electrolyzer_to_storage = electrolyzer_eta * (available_power - average_generation)

        electrolyzer_to_steel = electrolyzer_eta * average_generation
        storage_to_steel = electrolyzer_eta * available_less_than_average_difference
        
        

        assert not np.isnan(storage_to_steel)

        dispatch_edges = list(G_dispatch.edges)
        dispatch_IO = {
            ('generation', 'curtail'): {"value": generation_to_curtail}, 
            ('generation', 'electrolyzer'): {"value": generation_to_electrolyzer}, 
            ('electrolyzer', 'hydrogen_storage'): {"value": electrolyzer_to_storage}, 
            ('electrolyzer', 'steel'): {"value": electrolyzer_to_steel}, 
            ('hydrogen_storage', 'steel'): {"value": storage_to_steel}
        }

        nx.set_edge_attributes(G_dispatch, dispatch_IO)

        if False:
            layout = nx.random_layout(G_dispatch)
            nx.draw_networkx(G_dispatch, pos = layout, with_labels=True)
            nx.draw_networkx_edge_labels(G_dispatch, pos = layout, edge_labels=dispatch_IO)


        return G_dispatch
