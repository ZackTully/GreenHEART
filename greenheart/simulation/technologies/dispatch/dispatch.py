

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
    


    def step_simulation_model(self):
        # Either this lives in the controller or is attributed to the controller. 
        pass
    
    def get(self, data_name):
        
        return getattr(self, data_name)
        
