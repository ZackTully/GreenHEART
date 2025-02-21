import numpy as np

from greenheart.simulation.technologies.dispatch.control_model import ControlModel

class HydrogenStorage:
    def __init__(self):




        self.roundtrip_efficiency = 1
        self.max_capacity_kg = 812209.38400328 # kg
        # self.max_capacity_kg = 3e5 # kg
        self.min_capacity_kg = 0
        
        self.max_charge_rate_kg_hr = 3e3
        self.max_discharge_rate_kg_hr = self.max_charge_rate_kg_hr

        self.dt = 1 # [hr] TODO initialize this timestep from elswhere in greenheart for consistency


        self.storage_state = 1.541e5

        sim_duration = 8760

        self.store_storage_state = np.zeros(8760)
        self.store_charge = np.zeros(8760)
        self.create_control_model()


    def create_control_model(self):

        A = np.array([[1]])
        B = np.array([[1]])
        C = np.array([[0]])
        D = np.array([[-1]])
        E = np.array([[0]])
        F = np.array([[1]])

        bounds_dict = {
            "u_lb": np.array([-self.max_discharge_rate_kg_hr]),
            "u_ub": np.array([self.max_charge_rate_kg_hr]),
            "x_lb": np.array([0]),
            "x_ub": np.array([self.max_capacity_kg]),
            "y_lb": np.array([None]),
            "y_ub": np.array([None]),
        }


        self.control_model = ControlModel(A, B, C, D, E, F, bounds=bounds_dict, discrete=True)
     
        []



    def run(self):
        pass

    # def input_output(self, available_massflow, desired_massflow):
    #     # input is charging massflow must be >= 0
    #     # dispatch signal is charging/discharging massflow + or -

    #     actual_massflow = self.low_level_controller(available_massflow, desired_massflow)

    #     self.update_storage_state(actual_massflow)
    #     return actual_massflow
        

    def update_storage_state(self, input_massflow):
        # Euler integration
        self.storage_state += input_massflow * self.dt





    def low_level_controller(self, available_massflow, desired_massflow):
        # check that the desired hydrogen massflow charging/discharging does not violate constraints or the available resource

        # Calculate the upper and lower limits of charge and discharge
        # TODO include losses from roundtrip efficiency



        desired_setpoint = desired_massflow #+ available_massflow

        # charge rate 
        upper1 = self.max_charge_rate_kg_hr

        # cannot charge above max capacity
        upper2 = (self.max_capacity_kg - self.storage_state) / self.dt

        # cannot charge with more than is available
        upper3 = available_massflow

        # find the most restrictive upper constraint
        upper = np.min([upper1, upper2, upper3])


        # discharge rate
        lower1 = -self.max_discharge_rate_kg_hr

        # cannot discharge below min capacity 
        lower2 = (self.min_capacity_kg - self.storage_state) / self.dt

        # find the most restrictive lower constraint
        lower = np.max([lower1, lower2])

        assert lower <= upper, "Constraint logic gives a higher lower constraint than upper constraint"

        # Saturate desired at constraints
        control_massflow = desired_setpoint
        # control_massflow = desired_massflow
        if desired_setpoint >= upper:
            control_massflow = upper
        
        if desired_setpoint <= lower:
            control_massflow = lower


        u_model = control_massflow
        model_output = np.max([0, -control_massflow])
        # u_passthrough = available_massflow - np.abs(u_model)
        # u_passthrough = np.max([available_massflow - model_output, 0])
        u_passthrough = available_massflow - np.max([control_massflow, 0])
        u_curtail = 0.0

        return model_output, u_model, u_passthrough, u_curtail
        # return control_massflow


    def step(self, h2_input, dispatch, step_index):


        if isinstance(dispatch, (np.ndarray, list)):
            desired_massflow = dispatch[0]


        available_massflow = h2_input[0]
        available_temperature = h2_input[1]




        # output = self.input_output(available_massflow, desired_massflow)
        model_output, u_model, u_passthrough, u_curtail = self.low_level_controller(available_massflow, desired_massflow)

        self.update_storage_state(u_model)
        self.store_step(step_index, u_model)

    

        return model_output, u_passthrough, u_curtail


    def store_step(self, step_index, charging_massflow):
        self.store_storage_state[step_index] = self.storage_state
        self.store_charge[step_index] = charging_massflow


    def compute_hydrogen_storage_capacity(self):
        self.capacity = np.max(self.storage_state)
        return self.storage_state

    def consolidate_simulation_outcome(self):
        pass
