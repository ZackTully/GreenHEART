import numpy as np


class HydrogenStorage:
    def __init__(self):




        self.roundtrip_efficiency = 1
        self.max_capacity_kg = 3e5 # kg
        self.min_capacity_kg = 0
        
        self.max_charge_rate_kg_hr = 1e6
        self.max_discharge_rate_kg_hr = self.max_charge_rate_kg_hr

        self.dt = 1 # [hr] TODO initialize this timestep from elswhere in greenheart for consistency


        self.storage_state = 0

        sim_duration = 8760

        self.store_storage_state = np.zeros(8760)


    def run(self):
        pass

    def input_output(self, available_massflow, desired_massflow):
        # input is charging massflow must be >= 0
        # dispatch signal is charging/discharging massflow + or -

        actual_massflow = self.low_level_controller(available_massflow, desired_massflow)

        self.update_storage_state(actual_massflow)
        return actual_massflow
        

    def update_storage_state(self, input_massflow):
        # Euler integration
        self.storage_state += input_massflow * self.dt





    def low_level_controller(self, available_massflow, desired_massflow):
        # check that the desired hydrogen massflow charging/discharging does not violate constraints or the available resource

        # Calculate the upper and lower limits of charge and discharge
        # TODO include losses from roundtrip efficiency


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
        control_massflow = desired_massflow
        if desired_massflow >= upper:
            control_massflow = upper
        
        if desired_massflow <= lower:
            control_massflow = lower

        return control_massflow


    def step(self, h2_input, dispatch, step_index):

        dispatch = dispatch

        available_massflow = h2_input[0]
        available_temperature = h2_input[1]

        # TODO: Better way to interpret dispatch signal

        # if (available_massflow > 0) & (dispatch == 0): # Charge
        #     desired_massflow = available_massflow
        # if dispatch > 0:
        #     desired_massflow = -dispatch # positive means charging negative means discharging
        # else: 
        #     desired_massflow = 0 # TODO check back in on this

        desired_massflow = dispatch

        output = self.input_output(available_massflow, desired_massflow)
        
        self.store_step(step_index)

        return output


    def store_step(self, step_index):
        self.store_storage_state[step_index] = self.storage_state


    def compute_hydrogen_storage_capacity(self):
        self.capacity = np.max(self.storage_state)
        return self.storage_state

    def consolidate_simulation_outcome(self):
        pass
