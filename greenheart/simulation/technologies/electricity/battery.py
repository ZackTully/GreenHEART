import numpy as np

from hopp.simulation.technologies.battery.battery import Battery as Battery_hopp


class Battery(Battery_hopp):
    def __init__(self):
        

        self.roundtrip_efficiency = 1
        self.max_capacity_kWh = 5e6 # kWh
        self.min_capacity_kWh = 0
        
        self.max_charge_rate_kW = 50000
        self.max_discharge_rate_kW = self.max_charge_rate_kW

        self.dt = 1 # [hr] TODO initialize this timestep from elswhere in greenheart for consistency

        self.storage_state = 0
        sim_duration = 8760
        self.store_storage_state = np.zeros(8760)


    def run(self):
        pass

    def input_output(self, available_power, desired_power):
        # input is charging massflow must be >= 0
        # dispatch signal is charging/discharging massflow + or -

        actual_power = self.low_level_controller(available_power, desired_power)

        self.update_storage_state(actual_power)
        return actual_power
        

    def update_storage_state(self, input_power):
        # Euler integration
        self.storage_state += input_power * self.dt





    def low_level_controller(self, available_power, desired_power):
        # check that the desired hydrogen massflow charging/discharging does not violate constraints or the available resource

        # Calculate the upper and lower limits of charge and discharge
        # TODO include losses from roundtrip efficiency


        # charge rate 
        upper1 = self.max_charge_rate_kW

        # cannot charge above max capacity
        upper2 = (self.max_capacity_kWh - self.storage_state) / self.dt

        # cannot charge with more than is available
        upper3 = available_power

        # find the most restrictive upper constraint
        upper = np.min([upper1, upper2, upper3])


        # discharge rate
        lower1 = -self.max_discharge_rate_kW

        # cannot discharge below min capacity 
        lower2 = (self.min_capacity_kWh - self.storage_state) / self.dt

        # find the most restrictive lower constraint
        lower = np.max([lower1, lower2])

        assert lower <= upper, "Constraint logic gives a higher lower constraint than upper constraint"

        # Saturate desired at constraints
        control_power = desired_power
        if desired_power >= upper:
            control_power= upper
        
        if desired_power <= lower:
            control_power = lower

        return control_power


    def step(self, input, dispatch, step_index):

        available_power = input

        # TODO: Better way to interpret dispatch signal

        # if (input > 0) & (dispatch == 0): # Charge
        #     desired_power = input
        # elif dispatch > 0:
        #     desired_power = -dispatch # positive means charging negative means discharging
        # else:
        #     desired_power = 0 # TODO this might come back around to bite

        desired_power = dispatch

        output = self.input_output(available_power, desired_power)
        self.store_step(step_index)

        if output <= 0:
            output = -output
        else:
            output = 0.0


        return output


    def store_step(self, step_index):
        self.store_storage_state[step_index] = self.storage_state


    def compute_hydrogen_storage_capacity(self):
        self.capacity = np.max(self.storage_state)
        return self.storage_state

    def consolidate_simulation_outcome(self):
        pass





[]