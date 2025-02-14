import numpy as np

from hopp.simulation.technologies.battery.battery import Battery as Battery_hopp
from hopp.simulation.technologies.battery.battery import BatteryConfig

from greenheart.simulation.technologies.dispatch.control_model import ControlModel

class Battery():
# class Battery(Battery_hopp):
    def __init__(self, battery_config):

        # Need to get these from hopp config

        # config = BatteryConfig(*battery_config)


        self.roundtrip_efficiency = 1
    
        self.config = battery_config

        self.system_capacity_kwh = battery_config["system_capacity_kwh"]
        self.system_capacity_kw = battery_config["system_capacity_kw"]


        self.min_SOC = battery_config["minimum_SOC"]
        self.max_SOC = battery_config["maximum_SOC"]
        self.initial_SOC = battery_config["initial_SOC"]

        self.max_capacity_kWh = (self.max_SOC / 100) * self.system_capacity_kwh  # kWh
        # self.max_capacity_kWh = 2000000  # kWh
        self.min_capacity_kWh = (self.min_SOC / 100) * self.system_capacity_kwh
        # self.min_capacity_kWh = 0

        self.max_charge_rate_kW = self.system_capacity_kw
        self.max_discharge_rate_kW = self.system_capacity_kw

        self.dt = 1  # [hr] TODO initialize this timestep from elswhere in greenheart for consistency

        self.storage_state = (self.initial_SOC / 100) * self.system_capacity_kwh

        sim_duration = 8760
        self.store_storage_state = np.zeros(8760)
        self.store_charge_power = np.zeros(8760)

        self.control_model = self.create_control_model()
        
    def create_control_model(self):

        A = np.array([[1]])
        B = np.array([[1]])
        C = np.array([[0]])
        D = np.array([[-1]])
        E = np.array([[0]])
        F = np.array([[1]])

        bounds_dict = {
            "u_lb": np.array([-self.max_discharge_rate_kW]),
            "u_ub": np.array([self.max_charge_rate_kW]),
            "x_lb": np.array([self.min_capacity_kWh]),
            "x_ub": np.array([self.max_capacity_kWh]),
            "y_lb": np.array([None]),
            "y_ub": np.array([None]),
        }


        self.control_model = ControlModel(A, B, C, D, E, F, bounds=bounds_dict, discrete=True)


        pass

    def run(self):
        pass

    def input_output(self, available_power, desired_power):
        # input is charging massflow must be >= 0
        # dispatch signal is charging/discharging massflow + or -

        actual_power = self.low_level_controller(available_power, desired_power)
        actual_power = float(actual_power)

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

        assert (
            lower <= upper
        ), "Constraint logic gives a higher lower constraint than upper constraint"

        # Saturate desired at constraints
        control_power = desired_power
        if desired_power >= upper:
            control_power = upper

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
        self.store_step(output, step_index)

        if output <= 0:
            output = -output
        else:
            output = 0.0

        return output

    def store_step(self, charge_power, step_index):
        self.store_storage_state[step_index] = self.storage_state
        self.store_charge_power[step_index] = charge_power

    def compute_hydrogen_storage_capacity(self):
        self.capacity = np.max(self.storage_state)
        return self.storage_state

    def consolidate_simulation_outcome(self):
        pass


[]
