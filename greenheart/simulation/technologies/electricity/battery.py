import numpy as np

from hopp.simulation.technologies.battery.battery import Battery as Battery_hopp
from hopp.simulation.technologies.battery.battery import BatteryConfig
from hopp.simulation.technologies.dispatch.power_storage import (
    ExternallyDefinedBatteryDispatchHeuristic,
)

from greenheart.simulation.technologies.dispatch.control_model import ControlModel


class Battery:
    # class Battery(Battery_hopp):
    def __init__(self, config, battery_config, hopp_interface):

        # Need to get these from hopp config

        # config = BatteryConfig(*battery_config)

        # self.roundtrip_efficiency = 0.82

        self.use_hopp_outputs = True


        self.power_fraction = 0.9

        self.update_period = config.greenheart_config["realtime_simulation"][
            "dispatch"
        ]["update_period"]
        # self.horizon = config.greenheart_config['realtime_simulation']['dispatch']['mpc']['horizon']

        self.hopp_battery = Battery_hopp(
            site=hopp_interface.system.site, config=BatteryConfig(**battery_config)
        )
        self.hopp_battery.setup_performance_model()
        self.hopp_battery._dispatch = ExternallyDefinedBatteryDispatchHeuristic(
            pyomo_model=hopp_interface.system.dispatch_builder.pyomo_model,
            index_set=hopp_interface.system.dispatch_builder.pyomo_model.forecast_horizon,
            system_model=self.hopp_battery._system_model,
            financial_model=self.hopp_battery._financial_model,
            dispatch_options=hopp_interface.system.dispatch_builder.options
        )

        self.hopp_battery.dispatch.initialize_parameters()
        self.hopp_battery.dispatch.external_fixed_dispatch = np.zeros(8760 + config.greenheart_config['realtime_simulation']['dispatch']['mpc']['horizon'] + 24)


        # # self.hopp_battery.dispatch.external_fixed_dispatch = .7 * np.ones(8760)
        # # self.hopp_battery.dispatch.external_fixed_dispatch = 110 * np.ones(8760)
        # self.hopp_battery.dispatch.external_fixed_dispatch = np.concatenate([np.array([ 338166.36,  338166.36,  338166.36,  338166.36,  338166.36,  338166.36,  338166.36,  338166.36,  338166.36, -298530.5 , -195634.7 ,  338166.36,  338166.36,  338166.36,  338166.36,  338166.36,        338166.36,  338166.36,  338166.36,  195363.4 ,  169832.9 ,  192293.9 ,  235234.6 ,  203020.6 ]) / 1e3, 100 * np.ones(8760 - 24)])
        # self.hopp_battery.dispatch.set_fixed_dispatch(gen=1500 * np.ones(24), grid_limit = 20000 * np.ones(24), start_time=0)

        # self.hopp_battery.simulate_with_dispatch(n_periods=self.update_period, sim_start_time=0)

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

        self.max_charge_rate_kW = self.power_fraction * self.system_capacity_kw
        self.max_discharge_rate_kW = self.power_fraction * self.system_capacity_kw

        self.dt = 1  # [hr] TODO initialize this timestep from elswhere in greenheart for consistency

        self.storage_state = (self.initial_SOC / 100) * self.system_capacity_kwh

        sim_duration = 8760
        self.store_storage_state = np.zeros(8760)
        self.store_charge_power = np.zeros(8760)

        self.control_model = self.create_control_model()

    def create_control_model(self):

        eta_bes = 0.98

        A = np.array([[1]])
        B = np.array([[0.980876013779761, -0.9335137103501339]])
        # B = np.array([[0.980876013779761, -1/(0.9335137103501339 + 0.05)]])
        # B = np.array([[eta_bes, -1]])
        E = np.array([[0]])

        C = np.array([[0], [0]])
        D = np.array([[0, 0.9935416519391084], [-1, 0]])
        # D = np.array([[0, eta_bes], [-1, 0]])
        F = np.array([[0], [1]])
        # data_ss = np.array([[ 9.23946441e-01, -1.14887702e-02,  2.38266353e-01],
        #     [ 1.07312790e-04, -1.29614042e-04,  9.98602962e-01]])


        # A = np.array([data_ss[0,0, None]])
        # B = np.array([data_ss[0, 1:]])
        # E = np.array([data_ss[0, 1, None]])

        # C = np.array([[data_ss[1, 0]], [0]])
        # D = np.array([data_ss[1, 1:], [-1, 0]])
        # F = np.array([[data_ss[1,1]], [1]])

        bounds_dict = {
            "u_lb": np.array([0, 0]),
            "u_ub": np.array([self.max_charge_rate_kW, self.max_discharge_rate_kW]),
            "x_lb": np.array([self.min_capacity_kWh]),
            "x_ub": np.array([self.max_capacity_kWh]),
            "y_lb": np.array([0, 0]),
            "y_ub": np.array([None, None]),
        }

        control_model = ControlModel(
            A, B, C, D, E, F, bounds=bounds_dict, discrete=True
        )

        control_model.constraints(y_position=[1], constraint_type=["greater"])

        control_model.set_disturbance_domain([1, 0, 0])
        control_model.set_output_domain([1, 0, 0])
        control_model.set_disturbance_reshape(np.array([[1, 0, 0]]))

        return control_model

    def run(self):
        pass

    # def input_output(self, available_power, desired_power):
    #     # input is charging massflow must be >= 0
    #     # dispatch signal is charging/discharging massflow + or -

    #     u_model, u_passthrough, u_curtail = self.low_level_controller(available_power, desired_power)
    #     u_model = float(u_model)

    #     self.update_storage_state(u_model)
    #     return u_model

    def update_storage_state(self, input_power, step_index=None):
        if self.use_hopp_outputs:
            self.storage_state = self.hopp_battery.outputs.SOC[step_index] / 100 * self.max_capacity_kWh
        else:
            # Euler integration
            self.storage_state += input_power * self.dt

    def low_level_controller(self, available_power, desired_power):
        # check that the desired hydrogen massflow charging/discharging does not violate constraints or the available resource

        # Calculate the upper and lower limits of charge and discharge
        # TODO include losses from roundtrip efficiency

        desired_setpoint = desired_power  # + available_power

        # TODO catch an extra case when there would be passthrough but desired hits one of the limits

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
            lower <= upper + 0.1
        ), "Constraint logic gives a higher lower constraint than upper constraint"

        # Saturate desired at constraints
        control_power = desired_setpoint
        if desired_setpoint >= upper:
            control_power = upper

        if desired_setpoint <= lower:
            control_power = lower
        # control_power = desired_power
        # if desired_power >= upper:
        #     control_power = upper

        # if desired_power <= lower:
        #     control_power = lower

        # return u_model, u_passthrough, u_curtail

        u_model = control_power
        model_output = np.max([0, -control_power])
        # u_passthrough = available_power - np.abs(u_model)
        # u_passthrough = np.max([available_power - model_output, 0])
        u_passthrough = available_power - np.max([control_power, 0])
        u_curtail = 0.0

        return model_output, u_model, u_passthrough, u_curtail
        # return control_power


    def step_hopp_battery(self, available_power, desired_power, step_index):
        
        # self.hopp_battery.dispatch.external_fixed_dispatch = np.concatenate([np.array([ 338166.36,  338166.36,  338166.36,  338166.36,  338166.36,  338166.36,  338166.36,  338166.36,  338166.36, -298530.5 , -195634.7 ,  338166.36,  338166.36,  338166.36,  338166.36,  338166.36,        338166.36,  338166.36,  338166.36,  195363.4 ,  169832.9 ,  192293.9 ,  235234.6 ,  203020.6 ]) / 1e3, 100 * np.ones(8760 - 24)])
        self.hopp_battery.dispatch.external_fixed_dispatch[step_index] = -desired_power / 1e3
        self.hopp_battery.dispatch.set_fixed_dispatch(gen=1e-3 * available_power * np.ones(24), grid_limit = 1e9 * np.ones(24), start_time=step_index)

        self.hopp_battery.simulate_with_dispatch(n_periods=1, sim_start_time=step_index)
        
        P_battery = self.hopp_battery.outputs.P[step_index]
        model_output = np.max([P_battery, 0])
        assert available_power >= 0, "Sloppy but available should be positive"


        if available_power <= -P_battery:
            if (available_power  - -P_battery) < 1:
                u_passthrough = 0 
            else:
                u_passthrough = -1e5
                assert False, "This case shouldn't happen"
            u_curtail = 0.0
        else:
            if desired_power >= 0:
                charging_error = desired_power + P_battery
                u_passthrough = 0.0
                u_curtail = np.max([0, available_power]) - np.max([-P_battery, 0])
            else:
                u_passthrough = np.max([0, available_power]) - np.max([-P_battery, 0])
                u_curtail = 0.0

        return model_output, u_passthrough, u_curtail
        []

    def step(self, input, dispatch, step_index):

        if isinstance(input, (np.ndarray, list)):
            available_power = input[0]
        else:
            available_power = input



        if isinstance(dispatch, (np.ndarray, list)):
            if len(dispatch) == 1:
                desired_power = dispatch[0]
            else:
                desired_power = dispatch[0] - dispatch[1]
        else:
            desired_power = dispatch

        hopp_output, hopp_passthrough, hopp_curtail = self.step_hopp_battery(available_power, desired_power, step_index)

        model_output, u_model, u_passthrough, u_curtail = self.low_level_controller(
            available_power, desired_power
        )

        u_model = float(u_model)
        self.update_storage_state(u_model, step_index)
        self.store_step(u_model, step_index)
        output = model_output

        if self.use_hopp_outputs:
            return hopp_output, hopp_passthrough, hopp_curtail
        else:
            return output, u_passthrough, u_curtail

    def store_step(self, charge_power, step_index):
        self.store_storage_state[step_index] = self.storage_state
        self.store_charge_power[step_index] = charge_power

    def compute_hydrogen_storage_capacity(self):
        self.capacity = np.max(self.storage_state)
        return self.storage_state

    def consolidate_simulation_outcome(self):
        pass


if __name__ == "__main__":
    pass

    bes = Battery()

    t_start = 0
    t_stop = 100

    time = np.arange(t_start, t_stop)
