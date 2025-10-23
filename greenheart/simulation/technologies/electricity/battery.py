import numpy as np

from hopp.simulation.technologies.battery.battery import Battery as Battery_hopp
from hopp.simulation.technologies.battery.battery import BatteryConfig
from hopp.simulation.technologies.dispatch.power_storage import (
    ExternallyDefinedBatteryDispatchHeuristic,
)

from greenheart.simulation.technologies.dispatch.control_model import ControlModel

# For debugging printouts
# np.set_printoptions(legacy="1.25")


class Battery:
    def __init__(self, config, battery_config, hopp_interface):

        self.use_hopp_outputs = True
        self.power_fraction = 0.9

        self.hopp_battery = Battery_hopp(
            site=hopp_interface.system.site, config=BatteryConfig(**battery_config)
        )
        self.hopp_battery.setup_performance_model()
        if hasattr(hopp_interface.system, "dispatch_builder"):
            self.hopp_battery._dispatch = ExternallyDefinedBatteryDispatchHeuristic(
                pyomo_model=hopp_interface.system.dispatch_builder.pyomo_model,
                index_set=hopp_interface.system.dispatch_builder.pyomo_model.forecast_horizon,
                system_model=self.hopp_battery._system_model,
                financial_model=self.hopp_battery._financial_model,
                dispatch_options=hopp_interface.system.dispatch_builder.options,
            )

            self.hopp_battery.dispatch.initialize_parameters()
            self.hopp_battery.dispatch.external_fixed_dispatch = np.zeros(8760 + 2190)

        self.config = battery_config

        # System parameters
        self.system_capacity_kwh = battery_config["system_capacity_kwh"]
        self.system_capacity_kw = battery_config["system_capacity_kw"]

        self.min_SOC = battery_config["minimum_SOC"]
        self.max_SOC = battery_config["maximum_SOC"]
        self.initial_SOC = battery_config["initial_SOC"]

        self.max_capacity_kWh = (self.max_SOC / 100) * self.system_capacity_kwh  # kWh
        self.min_capacity_kWh = (self.min_SOC / 100) * self.system_capacity_kwh

        self.max_charge_rate_kW = self.power_fraction * self.system_capacity_kw
        self.max_discharge_rate_kW = self.power_fraction * self.system_capacity_kw

        self.dt = 1  # [hr] TODO initialize this timestep from elswhere in greenheart for consistency

        self.storage_state = (self.initial_SOC / 100) * self.system_capacity_kwh

        # Setup bookkeeping
        sim_duration = 8760
        self.store_storage_state = np.zeros(sim_duration)
        self.store_charge_power = np.zeros(sim_duration)

        self.control_model = self.create_control_model()

    def get_state_measurement(self, step_index, first_step=False):
        if self.use_hopp_outputs:
            hb = self.hopp_battery
            conf = hb.config
            if first_step:
                state = conf.initial_SOC / 100 * conf.system_capacity_kwh
            else:
                # min_soc_violation = (hb.outputs.SOC[step_index - 1] - hb._system_model.ParamsCell.minimum_SOC)
                state = hb.outputs.SOC[step_index - 1] / 100 * conf.system_capacity_kwh
        else:
            state = self.storage_state
        return state

    def run(self):
        pass

    def update_storage_state(self, input_power, step_index=None):
        if self.use_hopp_outputs:
            self.storage_state = (
                self.hopp_battery.outputs.SOC[step_index] / 100 * self.max_capacity_kWh
            )
        else:
            # Euler integration
            self.storage_state += input_power * self.dt

    def low_level_controller(self, available_power, desired_power):
        # Calculate the upper and lower limits of charge and discharge

        desired_setpoint = desired_power  # + available_power

        # 1. charge rate
        # 2. cannot charge above max capacity
        # 3. 4. cannot charge with more than is available
        upper1 = self.max_charge_rate_kW
        upper2 = 1.0 * (1.0 * self.max_capacity_kWh - self.storage_state) / self.dt
        upper3 = available_power
        upper4 = (
            (100 - self.hopp_battery._system_model.StatePack.SOC) / 100
        ) * self.system_capacity_kwh

        # find the most restrictive upper constraint
        upper = np.min([upper1, upper2, upper3, upper4])

        # 1. discharge rate
        # 2. cannot discharge below min capacity
        lower1 = -self.max_discharge_rate_kW
        lower2 = 1.0 * (1.0 * self.min_capacity_kWh - self.storage_state) / self.dt

        # find the most restrictive lower constraint
        lower = np.max([lower1, lower2])

        assert (
            lower <= upper + 0.1
        ), "Constraint logic gives a higher lower constraint than upper constraint"

        # Saturate desired power at constraints
        control_power = desired_setpoint
        if desired_setpoint >= upper:
            control_power = upper

        if desired_setpoint <= lower:
            control_power = lower

        # u_model used for both non-hopp and hopp model
        u_model = control_power

        # model_output and u_passthrough only for non-hopp model
        model_output = np.max([0, -control_power])
        u_passthrough = available_power - np.max([control_power, 0])
        u_curtail = 0.0

        return model_output, u_model, u_passthrough, u_curtail

    def step_hopp_battery(self, available_power, desired_power, step_index):

        current_SOC = self.hopp_battery._system_model.StatePack.SOC
        next_state = current_SOC / 100 * self.max_capacity_kWh + desired_power
        next_SOC_guess = (next_state) / self.max_capacity_kWh * 100

        # Set hopp battery dispatch
        self.hopp_battery.dispatch.external_fixed_dispatch[step_index] = (
            -desired_power / 1e3
        )

        # Trick controller that there is always enough grid 
        self.hopp_battery.dispatch.set_fixed_dispatch(
            gen=1e-3 * available_power * np.ones(24),
            grid_limit=1e9 * np.ones(24),
            start_time=step_index,
        )

        # Simulate battery with custom dispatch controller
        self.hopp_battery.simulate_with_dispatch(n_periods=1, sim_start_time=step_index)

        next_SOC_real = self.hopp_battery._system_model.StatePack.SOC

        # Power out is positive if discharging, negative if charging. Opposite of the 
        # sign convention for the rest of this battery class. Model output is only the 
        # discharging power.
        P_battery = self.hopp_battery.outputs.P[step_index]
        model_output = np.max([P_battery, 0])

        # # Checking for errors
        # sign_ctrl = np.sign(desired_power)
        # sign_model = np.sign(P_battery)

        # sign_agreement = -sign_model != sign_ctrl
        # absolute_difference = np.abs(np.abs(P_battery) - np.abs(desired_power))
        # percent_difference = (
        #     100
        #     * np.abs(np.abs(P_battery) - np.abs(desired_power))
        #     / (0.5 * (np.abs(P_battery) + np.abs(desired_power)))
        # )

       

        if available_power <= -P_battery:
            # This case indicates the battery charged with more power than available.
            if (available_power - -P_battery) < 1:
                # If the error is smaller than 1 kW, then call it numerical error.
                u_passthrough = 0
            else:
                u_passthrough = -1e5
                assert False, "This case shouldn't happen"
            u_curtail = 0.0
        else:
            if desired_power >= 0:
                # Battery is charging so extra input power should be curtailed
                u_passthrough = 0.0
                unused_available = np.max([0, available_power]) - np.max(
                    [-P_battery, 0]
                )
                unexpected_discharge = np.max([P_battery, 0])
                u_curtail = np.max([0, available_power]) - np.max([-P_battery, 0])
            else:
                # Battery is discharging so extra input power (input power should be 0) is passed through to downstream systems
                u_passthrough = np.max([0, available_power]) - np.max([-P_battery, 0])
                u_curtail = 0.0

        # if sign_agreement:
        #     pass

        # if absolute_difference > 200:
        #     pass

        # if percent_difference > 10:
        #     pass

        # if available_power > 10:
        #     if u_passthrough > 0.1 * available_power:
        #         pass
        #     if u_curtail > 0.1 * np.abs(desired_power):
        #         pass

        return model_output, u_passthrough, u_curtail

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

        model_output, u_model, u_passthrough, u_curtail = self.low_level_controller(
            available_power, desired_power
        )

        hopp_output, hopp_passthrough, hopp_curtail = self.step_hopp_battery(
            available_power, u_model, step_index
        )

        ## debugging
        # tol = 0.1
        # if available_power > 100:
        #     if (
        #         (u_passthrough > tol * available_power)
        #         or (u_curtail > tol * available_power)
        #         or (hopp_passthrough > tol * available_power)
        #         or (hopp_curtail > tol * available_power)
        #     ):
        #         # print(f"{available_power = :.4f}".rjust(100))
        #         # print(f"{desired_power = :.4f}".rjust(100))
        #         # print(f"{u_model = :.4f}".rjust(100))
        #         # print(f"{u_passthrough = :.4f}".rjust(100))
        #         # print(f"{u_curtail = :.4f}".rjust(100))
        #         # print(f"{hopp_passthrough = :.4f}".rjust(100))
        #         # print(f"{hopp_curtail = :.4f}".rjust(100))
        #         # print(f"{hopp_output = :.4f}".rjust(100))
        #         []

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

    def create_control_model(self):

        eta_bes = 0.98

        A = np.array([[1]])
        B = np.array([[eta_bes, -1 / eta_bes]])
        # B = np.array([[0.980876013779761, -0.9335137103501339]])
        # B = np.array([[0.980876013779761, -1/(0.9335137103501339 + 0.05)]])
        # B = np.array([[eta_bes, -1]])
        E = np.array([[0]])

        C = np.array([[0], [0]])
        D = np.array([[0, 1], [-1, 0]])
        # D = np.array([[0, 0.9935416519391084], [-1, 0]])
        # D = np.array([[0, eta_bes], [-1, 0]])
        F = np.array([[0], [1]])
        # data_ss = np.array([[ 9.23946441e-01, -1.14887702e-02,  2.38266353e-01],
        #     [ 1.07312790e-04, -1.29614042e-04,  9.98602962e-01]])

        bounds_dict = {
            "u_lb": np.array([0, 0]),
            "u_ub": np.array([self.max_charge_rate_kW, self.max_discharge_rate_kW]),
            "x_lb": np.array([1.00 * self.min_capacity_kWh]),
            "x_ub": np.array([1.00 * self.max_capacity_kWh]),
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


if __name__ == "__main__":
    from greenheart.simulation.greenheart_simulation import (
        setup_greenheart_simulation,
        GreenHeartSimulationConfig,
    )
    from pathlib import Path

    config_root = Path(
        "/Users/ztully/Documents/hybrids_code/GH_scripts/greenheart_scripts/minnesota_reference_design/configs/forecast_interp"
    )

    config = GreenHeartSimulationConfig(
        str(config_root / "plant" / "hopp_config.yaml"),
        str(config_root / "plant" / "greenheart_config.yaml"),
        str(config_root / "turbines" / "ATB2024_6MW_170RD_floris_turbine.yaml"),
        str(config_root / "floris" / "floris_input_lbw_6MW.yaml"),
        verbose=False,
        show_plots=False,
        save_plots=False,
        use_profast=True,
        post_processing=True,
        incentive_option=1,
        plant_design_scenario=1,
        output_level=8,
    )

    config, hi, wind_cost_results = setup_greenheart_simulation(config=config)

    config = {}
    battery_config = {
        "fin_model": {
            "cp_capacity_credit_percent": [0],
            "degradation": [0],
            "financial_parameters": {
                "admin_expense_percent_of_sales": 0.0,
                "analysis_start_year": 2030,
                "capital_gains_tax_rate": 15.0,
                "debt_percent": 75.4,
                "debt_type": "Revolving debt",
                "depreciation_method": "MACRS",
                "depreciation_period": 7,
                "federal_tax_rate": 21.0,
                "inflation_rate": 0.0,
                "installation_months": 36,
                "insurance_rate": 1.0,
                "months_working_reserve": 1,
                "property_tax_rate": 2.0,
                "real_discount_rate": 6.6,
                "sales_tax_rate_state": 0.0,
                "state_tax_rate": 4.74,
                "term_int_rate": 4.4,
            },
            "revenue": {"ppa_escalation": 0, "ppa_price_input": [...]},
            "system_costs": {
                "om_batt_capacity_cost": 0,
                "om_batt_fixed_cost": 0,
                "om_batt_replacement_cost": 0,
                "om_batt_variable_cost": [0],
                "om_capacity": [15.525],
                "om_fixed": [0],
                "om_production": [0],
                "om_replacement_cost_escal": 0,
            },
            "system_use_lifetime_output": 0,
        },
        "initial_SOC": 90.0,
        "maximum_SOC": 100.0,
        "minimum_SOC": 20.0,
        "system_capacity_kw": 108000,
        "system_capacity_kwh": 2000000,
    }

    bes = Battery(config, battery_config, hi)

    for k in range(8760):
        print(f"\r{k}               ", end="")
        bes.step(input=100e3, dispatch=np.array([100e3, 0]), step_index=k)

    []
