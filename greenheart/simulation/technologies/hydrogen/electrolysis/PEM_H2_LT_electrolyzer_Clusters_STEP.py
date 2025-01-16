import numpy as np
import pandas as pd


from greenheart.simulation.technologies.hydrogen.electrolysis.PEM_H2_LT_electrolyzer_Clusters import (
    PEM_H2_Clusters,
    calc_current,
)


class PEM_H2_Clusters_Step(PEM_H2_Clusters):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.cluster_status = 1

        # Initialize attributes that need to be saved after each step


        # degradation update period = 250 
        self.degradation_update_period = 250 #[hrs]
        self.update_degradation_index = 5


        sim_duration = 8760

        self.store_input_external_power_kw = np.zeros(sim_duration)
        self.store_input_power_kw = np.zeros(sim_duration)
        self.store_cluster_status = np.zeros(sim_duration)
        self.store_stack_current = np.zeros(sim_duration)
        self.store_V_init = np.zeros(sim_duration)
        self.store_V_cell = np.zeros(sim_duration)
        self.store_stack_power_consumed = np.zeros(sim_duration)
        self.store_system_power_consumed = np.zeros(sim_duration)
        self.store_h2_kg_hr_system_init = np.zeros(sim_duration)
        self.store_h2_kg_hr_system = np.zeros(sim_duration)
        self.store_deg_signal = np.zeros(sim_duration)
        self.store_power_per_stack = np.zeros(sim_duration)
        self.n_stacks_op = np.zeros(sim_duration)

    def run(self, power_to_cluster):
        # replaces the run method in the non-step PEM_cluster class
        
        for i in range(len(power_to_cluster)):
            self.step(power_to_cluster[i], step_index = i)
        
        h2_ts, h2_tot = self.consolidate_sim_outcome()
        return h2_ts, h2_tot


    def step(self, input_external_power_kw, step_index):
        startup_time = 600  # [sec]
        startup_ratio = 1 - (startup_time / self.dt)

        # if the cluster starts the step with status = 1 then no penalty
        if self.cluster_status:
            h2_multiplier = 1
        else:  # If the cluster starts the step with status = 0 then it was off and either it is still off or it has changed to on. Either way, apply the startup penalty
            h2_multiplier = startup_ratio

        # Saturate power at rated power
        input_power_kw = self.external_power_supply(input_external_power_kw)

        self.cluster_status = self.system_design(input_power_kw, self.max_stacks)

        # dont forget to count cluster cycles

        self.current_n_stacks_op = self.max_stacks * self.cluster_status
        if self.current_n_stacks_op > 0:
            power_per_stack = input_power_kw / self.current_n_stacks_op
        else:
            power_per_stack = 0

        stack_current = calc_current((power_per_stack, self.T_C), *self.curve_coeff)




        # every time record step is called, update the degradation level
        voltage_final, deg_signal = self.update_degradation(step_index)

        V_init = self.cell_design(self.T_C, stack_current)

        if step_index > 1:
            # stack_current = stack_current[-1]
            deg_signal = deg_signal[-1]

        if self.include_deg_penalty:
            stack_current = self.find_equivalent_input_power_4_deg(
                power_per_stack, V_init, deg_signal
            )


            V_cell_equiv = self.cell_design(self.T_C, stack_current)
            V_cell = V_cell_equiv + deg_signal
            # if step_index <=1 :
            #     V_cell = V_cell_equiv + deg_signal
            # else:
            #     V_cell = V_cell_equiv[-1] + deg_signal[-1]
        else:

            V_cell = V_init

        stack_power_consumed = (stack_current * V_cell * self.N_cells) / 1000
        system_power_consumed = self.current_n_stacks_op * stack_power_consumed
        h2_kg_hr_system_init = self.h2_production_rate(stack_current, self.current_n_stacks_op)
        h2_kg_hr_system = h2_kg_hr_system_init * h2_multiplier

        # Water used?

        self.record_step(
            step_index,
            input_external_power_kw,
            input_power_kw,
            self.cluster_status,
            stack_current,
            V_init,
            V_cell,
            stack_power_consumed,
            system_power_consumed,
            h2_kg_hr_system_init,
            h2_kg_hr_system,
            deg_signal,
            power_per_stack,
            self.current_n_stacks_op
        )

        return h2_kg_hr_system_init

    def record_step(
        self,
        step_index,
        input_external_power_kw,
        input_power_kw,
        cluster_status,
        stack_current,
        V_init,
        V_cell,
        stack_power_consumed,
        system_power_consumed,
        h2_kg_hr_system_init,
        h2_kg_hr_system,
        deg_signal,
        power_per_stack,
        current_n_stacks_op
    ):
        self.store_input_external_power_kw[step_index] = input_external_power_kw
        self.store_input_power_kw[step_index] = input_power_kw
        self.store_cluster_status[step_index] = cluster_status
        self.store_stack_current[step_index] = stack_current
        self.store_V_init[step_index] = V_init
        self.store_V_cell[step_index] = V_cell
        self.store_stack_power_consumed[step_index] = stack_power_consumed
        self.store_system_power_consumed[step_index] = system_power_consumed
        self.store_h2_kg_hr_system_init[step_index] = h2_kg_hr_system_init
        self.store_h2_kg_hr_system[step_index] = h2_kg_hr_system
        self.store_deg_signal[step_index] = deg_signal
        self.store_power_per_stack[step_index] = power_per_stack
        self.n_stacks_op[step_index] = current_n_stacks_op

    def update_degradation(self, step_index):

        if step_index <= 1:
            return 0, 0
        
        if step_index == self.update_degradation_index:
            self.update_degradation_index += self.degradation_update_period
            voltage_final, deg_signal = self.full_degradation(
                self.store_V_init[0:step_index], step_index
            )
        else:
            voltage_final = 0
            deg_signal = self.store_deg_signal



        # if step_index > 1:
        #     voltage_final, deg_signal = self.full_degradation(
        #         self.store_V_init[0:step_index], step_index
        #     )
        # else:
        #     voltage_final = 0
        #     deg_signal = 0
        return voltage_final, deg_signal

    def full_degradation(self, voltage_signal, step_index):
        # TODO: add reset if hits end of life degradation limit!
        voltage_signal = voltage_signal * self.cluster_status
        if self.use_uptime_deg:
            V_deg_uptime = self.calc_uptime_degradation(voltage_signal)
        else:
            V_deg_uptime = np.zeros(len(voltage_signal))
        if self.use_onoff_deg:
            V_deg_onoff = self.calc_onoff_degradation(self.store_cluster_status[0:step_index])
        else:
            V_deg_onoff = np.zeros(len(voltage_signal))

        V_signal = voltage_signal + np.cumsum(V_deg_uptime) + np.cumsum(V_deg_onoff)
        if self.use_fatigue_deg:
            V_fatigue = self.approx_fatigue_degradation(V_signal)
        else:
            V_fatigue = np.zeros(len(voltage_signal))
        deg_signal = np.cumsum(V_deg_uptime) + np.cumsum(V_deg_onoff) + V_fatigue

        self.cumulative_Vdeg_per_hr_sys = deg_signal
        voltage_final = voltage_signal + deg_signal

        # self.output_dict['Cumulative Degradation Breakdown']=pd.DataFrame({'Uptime':np.cumsum(V_deg_uptime),'On/off':np.cumsum(V_deg_onoff),'Fatigue':V_fatigue})
        return voltage_final, deg_signal

    def calc_onoff_degradation(self, cluster_status):

        change_stack = np.diff(cluster_status)
        cycle_cnt = np.where(change_stack < 0, -1 * change_stack, 0)
        cycle_cnt = np.array([0] + list(cycle_cnt))
        self.off_cycle_cnt = cycle_cnt
        stack_off_deg_per_hr = self.onoff_deg_rate * cycle_cnt
        self.output_dict["System Cycle Degradation [V]"] = np.cumsum(
            stack_off_deg_per_hr
        )[-1]
        self.output_dict["Off-Cycles"] = cycle_cnt
        return stack_off_deg_per_hr

    def consolidate_sim_outcome(self):

        voltage_final, deg_signal = self.full_degradation(self.store_V_init, 8760)
        self.store_deg_signal = deg_signal

        p_consumed_max, rated_h2_hr = self.rated_h2_prod()
        h20_gal_used_system = self.water_supply(self.store_h2_kg_hr_system)
        pem_cf = np.sum(self.store_h2_kg_hr_system) / (
            rated_h2_hr * len(self.store_input_power_kw) * self.max_stacks
        )
        efficiency = self.system_efficiency(
            self.store_input_power_kw, self.store_stack_current
        )  # Efficiency as %-HHV

        # NOTE: this is really sloppy
        self.current_cluster_states = self.cluster_status
        self.cluster_status = self.store_cluster_status
        time_until_replacement, stack_life = self.calc_stack_replacement_info(self.store_deg_signal)
        annual_performance = self.make_yearly_performance_dict(self.store_power_per_stack, self.store_deg_signal, self.store_V_init, I_op=[],grid_connected=False)

        h2_results = {}
        h2_results_aggregates = {}
        h2_results["Input Power [kWh]"] = self.store_input_external_power_kw
        h2_results["hydrogen production no start-up time"] = self.store_h2_kg_hr_system_init
        h2_results["hydrogen_hourly_production"] = self.store_h2_kg_hr_system
        h2_results["water_hourly_usage_kg"] = h20_gal_used_system * 3.79
        h2_results["electrolyzer_total_efficiency_perc"] = efficiency
        h2_results["kwh_per_kgH2"] = self.store_input_power_kw / self.store_h2_kg_hr_system
        h2_results["Power Consumed [kWh]"] = self.store_system_power_consumed

        h2_results_aggregates["Warm-Up Losses on H2 Production"] = np.sum(
            self.store_h2_kg_hr_system_init
        ) - np.sum(self.store_h2_kg_hr_system)

        h2_results_aggregates["Stack Life [hours]"] = stack_life
        h2_results_aggregates["Time until replacement [hours]"] = time_until_replacement
        h2_results_aggregates["Stack Rated Power Consumed [kWh]"] = p_consumed_max
        h2_results_aggregates["Stack Rated H2 Production [kg/hr]"] = rated_h2_hr
        h2_results_aggregates["Cluster Rated Power Consumed [kWh]"] = (
            p_consumed_max * self.max_stacks
        )
        h2_results_aggregates["Cluster Rated H2 Production [kg/hr]"] = (
            rated_h2_hr * self.max_stacks
        )
        h2_results_aggregates["gal H20 per kg H2"] = np.sum(
            h20_gal_used_system
        ) / np.sum(self.store_h2_kg_hr_system)
        h2_results_aggregates["Stack Rated Efficiency [kWh/kg]"] = (
            p_consumed_max / rated_h2_hr
        )
        h2_results_aggregates["Cluster Rated H2 Production [kg/yr]"] = (
            rated_h2_hr * len(self.store_input_power_kw) * self.max_stacks
        )
        h2_results_aggregates["Operational Time / Simulation Time (ratio)"] = (
            self.percent_of_sim_operating
        )  # added
        h2_results_aggregates["Fraction of Life used during sim"] = (
            self.frac_of_life_used
        )  # added

        h2_results_aggregates["PEM Capacity Factor (simulation)"] = pem_cf

        h2_results_aggregates["Total H2 Production [kg]"] = np.sum(self.store_h2_kg_hr_system)
        h2_results_aggregates["Total Input Power [kWh]"] = np.sum(
            self.store_input_external_power_kw
        )
        h2_results_aggregates["Total kWh/kg"] = np.sum(
            self.store_input_external_power_kw
        ) / np.sum(self.store_h2_kg_hr_system)
        h2_results_aggregates["Total Uptime [sec]"] = np.sum(
            self.cluster_status * self.dt
        )
        h2_results_aggregates["Total Off-Cycles"] = np.sum(self.off_cycle_cnt)
        h2_results_aggregates["Final Degradation [V]"] = (
            self.cumulative_Vdeg_per_hr_sys[-1]
        )
        h2_results_aggregates["Performance By Year"] = (
            annual_performance  # double check if errors
        )
        return h2_results, h2_results_aggregates

    def external_power_supply(self, input_external_power_kw):
        """
        External power source (grid or REG) which will need to be stepped
        down and converted to DC power for the electrolyzer.

        Please note, for a wind farm as the electrolyzer's power source,
        the model assumes variable power supplied to the stack at fixed
        voltage (fixed voltage, variable power and current)

        TODO: extend model to accept variable voltage, current, and power
        This will replicate direct DC-coupled PV system operating at MPP
        """
        power_converter_efficiency = (
            1.0  # this used to be 0.95 but feel free to change as you'd like
        )
        # power_curtailed_kw = np.where(
        #     input_external_power_kw > self.max_stacks * self.stack_rating_kW,
        #     input_external_power_kw - self.max_stacks * self.stack_rating_kW,
        #     0,
        # )

        if input_external_power_kw > self.max_stacks * self.stack_rating_kW:
            power_curtailed_kw = (
                input_external_power_kw - self.max_stacks * self.stack_rating_kW
            )
        else:
            power_curtailed_kw = 0

        # Where condition is true, return input_power - max_stacks * stack_rating

        # input_power_kw = np.where(
        #     input_external_power_kw > (self.max_stacks * self.stack_rating_kW),
        #     (self.max_stacks * self.stack_rating_kW),
        #     input_external_power_kw,
        # )

        if input_external_power_kw > (self.max_stacks * self.stack_rating_kW):
            input_power_kw = self.max_stacks * self.stack_rating_kW
        else:
            input_power_kw = input_external_power_kw

        self.output_dict["Curtailed Power [kWh]"] = power_curtailed_kw
        return input_power_kw

    def system_design(self, input_power_kw, cluster_size_mw):
        """
        Calculate whether the cluster is on or off based on input power

        TODO: add 0.1 (default turndown ratio) as input
        """
        # cluster_min_power = 0.1*self.max_stacks
        # cluster_min_power = 0.1*cluster_size_mw
        cluster_min_power = self.turndown_ratio * cluster_size_mw

        if input_power_kw > cluster_min_power:
            cluster_status = 1
        else:
            cluster_status = 0

        # cluster_status=np.where(input_power_kw<cluster_min_power,0,1)

        return cluster_status
