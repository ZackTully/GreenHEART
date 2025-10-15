import numpy as np
import scipy.interpolate
import scipy.optimize
import matplotlib.pyplot as plt
from greenheart.simulation.technologies.heat.materials import Quartz

from greenheart.simulation.technologies.dispatch.control_model import ControlModel


class ThermalEnergyStorage:

    def __init__(
        self,
        M_hot_capacity=22500e3,
        M_buffer_capacity=22500e3,
        mdot_max_charge=1082057,
        mdot_max_discharge=1082057,
        T_hot_target=1200,
        T_buffer_target=300,
        initial_SOC=0.5,
        M_total=22500e3,
    ):

        self.particle = Quartz()
        self.C2K = 273.15
        self.kJpkWh = 3600  # kJ per kWh
        self.kWhpkJ = 1 / 3600  # kWh per kJ

        self.lift_power_kWhpkg = (482 * 84.1 / 25248) * self.kWhpkJ
        self.m_charge_max_kgphr = mdot_max_charge
        self.m_discharge_max_kgphr = mdot_max_discharge

        self.eta_electric_heater = 0.98

        self.Q_loss_flag = True
        self.Q_loss_percpday = 0.01
        self.Q_loss_rate_kWhphr = self.Q_loss_percpday / 24  # 1% per day loss of heat energy

        self.T_hot_target = T_hot_target  # [C]
        self.T_buffer_target = T_buffer_target  # [C]
        # self.T_cold_target = 25 # [C]

        self.T_hot = 1200  # [C]
        self.T_buffer = 300  # [C]

        self.M_hot_max = M_hot_capacity
        self.M_hot_min = 0.01 * M_hot_capacity

        self.M_buffer_max = M_buffer_capacity
        self.M_buffer_min = 0.01 * M_buffer_capacity

        self.M_total = M_total  # kg

        self.M_hot = initial_SOC * self.M_total
        self.M_buffer = (1 - initial_SOC) * self.M_total

        self.H_hot_max_kWh = (
            self.particle.H(self.T_hot_target + self.C2K)
            / (self.particle.molar_mass / 1000)
            / 3600
            * self.M_hot_max
        )
        self.H_hot_min_kWh = (
            self.particle.H(self.T_hot_target + self.C2K)
            / (self.particle.molar_mass / 1000)
            / 3600
            * self.M_hot_min
        )
        self.H_buffer_max_kWh = (
            self.particle.H(self.T_buffer_target + self.C2K)
            / (self.particle.molar_mass / 1000)
            / 3600
            * self.M_buffer_max
        )
        self.H_buffer_min_kWh = (
            self.particle.H(self.T_buffer_target + self.C2K)
            / (self.particle.molar_mass / 1000)
            / 3600
            * self.M_buffer_min
        )

        self.H_capacity_kWh = (
            self.delta_H(self.T_buffer_target, self.T_hot_target) * self.M_total
        )

        self.max_charge_kWhphr = (
            self.delta_H(self.T_buffer_target, self.T_hot_target)
            * self.m_charge_max_kgphr
        )
        self.max_discharge_kWhphr = (
            -self.delta_H(self.T_hot_target, self.T_buffer_target)
            * self.m_discharge_max_kgphr
        )

        self.setup_storage()
        self.control_model = self.create_control_model()

        []

    def _SOC(self):
        return (self.tank_H("hot") - self.H_hot_min_kWh) / (self.H_hot_max_kWh - self.H_hot_min_kWh)
        # return self.tank_H("hot") / self.H_capacity_kWh
        # return (
        #     self.delta_H(self.T_buffer, self.T_hot) * self.M_hot
        # ) / self.H_capacity_kWh
        # return (self.tank_H("hot") - self.tank_H("buffer")) / self.H_capacity_kWh

    def tank_H(self, which=None):
        """_summary_

        Args:
            which (str, optional): "hot" or "buffer" which tank to return the enthalpy for

        Returns:
            float : [kWh] Current enthalpy of the tank
        """

        if which == "hot":
            H = (
                self.particle.H(self.T_hot + self.C2K)
                / (self.particle.molar_mass / 1000)
                / 3600
                * self.M_hot
            )
        elif which == "buffer":
            H = (
                self.particle.H(self.T_buffer + self.C2K)
                / (self.particle.molar_mass / 1000)
                / 3600
                * self.M_buffer
            )

        return H

    def delta_H(self, T1, T2):
        """_summary_

        Args:
            T1 (float): Start temperature [C]
            T2 (float): End temperature [C]

        Returns:
            (float): change in enthalpy in [kWh kg^-1]
        """
        return (
            self.kWhpkJ
            * (self.particle.H(T2 + self.C2K) - self.particle.H(T1 + self.C2K))
            / (self.particle.molar_mass / 1000)
        )

    def get_state_measurement(self, step_index):
        return self.tank_H("hot"), self.M_hot


    def step(self, available_power, dispatch, step_index=None):

        if dispatch.ndim > 0:
            if len(dispatch) == 1:
                dispatch = dispatch[0]

            elif len(dispatch) == 2:

                # dispatch = dispatch[0] - dispatch[1]
                dispatch_charge = dispatch[0]
                dispatch_discharge = dispatch[1]
            else:
                assert False, "Bad dispatch command"

        if available_power.ndim > 0:
            available_power = available_power[0]

        P_charge_desired_kWh = np.max([dispatch_charge, 0])
        Q_discharge_desired_kWh = np.max([dispatch_discharge, 0])
        # P_charge_desired_kWh = np.max([dispatch, 0])
        # Q_discharge_desired_kWh = -np.min([dispatch, 0])

        self.P_charge_desired_kWh = P_charge_desired_kWh
        self.Q_discharge_desired_kWh = Q_discharge_desired_kWh

        m_charge, m_discharge, unused_power = self.low_level_controller(
            available_power, P_charge_desired_kWh, Q_discharge_desired_kWh, step_index
        )

        self.m_charge = m_charge
        self.m_discharge = m_discharge
        self.unused_power = unused_power

        Q_out_kWh = self.step_model(m_charge, m_discharge, step_index)

        y_model = Q_out_kWh
        u_passthrough = 0
        u_curtail = unused_power

        return y_model, u_passthrough, u_curtail

    def low_level_controller(
        self, available_power, P_charge_desired_kWh, Q_discharge_desired_kWh, step_index
    ):

        if P_charge_desired_kWh > available_power:
            P_charge_desired_kWh = available_power
            unused_power = 0
        else:
            unused_power = available_power - P_charge_desired_kWh

        # P_charge_desired_kWh = np.min([available_power, P_charge_desired_kWh])

        m_charge = (self.eta_electric_heater * P_charge_desired_kWh) / (
            self.delta_H(self.T_buffer, self.T_hot_target) + self.lift_power_kWhpkg
        )

        m_charge_sat = np.min(
            [m_charge, self.m_charge_max_kgphr, self.M_buffer - self.M_buffer_min]
        )

        additional_unused_power = self.delta_H(self.T_buffer, self.T_hot_target) * max((m_charge - m_charge_sat), 0)

        assert m_charge_sat >= 0
        # Choose m_discharge

        # FIXME dont take the lifting power out of the heat energy out
        m_discharge = Q_discharge_desired_kWh / (
            -self.delta_H(self.T_hot, self.T_buffer_target) + self.lift_power_kWhpkg
        )

        m_discharge_sat = np.min(
            [m_discharge, self.m_discharge_max_kgphr, self.M_hot - self.M_hot_min]
        )

        assert m_discharge_sat >= 0

        self.P_used = P_charge_desired_kWh - additional_unused_power
        return m_charge_sat, m_discharge_sat, unused_power+ additional_unused_power

    def step_model(self, m_charge, m_discharge, step_index):

        delta_M_hot = m_charge - m_discharge
        delta_M_buffer = m_discharge - m_charge

        if self.Q_loss_flag:

            if (self.M_hot / self.M_hot_max) > 0.01:
                Q_loss_hot = self.Q_loss_rate_kWhphr * self.tank_H("hot")
                delta_T_hot = -scipy.optimize.fsolve(
                    lambda delta_T: Q_loss_hot
                    - self.delta_H(self.T_hot - delta_T, self.T_hot) * self.M_hot,
                    x0=1,
                    xtol=1e-2,
                    maxfev=50,
                )[0]
            else:
                delta_T_hot = 0

            if (self.M_buffer / self.M_buffer_max) > 0.01:
                Q_loss_buffer = self.Q_loss_rate_kWhphr * self.tank_H("buffer")
                delta_T_buffer = -scipy.optimize.fsolve(
                    lambda delta_T: Q_loss_buffer
                    - self.delta_H(self.T_buffer - delta_T, self.T_buffer)
                    * self.M_buffer,
                    x0=1,
                    xtol=1e-2,
                    maxfev=50,
                )[0]

            else:
                delta_T_buffer = 0

        else:

            delta_T_hot = 0
            delta_T_buffer = 0

        Q_out_kWh = -self.delta_H(self.T_hot, self.T_buffer_target) * m_discharge

        self.Q_out_kWh = Q_out_kWh[0]

        self.M_hot += delta_M_hot
        self.M_buffer += delta_M_buffer

        # self.T_hot += delta_T_hot
        # self.T_buffer += delta_T_buffer

        self.T_hot = (
            (self.T_hot + delta_T_hot) * self.M_hot + m_charge * self.T_hot_target
        ) / (m_charge + self.M_hot)
        self.T_buffer = (
            (self.T_buffer + delta_T_buffer) * self.M_buffer
            + m_discharge * self.T_buffer_target
        ) / (m_discharge + self.M_buffer)

        self.store_step(step_index)

        return Q_out_kWh

    def setup_storage(self):
        duration = 8760

        self.M_hot_store = np.zeros(duration)
        self.M_buffer_store = np.zeros(duration)

        self.T_hot_store = np.zeros(duration)
        self.T_buffer_store = np.zeros(duration)

        self.H_hot_store = np.zeros(duration)
        self.H_buffer_store = np.zeros(duration)

        self.SOC_store = np.zeros(duration)

        self.Q_out_store = np.zeros(duration)
        self.P_used_store = np.zeros(duration)

        self.P_charge_des_store = np.zeros(duration)
        self.Q_discharge_des_store = np.zeros(duration)

        self.m_charge_store = np.zeros(duration)
        self.m_discharge_store = np.zeros(duration)
        self.unused_power_store = np.zeros(duration)

    def store_step(self, step_index):

        self.M_hot_store[step_index] = self.M_hot
        self.M_buffer_store[step_index] = self.M_buffer

        self.T_hot_store[step_index] = self.T_hot
        self.T_buffer_store[step_index] = self.T_buffer

        self.H_hot_store[step_index] = self.tank_H("hot")
        self.H_buffer_store[step_index] = self.tank_H("buffer")

        self.SOC_store[step_index] = self._SOC()

        self.Q_out_store[step_index] = self.Q_out_kWh
        self.P_used_store[step_index] = self.P_used

        self.P_charge_des_store[step_index] = self.P_charge_desired_kWh
        self.Q_discharge_des_store[step_index] = self.Q_discharge_desired_kWh

        self.m_charge_store[step_index] = self.m_charge
        self.m_discharge_store[step_index] = self.m_discharge
        self.unused_power_store[step_index] = self.unused_power

    def create_control_model(self):
        m = 2
        n = 2
        p = 1
        o = 1

        A = np.array([[1.00665816, -0.00247259], [0.02881832, 0.99000026]])
        B = np.array([[1.1939346, -1.27682378], [3.30606126, -3.83598422]])
        E = np.array([[0], [0]])
        C = np.array([[0, 0], [0, 0]])
        D = np.array([[0, 1], [-1, 0]])
        F = np.array([[0], [1]])

        bounds_dict = {
            "u_lb": np.array([0, 0]),
            "u_ub": np.array([self.max_charge_kWhphr, self.max_discharge_kWhphr]),
            "x_lb": np.array(
                [
                    self.H_hot_min_kWh
                    + 0.01 * (self.H_hot_max_kWh - self.H_hot_min_kWh),
                    self.M_hot_min + 0.01 * (self.M_hot_max - self.M_hot_min),
                ]
            ),
            # "x_lb": np.array([self.H_hot_min_kWh, self.M_hot_min]),
            "x_ub": np.array([self.H_hot_max_kWh, self.M_hot_max]),
            "y_lb": np.array([None, None]),
            "y_ub": np.array([None, None]),
        }
        # m = 2
        # n = 1
        # p = 1
        # o = 1

        # A = np.array([[1 - self.Q_loss_rate_kWhphr]])
        # # A = np.array([[0.975562185090931]])
        # B = np.array([[1, -1]])
        # E = np.array([[0]])
        # C = np.array([[0], [0]])
        # D = np.array([[0, 1], [-1, 0]])
        # F = np.array([[0], [1]])

        # bounds_dict = {
        #     "u_lb": np.array([0, 0]),
        #     "u_ub": np.array([self.max_charge_kWhphr, self.max_discharge_kWhphr]),
        #     "x_lb": np.array([self.H_hot_min_kWh]),
        #     "x_ub": np.array([self.H_hot_max_kWh]),
        #     "y_lb": np.array([None, None]),
        #     "y_ub": np.array([None, None]),
        # }
        # bounds_dict = {
        #     "u_lb": np.array([0, 0]),
        #     "u_ub": np.array([self.max_charge_kWhphr, self.max_discharge_kWhphr]),
        #     "x_lb": np.array([0]),
        #     "x_ub": np.array([self.H_capacity_kWh]),
        #     "y_lb": np.array([None, None]),
        #     "y_ub": np.array([None, None]),
        # }

        control_model = ControlModel(
            A, B, C, D, E, F, bounds=bounds_dict, discrete=True
        )

        control_model.set_disturbance_domain([1, 0, 0])
        control_model.set_output_domain([0, 1, 0])
        control_model.set_disturbance_reshape([1, 0, 0])

        control_model.constraints(y_position=[1], constraint_type=["greater"])
        # control_model.constraints(y_position=[1], constraint_type=["greater"])

        return control_model


class EnduringGeneral:
    operation_and_maintenance_usdpkwhth = 0.00171  # usd / kwhth


class EnduringParticleHeater:
    # Table 15 on page 32

    single_unit_capacity_MW = 316.0
    single_unit_equipment_cost_USD = 1316688.0
    single_unit_capital_cost_USD = 2304205.0
    unitized_capital_cost_USDpkW = 7.3

    gross_capital_cost_3units_USD = 6912614

    heating_efficiency = 0.98  # from Ma et al. 2021


class EnduringLockHopper:
    # Table 14 on page 31

    single_lock_hopper_equipment_cost_USD = 221607
    top_gate_valve_equipment_cost_USD = 15652
    bottom_gate_valve_equipment_cost_USD = 10743

    single_unit_capital_cost_USD = 359602
    unitized_capital_cost_USDpkW = 21.31

    gross_capital_cost_24units_USD = 8630459

    particle_load_per_hopper_kg = 293976
    lock_hopper_volume_m3 = 313

    charge_time_s = 201
    discharge_time_s = 462


class EnduringPressurizedFluidizedBedHeatExchanger:

    # Table 13

    PFBPV_equipment_cost_USD = 2071334.0
    PFBHX_equipment_cost_USD = 4574561.0
    particle_separation_cyclone_equipment_cost_USD = 57124.0

    single_unit_capital_cost_USD = 9719377.0  # 3 units?
    total_PFBHX_system_capital_cost_USD = 29158132.0

    unitized_capital_cost_USDpkW = 72.0  # dollars per kW electric


class EnduringPipeline:
    # from Table 12 page 29

    pipe_length_m = 12
    total_pressure_drop_kPa = 4.04

    single_unit_capital_cost_usd = 675932
    gross_capital_cost_3units_USD = 2027797


class EnduringParticleHoist:
    # from Table 10 page 26

    lift_time_s = 84.1
    particle_load_per_lift_kg = 25248

    skip_hoist_equipment_cost_USD = 613913.0
    single_unit_capital_cost_USD = 1074348.0
    gross_capital_cost_3units_USD = 3223044.0
    unitized_capital_cost_USDpkWh = 0.042  # USD / kWh_th (kWh thermal?)


class EnduringParticleSilo:
    # Table 9 page 25
    single_unit_capacity_kg = 22500 * 1e3
    silo_height_m = 65.8
    silo_diameter_m = 20
    single_unit_capacity_GWhth = 6.4  # should this just be 6.4 ?

    single_unit_capital_cost_USD = 12503325.0
    single_unit_construction_cost_USD = 11731455.0

    system_foundation_cost_USD = 3857262.0  # for 12 silos
    system_insulation_cost_USD = 7874193.0  # for 12 silos

    silo_foundation_cost_USD = 3857262.0
    insulation_cost_USD = 7874193.0
    single_unit_storage_media_cost_USD = 771870.0

    gross_capital_cost_12units_USD = 150039896.0
    unitized_capital_cost_USDpkWhth = 1.96


# class EnduringCosts:


class HighTempTES:
    def __init__(
        self,
        ChargeRate,
        DischargeRate,
        StorageHours,
        ChargeEfficiency=0.98,
        DischargeEfficiency=0.8,
        dt=3600,
    ):
        """
        Inputs:
            ChargeRate [MWe]: 315
            DischargeRate [MWe]: 135
            StorageHours
            ChargeEfficiency
            DischargeEfficiency
            StorageEfficiency
        """
        self.particle_Cp = 1.155  # kJ/kg-K
        self.particle_density = 2650  # kg/m^3

        self.air_Cp = 1  # kJ/kg-K
        self.air_mol_weight = 28.7  # g/mol

        self.ChargeRate = ChargeRate * 1e3  # kWe
        self.DischargeRate = DischargeRate * 1e3  # kWth
        self.StorageHours = StorageHours  # hours
        # self.StorageCapacity = StorageHours*ChargeRate

        self.ChargeEfficiency = ChargeEfficiency / 100  # [%]
        self.DischargeEfficiency = DischargeEfficiency / 100  # [%]
        # self.StorageEfficiency = StorageEfficiency/100 #[%]

        # Oversize to account for output losses
        self.StorageCapacity = (
            DischargeRate / self.DischargeEfficiency
        ) * StorageHours  # kWh_th
        self.dt = dt

        pass

    def run(
        self,
        electric_energy: np.ndarray,
        thermal_temperature_demand: float,
        thermal_energy_demand: float,
        SOC_start=0,
        particle_T0=300,
    ):
        """
        electric_energy: kiloWatt-hours electric
        heat_demand_temp: Celsius
        thermal_energy_demand: kiloWatt-hours thermal
        """
        t_sim = len(electric_energy)

        T_dmd = np.ones(t_sim) * thermal_temperature_demand
        T_delta = thermal_temperature_demand - particle_T0
        Eth_dmd = np.ones(t_sim) * thermal_energy_demand

        # curtailed_electric_energy = np.where(electric_energy>self.ChargeRate,electric_energy-self.ChargeRate,0) #kWh-e
        usable_electric_energy = np.where(
            electric_energy > self.ChargeRate, self.ChargeRate, electric_energy
        )  # kWh-e
        usable_thermal_energy = self.ChargeEfficiency * usable_electric_energy  # kWh-th
        required_thermal_energy = Eth_dmd / (self.DischargeEfficiency)  # kWh-th

        tes_discharged, excess_input_energy, tes_SOC = self.simple_dispatch(
            usable_thermal_energy, required_thermal_energy
        )

        tes_output = tes_discharged * self.DischargeEfficiency
        tes_input = usable_thermal_energy - excess_input_energy
        return tes_input, tes_output, tes_SOC

    def simple_dispatch(self, curtailed_energy, energy_shortfall):
        tsim = len(curtailed_energy)
        tes_SOC = np.zeros(tsim)
        tes_discharged = np.zeros(tsim)
        excess_energy = np.zeros(tsim)
        for i in range(tsim):
            # should you charge
            if curtailed_energy[i] > 0:
                if i == 0:
                    tes_SOC[i] = np.min([curtailed_energy[i], self.ChargeRate])
                    amount_charged = tes_SOC[i]
                    excess_energy[i] = curtailed_energy[i] - amount_charged
                else:
                    if tes_SOC[i - 1] < self.StorageCapacity:
                        add_gen = np.min([curtailed_energy[i], self.ChargeRate])
                        tes_SOC[i] = np.min(
                            [tes_SOC[i - 1] + add_gen, self.StorageCapacity]
                        )
                        amount_charged = tes_SOC[i] - tes_SOC[i - 1]
                        excess_energy[i] = curtailed_energy[i] - amount_charged
                    else:
                        tes_SOC[i] = tes_SOC[i - 1]
                        excess_energy[i] = curtailed_energy[i]

            # should you discharge
            else:
                if i > 0:
                    if tes_SOC[i - 1] > 0:

                        tes_discharged[i] = np.min(
                            [energy_shortfall[i], tes_SOC[i - 1], self.DischargeRate]
                        )
                        tes_SOC[i] = tes_SOC[i - 1] - tes_discharged[i]

        return tes_discharged, excess_energy, tes_SOC

    def convert_electric_to_thermal(self, electric_energy):
        return self.ChargeEfficiency * electric_energy

    def charge_storage_silo(self, electric_energy):
        usable_electric_energy = np.where(
            electric_energy > self.ChargeRate, self.ChargeRate, electric_energy
        )  # kWh-e
        usable_thermal_energy = self.ChargeEfficiency * usable_electric_energy  # kWh-th


if __name__ == "__main__":

    # test-like stuff - Reproduce the plant described in the tech report and check if the cost is close

    # config = {
    #     "ChargeRate": 315,
    #     "DischargeRate": 135,
    #     "StorageHours": 100,
    #     "ChargeEfficiency": 98,
    #     "DischargeEfficiency": 52,
    #     # "StorageEfficiency":97,
    #     "dt": 3600,
    # }
    # # GWhth = MWe*(eta)
    # # 52 percent efficient
    # #
    # tes = HighTempTES(**config)

    # # TES = ThermalEnergyStorage()

    # quart = Quartz()

    # def func(T, H):
    #     return quart.H(T) - H

    # H_state = 10.7

    # T_state = scipy.optimize.fsolve(func, x0=500, args=(H_state), xtol=1e-3)

    # tes = ThermalEnergyStorage()

    # # print(tes.T_hot)
    # # tes.step_tank(50, 0, 100, 0)
    # # print(tes.T_hot)

    # H_1 = tes.H_hot

    # Q_out = tes.step(available_power=10639227.25826057, dispatch=10639227.25826057)

    # H_2 = tes.H_hot

    # print(H_2 - H_1)

    # time = np.arange(0, 1000, 1)

    # # Try for 316 MW input
    # power_multiplier = 316e3
    # P_avail = power_multiplier * np.sin(time / 20)
    # P_avail = np.where(P_avail >= 0, P_avail, 0)
    # P_dis = power_multiplier * np.sin((time + 10) / 15)

    # for k in range(len(time)):
    #     Q_out = tes.step(P_avail[k], P_dis[k], step_index=k)

    # fig, ax = plt.subplots(5, 1, sharex="col", layout="constrained")

    # ax[0].plot(time, P_avail, label="P avail")
    # ax[0].plot(time, P_dis, label="Dispatch")

    # ax[1].plot(time, tes.M_cold_store[0 : len(time)], label="M cold")
    # ax[1].plot(time, tes.M_hot_store[0 : len(time)], label="M hot")
    # ax1t = ax[1].twinx()
    # ax1t.plot(time, tes.m_c2h_store[0 : len(time)], label="m c2h")
    # ax1t.plot(time, tes.m_h2c_store[0 : len(time)], label="m h2c")
    # ax1t.legend()

    # ax[2].plot(time, tes.H_hot_store[0 : len(time)], label="H hot")
    # # ax[2].plot(time, tes.H_cold_store[0 : len(time)], label="H cold")

    # ax2t = ax[2].twinx()
    # ax2t.plot(time, tes.SOC_store[0:len(time)], label="SOC")
    # ax2t.legend()

    # ax[3].plot(time, tes.T_hot_store[0 : len(time)], label="T hot")
    # # ax[3].plot(time, tes.T_cold_store[0:len(time)], label="T cold")

    # ax[4].plot(time, tes.Q_out_store[0 : len(time)], label="Q out")

    # for a in ax:
    #     a.legend()

    # m = 22500e3  # kg
    # GWhth_per_silo = (
    #     ((m / tes.mm) * (tes.sand.H(1200 + tes.C2K) - tes.sand.H(300 + tes.C2K)))
    #     / 1e6
    #     / 3600
    # )

    # Updated model from March 8 2025
    TES = ThermalEnergyStorage()

    # Do timeseries comparing the control model to the simulation model

    t_start = 0
    t_stop = 2000  # hours
    dt = 1  # hour

    time = np.arange(t_start, t_stop, dt)

    P_available = 125e3 * np.ones(len(time))  # 400 MW available
    P_charge_desired = 100e3 * np.sin(
        time / (100)
    )  # charge with 300 MW. Close to maximium charging rate


    P_charge_desired = np.concatenate([
        100e3 * np.ones(100),
        -100e3 * np.ones(100), 
        np.zeros(200),
        100e3 * np.ones(100),
        np.zeros(100), 
        -100e3 * np.ones(100), 
        np.zeros(1300)
    ])




    Q_out_store = np.zeros(len(time))
    P_pt_store = np.zeros(len(time))
    P_curtail_store = np.zeros(len(time))

    A = TES.control_model.A
    B = TES.control_model.B
    C = TES.control_model.C
    D = TES.control_model.D
    E = TES.control_model.E
    F = TES.control_model.F

    for i in range(len(time)):
        Q_out, P_pt, P_curtail = TES.step(
            available_power=P_available[i], dispatch=P_charge_desired[i], step_index=i
        )

        Q_out_store[i] = Q_out
        P_pt_store[i] = P_pt
        P_curtail_store[i] = P_curtail

    fig, ax = plt.subplots(2, 3, sharex="all", sharey="col", layout="constrained")

    ax[0, 0].plot(time, P_available, label="Available")
    ax[0, 0].plot(time, P_charge_desired, label="Dispatch")

    ax[0, 0].plot(time, Q_out_store, label="Q out")
    ax[0, 0].plot(time, P_pt_store, label="passthrough")
    ax[0, 0].plot(time, P_curtail_store, label="curtail")
    ax[0, 2].plot(time, TES.SOC_store[0:t_stop])

    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            ax[i, j].legend()

    fig, ax = plt.subplots(1, 4, sharex="all", layout="constrained")

    ax[0].plot(time, TES.M_hot_store[0:t_stop], label="M hot")
    ax[0].plot(time, TES.M_buffer_store[0:t_stop], label="M buffer")

    ax[1].plot(time, TES.T_hot_store[0:t_stop], label="T hot")
    ax[1].plot(time, TES.T_buffer_store[0:t_stop], label="T buffer")

    []
