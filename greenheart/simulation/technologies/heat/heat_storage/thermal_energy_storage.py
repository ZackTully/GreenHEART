import numpy as np
import scipy.interpolate
import scipy.optimize
import matplotlib.pyplot as plt


class ShomateEquation:
    T: np.ndarray

    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    d: np.ndarray
    e: np.ndarray
    f: np.ndarray
    g: np.ndarray
    h: np.ndarray

    a_interp: scipy.interpolate.interp1d
    b_interp: scipy.interpolate.interp1d
    c_interp: scipy.interpolate.interp1d
    d_interp: scipy.interpolate.interp1d
    e_interp: scipy.interpolate.interp1d
    f_interp: scipy.interpolate.interp1d
    G_interp: scipy.interpolate.interp1d
    h_interp: scipy.interpolate.interp1d

    def __init__(self):
        self.a_interp = scipy.interpolate.interp1d(self.T, self.a, kind="previous")
        self.b_interp = scipy.interpolate.interp1d(self.T, self.b, kind="previous")
        self.c_interp = scipy.interpolate.interp1d(self.T, self.c, kind="previous")
        self.d_interp = scipy.interpolate.interp1d(self.T, self.d, kind="previous")
        self.e_interp = scipy.interpolate.interp1d(self.T, self.e, kind="previous")
        self.f_interp = scipy.interpolate.interp1d(self.T, self.f, kind="previous")
        self.g_interp = scipy.interpolate.interp1d(self.T, self.g, kind="previous")
        self.h_interp = scipy.interpolate.interp1d(self.T, self.h, kind="previous")

    def Cp(self, temperature):
        # return heat capacity [j mol^-1 K^-1]

        t = temperature / 1000

        Cp = (
            self.a_interp(temperature)
            + self.b_interp(temperature) * t
            + self.c_interp(temperature) * t**2
            + self.d_interp(temperature) * t**3
            + self.e_interp(temperature) / (t**2)
        )

        return Cp

    def H(self, temperature):
        # Return standard enthalpy [kJ mol^-1]
        # Return H^circle - H^circle_298.15

        t = temperature / 1000

        H = (
            self.a_interp(temperature) * t
            + self.b_interp(temperature) * t**2 / 2
            + self.c_interp(temperature) * t**3 / 3
            + self.d_interp(temperature) * t**4 / 4
            - self.e_interp(temperature) / t
            + self.f_interp(temperature)
            - self.h_interp(temperature)
        )

        return H

    def S(self, temperature):

        # Return standard entropy [J mol^-1 K^-1]

        t = temperature / 1000

        S = (
            self.a_interp(temperature) * np.log(t)
            + self.b_interp(temperature) * t
            + self.c_interp(temperature) * t**2 / 2
            + self.d_interp(temperature) * t**3 / 3
            - self.e_interp(temperature) / (2 * t**2)
            + self.g_interp(temperature)
        )

        return S


class Quartz(ShomateEquation):
    # SiO2

    # https://webbook.nist.gov/cgi/cbook.cgi?ID=C14808607&Type=JANAFS&Table=on

    T_max = 1996

    # Temperature (K)	298. to 847.	847. to 1996.
    T = np.array([298, 847, 1996])
    a = np.array([-6.076591, 58.75340, 58.75340])
    b = np.array([251.6755, 10.27925, 10.27925])
    c = np.array([-324.7964, -0.131384, -0.131384])
    d = np.array([168.5604, 0.025210, 0.025210])
    e = np.array([0.002548, 0.025601, 0.025601])
    f = np.array([-917.6893, -929.3292, -929.3292])
    g = np.array([-27.96962, 105.8092, 105.8092])
    h = np.array([-910.8568, -910.8568, -910.8568])

    molar_mass = 60.0843  # [g mol^-1]


class ThermalEnergyStorage:

    # TODO check units on inputs for consistency kWH
    # TODO also check for kelvin celsisus consistency
    def __init__(self):
        self.sand = Quartz()

        self.mm = self.sand.molar_mass * 1e-3  # [kg mol^-1] molar mass

        self.C2K = 273.15  # convert celsius to kelvin

        self.eta_P_in = 0.98  # from Table 3
        self.eta_Q_out = 0.98

        self.M_hot_capacity = 22500e3  # [kg] assuming a single silo
        self.M_cold_capacity = 22500e3  # [kg] assume a single silo

        self.mdot_max = 25248 / 84 * 3600  # [kg hr^-1] of sand
        self.T_max = 1700  # C

        self.P_in_max = 316  # [MW]

        self.T_hot_target = 1200  # C
        self.T_cold_target = 300  # C

        # states
        self.T_hot = 1200  # C
        self.M_hot = 11250000.0  # kg
        # self.H_hot = self.M_hot * self.sand.H(self.T_hot) * self.sand.molar_mass * 1e-3
        self.H_hot = (self.M_hot * 1000 / self.sand.molar_mass) * self.sand.H(
            self.T_hot + self.C2K
        )

        self.T_cold = 300  # C
        self.M_cold = 11250000.0  # kg
        # self.H_cold = self.M_cold * self.sand.H(self.T_cold) * self.sand.molar_mass * 1e-3
        self.H_cold = (self.M_cold * 1000 / self.sand.molar_mass) * self.sand.H(
            self.T_cold + self.C2K
        )

        H_loss_pday = 0.01  # 1% loss per day Table 3
        self.H_loss_phr = H_loss_pday / 24
        self.H_loss_phr = 0.03 / (5 * 24)

        # self.step_tank(0, 0, 0, 0)

        # energy capacity in kJ
        self.energy_capacity = (self.M_hot_capacity / self.mm) * (
            self.sand.H(self.T_hot_target + self.C2K)
            - self.sand.H(self.T_cold_target + self.C2K)
        )


        self.SOC = (self.H_hot - self.H_cold) / self.energy_capacity

        self.setup_storage()

    def setup_storage(self):
        self.M_hot_store = np.zeros(8760)
        self.T_hot_store = np.zeros(8760)
        self.H_hot_store = np.zeros(8760)
        self.M_cold_store = np.zeros(8760)
        self.T_cold_store = np.zeros(8760)
        self.H_cold_store = np.zeros(8760)

        self.SOC_store = np.zeros(8760)

        self.Q_out_store = np.zeros(8760)
        self.P_used_store = np.zeros(8760)

        self.m_c2h_store = np.zeros(8760)
        self.m_h2c_store = np.zeros(8760)
        self.T_c2h_store = np.zeros(8760)
        self.T_h2c_store = np.zeros(8760)

    def store_step(self, Q_out, P_used, m_c2h, T_c2h, m_h2c, T_h2c, step_index):
        self.M_hot_store[step_index] = self.M_hot
        self.T_hot_store[step_index] = self.T_hot
        self.H_hot_store[step_index] = self.H_hot
        self.M_cold_store[step_index] = self.M_cold
        self.T_cold_store[step_index] = self.T_cold
        self.H_cold_store[step_index] = self.H_cold
        
        self.SOC_store[step_index] = self.SOC
        
        self.Q_out_store[step_index] = Q_out

        self.P_used_store[step_index] = P_used

        self.m_c2h_store[step_index] = m_c2h
        self.m_h2c_store[step_index] = m_h2c
        self.T_c2h_store[step_index] = T_c2h
        self.T_h2c_store[step_index] = T_h2c

    def step(self, available_power, dispatch, step_index=None):

        # translate into joules
        available_power_kJ = available_power * 3600
        dispatch_kJ = dispatch * 3600

        u_LLC = self.low_level_controller(available_power, dispatch, step_index)
        Q_out, unused_power = self.step_tank(
            available_power, dispatch, *u_LLC, step_index=step_index
        )

        # translate into kWh
        Q_out_kWh = Q_out / 3600

        return Q_out

    def low_level_controller(self, available_power, dispatch, step_index=None):
        # Takes power charging request or heat dischargin request
        # Checks if it can fulfill
        # Then calculates mdot_h2c, P_in, mdot_c2h, Q_out

        P_in_dispatch = np.max([dispatch, 0])
        Q_out_dispatch = -np.min([dispatch, 0])

        # Calculate the theoretical system inputs
        T_c2h = self.T_hot_target + (self.T_hot_target - self.T_hot)
        # T_c2h = self.T_hot_target
        m_c2h = (P_in_dispatch * (self.eta_P_in) * self.mm) / (
            self.sand.H(T_c2h + self.C2K)
            - self.sand.H(self.T_cold + self.C2K)
        )

        T_h2c = self.T_cold_target + (self.T_cold_target - self.T_cold)
        # T_h2c = self.T_cold_target
        m_h2c = (-Q_out_dispatch * self.eta_Q_out * self.mm) / (
            self.sand.H(T_h2c + self.C2K)
            - self.sand.H(self.T_hot + self.C2K)
        )

        # Check with constraints and refues to do it if the cosntraints are violated

        c2h1 = self.mdot_max
        c2h2 = self.M_hot_capacity - self.M_hot
        # c2h3 = 0 # Consider doing a temperature constraint

        c2h_constraint = np.min([c2h1, c2h2])

        assert m_c2h <= c2h_constraint

        m_c2h = np.min([c2h_constraint, m_c2h])

        h2c1 = self.mdot_max
        h2c2 = self.M_cold_capacity - self.M_cold

        h2c_constraint = np.min([h2c1, h2c2])

        assert m_h2c <= h2c_constraint

        m_h2c = np.min([h2c_constraint, m_h2c])

        return (m_c2h, m_h2c, P_in_dispatch, Q_out_dispatch, T_c2h, T_h2c)

    def step_tank(
        self,
        available_power,
        dispatch,
        m_c2h,
        m_h2c,
        P_in_LLC,
        Q_out_LLC,
        T_c2h,
        T_h2c,
        heat_loss=True,
        step_index=None,
    ):

        heat_loss = False

        M_hot_next = self.M_hot + m_c2h - m_h2c
        M_cold_next = self.M_cold - m_c2h + m_h2c

        if heat_loss:
            H_hot_loss = self.H_loss_phr * self.H_hot
            H_cold_loss = self.H_loss_phr * self.H_cold
        else:
            H_hot_loss = 0
            H_cold_loss = 0

        H_hot_next = (
            self.H_hot
            + m_c2h / self.mm * self.sand.H(T_c2h + self.C2K)
            - m_h2c / self.mm * self.sand.H(self.T_hot + self.C2K)
            - H_hot_loss
        )
        H_cold_next = (
            self.H_cold
            + m_h2c / self.mm * self.sand.H(T_h2c + self.C2K)
            - m_c2h / self.mm * self.sand.H(self.T_cold + self.C2K)
            - H_cold_loss
        )
        # H_hot_next = self.H_hot + m_c2h / self.mm * self.sand.H(T_c2h) - H_hot_loss
        # H_cold_next = self.H_cold + m_h2c / self.mm * self.sand.H(T_h2c) - H_cold_loss

        def func(T, H, M):
            return self.sand.H(T + self.C2K) - H / (M / (self.sand.molar_mass * 1e-3))

        T_hot_next = scipy.optimize.fsolve(
            func, x0=self.T_hot_target, args=(self.H_hot, self.M_hot), xtol=1e-3
        )[0]
        T_cold_next = scipy.optimize.fsolve(
            func, x0=self.T_cold_target, args=(self.H_cold, self.M_cold), xtol=1e-3
        )[0]

        P_used = (
            (1 / self.eta_P_in)
            * m_c2h
            / self.mm
            * (self.sand.H(T_c2h + self.C2K) - self.sand.H(self.T_cold + self.C2K))
        )
        Q_out = (
            self.eta_Q_out
            * m_h2c
            / self.mm
            * (self.sand.H(self.T_hot + self.C2K) - self.sand.H(T_h2c + self.C2K))
        )

        unused_power = available_power - P_used

        self.store_step(Q_out, P_used, m_c2h, T_c2h, m_h2c, T_h2c, step_index)

        # Update states
        self.M_hot = M_hot_next
        self.T_hot = T_hot_next
        self.H_hot = H_hot_next
        self.M_cold = M_cold_next
        self.T_cold = T_cold_next
        self.H_cold = H_cold_next
        self.SOC = (H_hot_next - H_cold_next) / self.energy_capacity

        assert self.M_hot <= self.M_hot_capacity
        assert self.M_cold <= self.M_cold_capacity
        assert self.T_hot <= self.T_max

        return Q_out, unused_power


class EnduringParticleHeater:
    # Table 15 on page 32

    single_unit_capacity_MW = 316.0
    single_unit_equipment_cost_USD = 1316688.0
    single_unit_capital_cost_USD = 2304205.0
    unitized_capital_cost_USDpkW = 7.3

    gross_capital_cost_3units_USD = 6912614


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


class EnduringParticleFluidizedBed:

    # Table 13

    PFBPV_equipment_cost_USD = 2071334.0
    PFBHX_equipment_cost_USD = 4574561.0
    particle_separation_cyclone_equipment_cost_USD = 57124.0

    single_PFBHX_capital_cost_USD = 9719377.0  # 3 units?
    total_PFBHX_system_capital_cost_USD = 29158132.0

    unitized_capital_cost_USDpkW = 72.0  # dollars per kW electric


class EnduringPipeline:
    pass


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


class ThermalEnergyStorage_old:
    def __init__(self):
        self.particle_cp = 1.155  # kJ kg^-1 K^-1

        self.dt = 3600  # [s]
        self.T_tank = 1200  # [C] temperature of the particles in the storage tank
        self.T_ambient = 15  # [C]

        self.m_tank = 1000000  # [kg] mass of particles in storage tank

        # Low-level controller parameters
        self.T_max = 1710  # [C] maximum tank temperature, particle melting temperature
        self.T_min = 1000  # [C]

        # How to define max charge rate? heat flow rate, delta temp., particle mass flow
        self.Q_in_max = 5000  # [kW]
        self.Q_out_max = 5000  # [kW]

        # state storage
        self.T_tank_storage = np.zeros(8760)

    def step(self, Qdot_in_available, dispatch, step_index):

        Qdot_desired = dispatch

        # if dispatch > 0:
        #     Qdot_desired = - dispatch
        # elif Qdot_in_available >= 0:
        #     Qdot_desired = Qdot_in_available

        Qdot_controller = self.control(Qdot_in_available, Qdot_desired)
        Qdot_controller = float(Qdot_controller)
        self.step_particle_tank(Qdot_controller)

        self.record(step_index)

        if Qdot_controller <= 0:
            Qdot_output = -Qdot_controller
        else:
            Qdot_output = 0.0

        return Qdot_output

    def record(self, step_index):
        self.T_tank_storage[step_index] = self.T_tank

    def control(self, Qdot_in_available, Qdot_desired):
        # Qdot_in_avaible: [kW s^-1] should be > 0
        # Qdot_desired: [kW s^-1] can be positive or negative

        assert Qdot_in_available >= 0, "Cannot have negative available heat"

        # state constraint
        upper1 = (self.T_max - self.T_tank) * self.m_tank * self.particle_cp / self.dt
        lower1 = (self.T_min - self.T_tank) * self.m_tank * self.particle_cp / self.dt

        # rate constraint
        upper2 = self.Q_in_max
        lower2 = -self.Q_out_max

        # available constraint
        upper3 = Qdot_in_available

        upper = np.min([upper1, upper2, upper3])
        lower = np.max([lower1, lower2])

        if Qdot_desired >= upper:
            Qdot_out = upper
        elif Qdot_desired <= lower:
            Qdot_out = lower
        else:
            Qdot_out = Qdot_desired

        return Qdot_out

    def heat_exchanger(self, Qdot_io, mdot_particle, T_in_particle):
        # Qdot_io: [kW/s] heatflow positive or negative
        T_out_particle = T_in_particle + Qdot_io / (mdot_particle * self.particle_cp)

        assert T_out_particle > -273, "output needs to be hotter than absolute zero"
        return mdot_particle, T_out_particle

    def step_particle_tank(self, Qdot_io, mdot_particle=100):

        # TODO include heat loss to ambient

        mdot_out = mdot_particle
        T_out = self.T_tank
        mdot_in, T_in = self.heat_exchanger(Qdot_io, mdot_out, T_out)
        T_tank_delta_dot = (mdot_in / self.m_tank) * (T_in - self.T_tank)
        self.T_tank += T_tank_delta_dot * self.dt


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
    config = {
        "ChargeRate": 315,
        "DischargeRate": 135,
        "StorageHours": 100,
        "ChargeEfficiency": 98,
        "DischargeEfficiency": 52,
        # "StorageEfficiency":97,
        "dt": 3600,
    }
    # GWhth = MWe*(eta)
    # 52 percent efficient
    #
    tes = HighTempTES(**config)

    # TES = ThermalEnergyStorage()

    quart = Quartz()

    def func(T, H):
        return quart.H(T) - H

    H_state = 10.7

    T_state = scipy.optimize.fsolve(func, x0=500, args=(H_state), xtol=1e-3)

    tes = ThermalEnergyStorage()

    # print(tes.T_hot)
    # tes.step_tank(50, 0, 100, 0)
    # print(tes.T_hot)

    H_1 = tes.H_hot

    Q_out = tes.step(available_power=10639227.25826057, dispatch=10639227.25826057)

    H_2 = tes.H_hot

    print(H_2 - H_1)

    time = np.arange(0, 1000, 1)

    # Try for 316 MW input
    power_multiplier = 316e3
    P_avail = power_multiplier * np.sin(time / 20)
    P_avail = np.where(P_avail >= 0, P_avail, 0)
    P_dis = power_multiplier * np.sin((time + 10) / 15)

    for k in range(len(time)):
        Q_out = tes.step(P_avail[k], P_dis[k], step_index=k)

    fig, ax = plt.subplots(5, 1, sharex="col", layout="constrained")

    ax[0].plot(time, P_avail, label="P avail")
    ax[0].plot(time, P_dis, label="Dispatch")

    ax[1].plot(time, tes.M_cold_store[0 : len(time)], label="M cold")
    ax[1].plot(time, tes.M_hot_store[0 : len(time)], label="M hot")
    ax1t = ax[1].twinx()
    ax1t.plot(time, tes.m_c2h_store[0 : len(time)], label="m c2h")
    ax1t.plot(time, tes.m_h2c_store[0 : len(time)], label="m h2c")
    ax1t.legend()

    ax[2].plot(time, tes.H_hot_store[0 : len(time)], label="H hot")
    # ax[2].plot(time, tes.H_cold_store[0 : len(time)], label="H cold")

    ax2t = ax[2].twinx()
    ax2t.plot(time, tes.SOC_store[0:len(time)], label="SOC")
    ax2t.legend()

    ax[3].plot(time, tes.T_hot_store[0 : len(time)], label="T hot")
    # ax[3].plot(time, tes.T_cold_store[0:len(time)], label="T cold")

    ax[4].plot(time, tes.Q_out_store[0 : len(time)], label="Q out")

    for a in ax:
        a.legend()

    m = 22500e3  # kg
    GWhth_per_silo = (
        ((m / tes.mm) * (tes.sand.H(1200 + tes.C2K) - tes.sand.H(300 + tes.C2K)))
        / 1e6
        / 3600
    )

    []
