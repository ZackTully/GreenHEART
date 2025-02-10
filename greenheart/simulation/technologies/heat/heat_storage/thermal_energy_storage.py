import numpy as np


class ThermalEnergyStorage:
    def __init__(self):
        self.particle_cp = 1.155 # kJ kg^-1 K^-1

        self.dt = 3600 # [s]
        self.T_tank = 1200 # [C] temperature of the particles in the storage tank
        self.T_ambient = 15 # [C]

        self.m_tank = 1000000 # [kg] mass of particles in storage tank

        # Low-level controller parameters
        self.T_max = 1710 # [C] maximum tank temperature, particle melting temperature
        self.T_min = 1000 # [C]


        # How to define max charge rate? heat flow rate, delta temp., particle mass flow
        self.Q_in_max = 5000 # [kW]
        self.Q_out_max = 5000 # [kW]

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
        lower1 = (self.T_min - self.T_tank) * self.m_tank * self.particle_cp /self.dt

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
    def __init__(self,ChargeRate,DischargeRate,StorageHours,ChargeEfficiency=0.98,DischargeEfficiency=0.8,dt=3600):
        """
        Inputs:
            ChargeRate [MWe]: 315
            DischargeRate [MWe]: 135
            StorageHours
            ChargeEfficiency
            DischargeEfficiency
            StorageEfficiency
        """
        self.particle_Cp = 1.155 #kJ/kg-K
        self.particle_density = 2650 #kg/m^3

        self.air_Cp = 1 #kJ/kg-K
        self.air_mol_weight = 28.7 #g/mol

        self.ChargeRate = ChargeRate*1e3 #kWe
        self.DischargeRate = DischargeRate*1e3 #kWth
        self.StorageHours = StorageHours #hours
        # self.StorageCapacity = StorageHours*ChargeRate

        self.ChargeEfficiency = ChargeEfficiency/100 #[%]
        self.DischargeEfficiency = DischargeEfficiency/100 #[%]
        # self.StorageEfficiency = StorageEfficiency/100 #[%]

        #Oversize to account for output losses
        self.StorageCapacity = (DischargeRate/self.DischargeEfficiency)*StorageHours #kWh_th
        self.dt = dt

        pass

    def run(self,electric_energy:np.ndarray,thermal_temperature_demand:float,thermal_energy_demand:float,SOC_start = 0, particle_T0 = 300):
        """
        electric_energy: kiloWatt-hours electric
        heat_demand_temp: Celsius
        thermal_energy_demand: kiloWatt-hours thermal
        """
        t_sim = len(electric_energy)
        
        T_dmd = np.ones(t_sim)*thermal_temperature_demand
        T_delta = thermal_temperature_demand-particle_T0
        Eth_dmd = np.ones(t_sim)*thermal_energy_demand

        # curtailed_electric_energy = np.where(electric_energy>self.ChargeRate,electric_energy-self.ChargeRate,0) #kWh-e
        usable_electric_energy = np.where(electric_energy>self.ChargeRate,self.ChargeRate,electric_energy) #kWh-e
        usable_thermal_energy = self.ChargeEfficiency*usable_electric_energy #kWh-th
        required_thermal_energy = Eth_dmd/(self.DischargeEfficiency) #kWh-th

        tes_discharged,excess_input_energy,tes_SOC = self.simple_dispatch(usable_thermal_energy,required_thermal_energy)    

        tes_output = tes_discharged*self.DischargeEfficiency
        tes_input = usable_thermal_energy-excess_input_energy
        return tes_input,tes_output,tes_SOC        

    def simple_dispatch(self,curtailed_energy,energy_shortfall):
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
                    if tes_SOC[i-1] < self.StorageCapacity:
                        add_gen = np.min([curtailed_energy[i], self.ChargeRate])
                        tes_SOC[i] = np.min([tes_SOC[i-1] + add_gen, self.StorageCapacity])
                        amount_charged = tes_SOC[i] - tes_SOC[i-1]
                        excess_energy[i] = curtailed_energy[i] - amount_charged
                    else:
                        tes_SOC[i] = tes_SOC[i - 1]
                        excess_energy[i] = curtailed_energy[i]

            # should you discharge
            else:
                if i > 0:
                    if tes_SOC[i-1] > 0:
                        
                        tes_discharged[i] = np.min([energy_shortfall[i], tes_SOC[i-1],self.DischargeRate])
                        tes_SOC[i] = tes_SOC[i-1] - tes_discharged[i]

        return tes_discharged,excess_energy,tes_SOC
    def convert_electric_to_thermal(self,electric_energy):
        return self.ChargeEfficiency*electric_energy
    def charge_storage_silo(self,electric_energy):
        usable_electric_energy = np.where(electric_energy>self.ChargeRate,self.ChargeRate,electric_energy) #kWh-e
        usable_thermal_energy = self.ChargeEfficiency*usable_electric_energy #kWh-th
        
        


if __name__ == "__main__":
    config = {
        "ChargeRate": 315,
        "DischargeRate": 135,
        "StorageHours": 100,
        "ChargeEfficiency":98,
        "DischargeEfficiency":52,
        # "StorageEfficiency":97,
        "dt":3600,
    }
    #GWhth = MWe*(eta)
    #52 percent efficient
    #
    tes = HighTempTES(**config)
    

    TES = ThermalEnergyStorage()

    []