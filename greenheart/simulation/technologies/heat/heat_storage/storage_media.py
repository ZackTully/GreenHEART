
class StorageParticle:
    Cp = 1155 #J/kg-K
    k  = 0.7 #W/m-K
    T_melt = 1710
    name = "silica sand"
    def __init__(self,particle_config=None):
        if particle_config is not None:
            self.Cp = particle_config["heat capacity"]
            self.k = particle_config["thermal conductivity"]
            self.T_melt = particle_config["melting temperature"]
            self.name = particle_config["particle name"]


class StorageModel:
    T_storage = 1200
    T_charge = 300
    def __init__(self,particle: StorageParticle,tes_config=None):
        if tes_config is not None:
            self.T_storage = tes_config["storage temperature"]
            self.T_charge = tes_config["cooled particle inlet temperature"]
        self.particle = particle
    
class HeatedGas:
    name = "H2"
    Cp = 14.304*1e3 #J/kg-K
    Tf_target = 900
    T0 = 80
    M_gas = 2.016 #grams/mol molecular weight of gas
    def __init__(self,gas_config=None):
        if gas_config is not None:
            self.Cp = gas_config["heat capacity"]
            self.Tf_target = gas_config["target temperature"]
            self.T0 = gas_config["electrolyzer temperature"]
            self.name = gas_config["gas name"]
            self.M_gas = gas_config["molecular weight"]

class H2StorageTemp:
    def __init__(self):
        self.ref_h2_storage_capacity_kg = 500*1e3
        self.storage_outlet_temp_C = 20
        pass
    def salt_cavern(self):
        #120 bar at 765m underground
        # 0.156 bar/m depth pressure increase
        pressure_range = [55,152] #bar
        self.max_pressure_bar = 120

        pass
    def buried_pipes(self):
        #24" pipe schedule 60 can store H2 at max pressure of 100 bar
        # 500 tons of H2 storage requires 24,000 pipe segments
        pressure_range = [7,100] #bar
        self.max_pressure_bar = 100
        self.pipe_diameter = 12
        self.pipe_schedule = 60
        self.n_pipes = 24000
        

        pass
    def pressure_vessels(self):
        pressure_range = [1,10.4] #bar
        pass
    def lined_rock_cavern(self):
        #500 tonnes of H2 stored at 150 bar
        #required 62m diameter and 71m height
        pressure_range = [10,230] #bar
        self.max_pressure_bar = 150
        pass