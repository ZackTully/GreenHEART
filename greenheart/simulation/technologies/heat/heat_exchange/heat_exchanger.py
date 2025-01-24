# naive model - just use heat capacity time mass

import numpy as np

from attrs import define, field
from typing import Optional
from hopp.type_dec import FromDictMixin


@define
class StorageParticle(FromDictMixin):
    # From ESG model
    Cp: Optional[float] = field(default=1155.0)  # J/kg-K
    k: Optional[float] = field(default=0.7)  # W/m-K
    T_melt: Optional[int] = field(default=1710)  # C
    name: Optional[str] = field(default="silica sand")


@define
class H2Gas(FromDictMixin):
    # From ESG model
    Cp: Optional[float] = field(default=14304)  # J/kg-K
    M_gas: Optional[float] = field(default=2.016)  # grams/mol
    name: Optional[str] = field(default="H2")
    T_final: Optional[float] = field(default=900)  # deg C

@define
class Temperatures(FromDictMixin):
    T_H2_DRI: Optional[float] = field(default=900) # [C] hydrogen input to DRI
    T_H2_electrolyzer: Optional[float] = field(default=80) # [C] hydrogen outlet from electrolyzer
    T_H2_storage: Optional[float] = field(default=20) # [C] hydrogen outlet temperature from storage

    T_P_storage: Optional[float] = field(default=1200) # [C] particle outlet temperature from storage
    T_P_ondeck: Optional[float] = field(default=300) # [C] particle temperature on-deck ready to heat and store NOTE: Is this 300 K?




class HeatExchanger:
    def __init__(self, material_hot = None, material_cold = None, power_to_material=False):

        assert not (power_to_material and (material_hot is not None))
        
        
        self.eta_HX = 0.99

        self.power_to_material = power_to_material # if False, then material to material
        self.material_hot = material_hot
        self.material_cold = material_cold


        self.Pin = 1.0

        self.mdot_hot = 1.0
        self.Tin_hot = 1.0
        self.Tout_hot = 1.0
        
        
        self.mdot_cold = 1.0
        self.Tin_cold = 1.0
        self.Tout_cold = 1.0

        # electric particle heating
        self.parameters = ["mdot_cold"]
        self.inputs = ["Pin", "Tin_cold"]
        self.outputs = ["Tout_cold"]

        # electric hydrogen heating 
        self.parameters = ["Tout_cold"]
        self.inputs = ["mdot_cold", "Tin_cold"]
        self.outputs = ["Pin"]

        # particle to hydrogen heating
        self.parameters = ["mdot_hot", "Tout_cold"]
        self.inputs = ["mdot_cold", "Tin_cold", "Tin_hot"]
        self.outputs = ["Tout_hot"]






        # maybe should have heat calculation output and step model output

        if self.inputs == "mdot_hot":
            self.hx_eqn = self.mat2mat_T1out



        self.material_to_material = False # if it is sand to hydrogen
        self.power_to_material = False # if it is power to sand or power to hydrogen 


        if self.material_to_material:
            self.material1 = StorageParticle() # hot materical
            self.material2 = H2Gas() # cold material

        if self.power_to_material:
            self.material2 = StorageParticle()
            self.material2 = H2Gas()
            # TODO figure out a good way to initialize this



        self.particle_properties = StorageParticle()
        self.hydrogen_properties = H2Gas()


    def input_output(self, input):
        output = self.hx_eqn(input)

    def step(self):
        
        # return what is the input and what is the output?
        # H2 mdot should be the output assuming it is at the correct temperature
        # Include a way to check temperature just in case
        pass



    def mat2mat_T1out(self, mdot1, T1in, mdot2, T2in, T2out):
        pass


    def mat2mat_mdot1(self, T1in, T1out, mdot2, T2in, T2out):
        pass

    def mat2mat_T2out(self, mdot1, T1in, T1out, mdot2, T2in):
        pass

    
    def P2mat_mdot2(self, Pin, T2in, T2out):
        pass

    def P2mat_T2out(self, Pin, mdot2, T2in):
        pass

    def P2mat_Pin(self, mdot2, T2in, T2out):
        pass



    def input_output(
        self, mdot_h2, T_h2_in, T_h2_out, mdot_particle, T_particle_in, T_particle_out
    ):
        
        # if self.type == somethng: self.P2mat_Pin()




        # mdot in kg / s

        assert T_h2_out < T_particle_in, "heat wont transfer to hydrogen otherwise"

        # Assume no heat is lost in the heat exchange
        Qdot_to_hydrogen = self.hydrogen_properties.Cp * mdot_h2 * (T_h2_out - T_h2_in)
        Qdot_from_particle = -Qdot_to_hydrogen
        T_particle_out = (
            Qdot_from_particle / (self.particle_properties.Cp * mdot_particle)
            + T_particle_in
        )

        assert T_particle_out > -273 , "can be below absolute zero"

        return Qdot_to_hydrogen, T_particle_out


    def step(self, input, desired_output, step_index):
        return np.mean([input, desired_output])


    # Or maybe inheritance and the inherited class just sets the input output method = super().mat2mat_xxx


if __name__ == "__main__":
    HX = HeatExchanger()

    Qdot, T_particle_out = HX.input_output(
        mdot_h2=100 / 3600,
        T_h2_in=25,
        T_h2_out=900,
        mdot_particle=1000 / 3600,
        T_particle_in=1500,
        T_particle_out=10,
    )

    []
