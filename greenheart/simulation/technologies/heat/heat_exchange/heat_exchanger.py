# naive model - just use heat capacity time mass

import numpy as np

from attrs import define, field
from typing import Optional
from hopp.type_dec import FromDictMixin

from greenheart.simulation.technologies.dispatch.control_model import ControlModel
from greenheart.simulation.technologies.heat.materials import Hydrogen

# TODO use materials from materials file


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
    T_H2_DRI: Optional[float] = field(default=900)  # [C] hydrogen input to DRI
    T_H2_electrolyzer: Optional[float] = field(
        default=80
    )  # [C] hydrogen outlet from electrolyzer
    T_H2_storage: Optional[float] = field(
        default=20
    )  # [C] hydrogen outlet temperature from storage

    T_P_storage: Optional[float] = field(
        default=1200
    )  # [C] particle outlet temperature from storage
    T_P_ondeck: Optional[float] = field(
        default=300
    )  # [C] particle temperature on-deck ready to heat and store NOTE: Is this 300 K?


# class HeatExchanger:
#     def __init__(self, material_hot = None, material_cold = None, power_to_material=False):

#         assert not (power_to_material and (material_hot is not None))


#         self.eta_HX = 0.99

#         self.power_to_material = power_to_material # if False, then material to material
#         self.material_hot = material_hot
#         self.material_cold = material_cold


#         self.Pin = 1.0

#         self.mdot_hot = 1.0
#         self.Tin_hot = 1.0
#         self.Tout_hot = 1.0


#         self.mdot_cold = 1.0
#         self.Tin_cold = 1.0
#         self.Tout_cold = 1.0

#         # electric particle heating
#         self.parameters = ["mdot_cold"]
#         self.inputs = ["Pin", "Tin_cold"]
#         self.outputs = ["Tout_cold"]

#         # electric hydrogen heating
#         self.parameters = ["Tout_cold"]
#         self.inputs = ["mdot_cold", "Tin_cold"]
#         self.outputs = ["Pin"]

#         # particle to hydrogen heating
#         self.parameters = ["mdot_hot", "Tout_cold"]
#         self.inputs = ["mdot_cold", "Tin_cold", "Tin_hot"]
#         self.outputs = ["Tout_hot"]


#         # maybe should have heat calculation output and step model output

#         if self.inputs == "mdot_hot":
#             self.hx_eqn = self.mat2mat_T1out


#         self.material_to_material = False # if it is sand to hydrogen
#         self.power_to_material = False # if it is power to sand or power to hydrogen


#         if self.material_to_material:
#             self.material1 = StorageParticle() # hot materical
#             self.material2 = H2Gas() # cold material

#         if self.power_to_material:
#             self.material2 = StorageParticle()
#             self.material2 = H2Gas()
#             # TODO figure out a good way to initialize this


#         self.particle_properties = StorageParticle()
#         self.hydrogen_properties = H2Gas()


#     def input_output(self, input):
#         output = self.hx_eqn(input)

#     def step(self):

#         # return what is the input and what is the output?
#         # H2 mdot should be the output assuming it is at the correct temperature
#         # Include a way to check temperature just in case
#         pass


#     def mat2mat_T1out(self, mdot1, T1in, mdot2, T2in, T2out):
#         pass


#     def mat2mat_mdot1(self, T1in, T1out, mdot2, T2in, T2out):
#         pass

#     def mat2mat_T2out(self, mdot1, T1in, T1out, mdot2, T2in):
#         pass


#     def P2mat_mdot2(self, Pin, T2in, T2out):
#         pass

#     def P2mat_T2out(self, Pin, mdot2, T2in):
#         pass

#     def P2mat_Pin(self, mdot2, T2in, T2out):
#         pass


#     def input_output(
#         self, mdot_h2, T_h2_in, T_h2_out, mdot_particle, T_particle_in, T_particle_out
#     ):

#         # if self.type == somethng: self.P2mat_Pin()


#         # mdot in kg / s

#         assert T_h2_out < T_particle_in, "heat wont transfer to hydrogen otherwise"

#         # Assume no heat is lost in the heat exchange
#         Qdot_to_hydrogen = self.hydrogen_properties.Cp * mdot_h2 * (T_h2_out - T_h2_in)
#         Qdot_from_particle = -Qdot_to_hydrogen
#         T_particle_out = (
#             Qdot_from_particle / (self.particle_properties.Cp * mdot_particle)
#             + T_particle_in
#         )

#         assert T_particle_out > -273 , "can be below absolute zero"

#         return Qdot_to_hydrogen, T_particle_out


#     def step(self, input, desired_output, step_index):
#         return np.mean([input, desired_output])


#     # Or maybe inheritance and the inherited class just sets the input output method = super().mat2mat_xxx


class HeatExchanger:
    def __init__(self, separate_cm_constraint=True):

        self.H2 = Hydrogen()
        self.C2K = 273.15

        # Put some of this stuff into a config file

        self.eta_HX = 1
        self.Tout_desired = 900  # [C]
        self.Tout = self.Tout_desired
        self.cp = 14.304  # [kJ kg^-1 K^-1]

        self.output_variable = "Tout"
        self.output_variable = "mdot"
        self.output_variable = "Qdot"

        duration = 8760
        self.input_store = np.zeros((4, duration))
        self.output_store = np.zeros((4, duration))
        self.wasted_store = np.zeros((4, duration))

        self.control_model = self.create_control_model(separate_cm_constraint)

    def calc_heating_ratio(self, T_in, T_out):
        delta_H_kwhpkg = self.H2.H_kwhpkg(T_out + self.C2K) - self.H2.H_kwhpkg(
            T_in + self.C2K
        )
        return delta_H_kwhpkg

    def step_heating(self, mdot_in, Qdot_in, T_in, T_out=None):

        if T_out is None:
            T_out = self.Tout_desired

        hr_kWhpkg = self.calc_heating_ratio(
            T_in, T_out
        )  # ratio of heat input to hydrogen input

        mdot_out = np.min([mdot_in, (1 / hr_kWhpkg) * Qdot_in])
        Qdot_used = mdot_out * hr_kWhpkg
        mdot_waste = mdot_in - mdot_out
        Qdot_waste = Qdot_in - Qdot_used

        return mdot_out, mdot_waste, Qdot_waste

    def create_control_model(self, separate_cm_constraint=True):
        m = 0
        n = 0
        p = 2
        o = 2

        # Assume the input is halfway between 20C from storage and 80C from electrolyzer
        heating_ratio = self.calc_heating_ratio(50, 900)

        A = np.zeros((n, n))
        B = np.zeros((n, m))
        C = np.zeros((p, n))
        D = np.zeros((p, m))
        E = np.zeros((n, o))
        F = np.array([[1, 0], [-heating_ratio, 1]])

        bounds_dict = {
            "u_lb": np.array([]),
            "u_ub": np.array([]),
            "x_lb": np.array([]),
            "x_ub": np.array([]),
            "y_lb": np.array([0, 0]),
            "y_ub": np.array([None, None]),
        }

        control_model = ControlModel(
            A, B, C, D, E, F, bounds=bounds_dict, discrete=True
        )

        if separate_cm_constraint:
            control_model.constraints(y_position=[1], constraint_type=["greater"])

        control_model.set_disturbance_domain([1, 1, 1])
        control_model.set_output_domain([0, 0, 1])
        control_model.set_disturbance_reshape(np.array([[0, 0, 1], [1, 1, 0]]))

        return control_model

    def inputs(self, inputs):

        # inputs = list of arrays shape (4, -)
        # inputs in format [[P, Q, mdot, T]]

        inputs = np.array(inputs)

        Pin = inputs[0]
        Qdotin = inputs[1]

        Qdot = Qdotin + Pin

        mdot = inputs[2]
        Tin = inputs[3]

        self.Qdot = Qdot
        self.mdot = mdot
        self.Tin = Tin

        return Qdot, mdot, Tin

    def calc_Tout(self, Qdotin, mdotin, Tin, Tout):
        # Assume all of Qdotin is used to heat all of mdotin

        # How hot can mdot be heated using Qdot?
        Tout = Tin + (self.eta_HX * Qdotin) / (mdotin * self.cp)

        mdotout = mdotin

        output = [0, 0, mdotout, Tout]
        wasted = [0, 0, 0, self.Tout_desired - Tout]

        return output, wasted

    def calc_mdot(self, Qdotin, mdotin, Tin, Tout):
        # Assume all of Qdotin is used to heat some of mdotin to Tout

        # How much mdot can be heated to Tout using Qdot?
        mdotout = (self.eta_HX * Qdotin) / (self.cp * (Tout - Tin))

        output = [0, 0, mdotout, Tout]
        wasted = [0, 0, mdotin - mdotout, 0]

        return output, wasted

    def calc_Qdot(self, Qdotin, mdotin, Tin, Tout):
        # Assume Assume some of Qdotin is used to heat all of mdotin to Tout

        # How much Qdot is needed to heat mdot to Tout?

        Qdotout = (mdotin * self.cp * (Tout - Tin)) / (self.eta_HX)

        output = [0, 0, mdotin, Tout]
        wasted = [0, Qdotin - Qdotout, 0, 0]

        return output, wasted

    def calculate(self, inputs):
        Qdotin, mdotin, Tin = self.inputs(inputs)

        if self.output_variable == "Tout":
            output, wasted = self.calc_Tout(Qdotin, mdotin, Tin, self.Tout_desired)

        elif self.output_variable == "mdot":
            output, wasted = self.calc_mdot(Qdotin, mdotin, Tin, self.Tout_desired)

        elif self.output_variable == "Qdot":
            output, wasted = self.calc_Qdot(Qdotin, mdotin, Tin, self.Tout_desired)

        return output, wasted

    def step(self, inputs, dispatch, step_index):

        Qdotin, mdotin, Tin = self.inputs(inputs)
        Qdotin = Qdotin * 3600

        mdotout, mdotwaste, Qdotwaste = self.step_heating(mdotin, Qdotin, Tin)
        output = np.array([0, 0, mdotout, 900])
        wasted = np.array([0, Qdotwaste, mdotwaste, Tin])


        # out_mdot, waste_mdot = self.calc_mdot(Qdotin, mdotin, Tin, self.Tout_desired)
        # out_Qdot, waste_Qdot = self.calc_Qdot(Qdotin, mdotin, Tin, self.Tout_desired)

        # if waste_mdot[2] < 0:  # Too much heat
        #     output = out_Qdot
        #     wasted = waste_Qdot

        # elif waste_Qdot[1] < 0:  # Too much mdot
        #     output = out_mdot
        #     wasted = waste_mdot

        # elif (waste_Qdot[1] == 0) and (waste_mdot[2] == 0):  # perfect
        #     # Both calculations should give the same
        #     output = out_mdot
        #     wasted = waste_mdot

        self.output = output
        self.wasted = wasted

        self.store_step(inputs, output, wasted, step_index)

        # TODO come back and make the output better

        # return output, wasted
        # return (output[2], output[3])
        u_passthrough = 0
        u_curtail = wasted[0:3]

        return mdotout, u_passthrough, u_curtail

    def store_step(self, input, output, wasted, step_index):
        self.input_store[:, step_index] = input
        self.output_store[:, step_index] = output
        self.wasted_store[:, step_index] = wasted


if __name__ == "__main__":
    HX = HeatExchanger(separate_cm_constraint=False)

    import matplotlib.pyplot as plt
    import matplotlib as mpl

    n_mdot = 25
    n_Qdot = 25

    mdot_in = np.linspace(0, 1000, n_mdot)
    Qdot_in = np.linspace(0, 1000, n_Qdot)

    mdot_out = np.zeros((n_mdot, n_Qdot))
    mdot_waste = np.zeros((n_mdot, n_Qdot))
    Qdot_waste = np.zeros((n_mdot, n_Qdot))

    cm_mdot_out = np.zeros((n_mdot, n_Qdot))
    cm_mdot_waste = np.zeros((n_mdot, n_Qdot))
    cm_Qdot_waste = np.zeros((n_mdot, n_Qdot))

    for i in range(len(mdot_in)):
        for j in range(len(Qdot_in)):
            mdo, mdw, qdw = HX.step_heating(mdot_in[i], Qdot_in[j], T_in=80)

            mdot_out[i, j] = mdo
            mdot_waste[i, j] = mdw
            Qdot_waste[i, j] = qdw

            y_HX = HX.control_model.F @ np.array([[mdot_in[i], Qdot_in[j]]]).T
            cm_mdot_out[i, j] = y_HX[0]
            cm_Qdot_waste[i, j] = y_HX[1]

    n_levels = 20
    mdot_out_levels = np.linspace(
        np.min([np.min(mdot_out), np.min(cm_mdot_out)]),
        np.max([np.max(mdot_out), np.max(cm_mdot_out)]),
        n_levels,
    )
    mdot_waste_levels = np.linspace(
        np.min([np.min(mdot_waste), np.min(cm_mdot_waste)]),
        np.max([np.max(mdot_waste), np.max(cm_mdot_waste)]),
        n_levels,
    )
    Qdot_waste_levels = np.linspace(
        np.min([np.min(Qdot_waste), np.min(cm_Qdot_waste)]),
        np.max([np.max(Qdot_waste), np.max(cm_Qdot_waste)]),
        n_levels,
    )

    mdot_norm = mpl.colors.TwoSlopeNorm(
        0,
        np.min([np.min(mdot_out), np.min(cm_mdot_out)]) - 0.1,
        np.max([np.max(mdot_out), np.max(cm_mdot_out)]),
    )
    mdot_waste_norm = mpl.colors.TwoSlopeNorm(
        0,
        np.min([np.min(mdot_waste), np.min(cm_mdot_waste)]) - 0.1,
        np.max([np.max(mdot_waste), np.max(cm_mdot_waste)]),
    )
    Qdot_waste_norm = mpl.colors.TwoSlopeNorm(
        0,
        np.min([np.min(Qdot_waste), np.min(cm_Qdot_waste)]) - 0.1,
        np.max([np.max(Qdot_waste), np.max(cm_Qdot_waste)]),
    )

    MD, QD = np.meshgrid(mdot_in, Qdot_in)

    fig, ax = plt.subplots(2, 3, sharex="all", sharey="all", layout="constrained")

    mdot_contour = ax[0, 0].contourf(
        MD, QD, mdot_out, levels=mdot_out_levels, norm=mdot_norm, cmap="seismic"
    )
    mwaste_contour = ax[0, 1].contourf(
        MD,
        QD,
        mdot_waste,
        levels=mdot_waste_levels,
        norm=mdot_waste_norm,
        cmap="seismic",
    )
    qwaste_contour = ax[0, 2].contourf(
        MD,
        QD,
        Qdot_waste,
        levels=Qdot_waste_levels,
        norm=Qdot_waste_norm,
        cmap="seismic",
    )

    fig.colorbar(
        mdot_contour,
        ax=ax[0, 0],
        location="bottom",
        ticks=[
            np.min([np.min(mdot_out), np.min(cm_mdot_out)]),
            np.max([np.max(mdot_out), np.max(cm_mdot_out)]),
        ],
    )
    fig.colorbar(
        mwaste_contour,
        ax=ax[0, 1],
        location="bottom",
        ticks=[
            np.min([np.min(mdot_waste), np.min(cm_mdot_waste)]),
            np.max([np.max(mdot_waste), np.max(cm_mdot_waste)]),
        ],
    )
    fig.colorbar(
        qwaste_contour,
        ax=ax[0, 2],
        location="bottom",
        ticks=[
            np.min([np.min(Qdot_waste), np.min(cm_Qdot_waste)]),
            np.max([np.max(Qdot_waste), np.max(cm_Qdot_waste)]),
        ],
    )

    ax[1, 0].contourf(
        MD, QD, cm_mdot_out, levels=mdot_out_levels, norm=mdot_norm, cmap="seismic"
    )
    ax[1, 1].contourf(
        MD,
        QD,
        cm_mdot_waste,
        levels=mdot_waste_levels,
        norm=mdot_waste_norm,
        cmap="seismic",
    )
    ax[1, 2].contourf(
        MD,
        QD,
        cm_Qdot_waste,
        levels=Qdot_waste_levels,
        norm=Qdot_waste_norm,
        cmap="seismic",
    )

    ax[0, 0].set_ylabel("Qdot in")
    ax[1, 0].set_ylabel("Qdot in")
    ax[1, 0].set_xlabel("mdot in")
    ax[1, 1].set_xlabel("mdot in")
    ax[1, 2].set_xlabel("mdot in")

    ax[0, 0].set_title("mdot out")
    ax[0, 1].set_title("mdot waste")
    ax[0, 2].set_title("Qdot waste")

    for i in range(ax.shape[0]):
        for j in range(ax.shape[1]):
            ax[i, j].set_aspect("equal")

    # Qdot, T_particle_out = HX.input_output(
    #     mdot_h2=100 / 3600,
    #     T_h2_in=25,
    #     T_h2_out=900,
    #     mdot_particle=1000 / 3600,
    #     T_particle_in=1500,
    #     T_particle_out=10,
    # )

    []
