import numpy as np
import scipy.interpolate


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
    g_interp: scipy.interpolate.interp1d
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


class Gas(ShomateEquation):
    pass


class Solid(ShomateEquation):
    pass


class Hydrogen(Gas):
    # H2
    # https://webbook.nist.gov/cgi/cbook.cgi?ID=C1333740&Mask=1&Type=JANAFG&Table=on#JANAFG

    T_max = 6000

    # Temperature (K)	298. to 1000.	1000. to 2500.	2500. to 6000.
    T = np.array([298, 1000, 2500, 6000])
    a = np.array([33.066178, 18.563083, 43.413560, 43.413560])
    b = np.array([-11.363417, 12.257357, -4.293079, -4.293079])
    c = np.array([11.432816, -2.859786, 1.272428, 1.272428])
    d = np.array([-2.772874, 0.268238, -0.096876, -0.096876])
    e = np.array([-0.158558, 1.977990, -20.533862, -20.533862])
    f = np.array([-9.980797, -1.147438, -38.515158, -38.515158])
    g = np.array([172.707974, 156.288133, 162.081354, 162.081354])
    h = np.array([0.0, 0.0, 0.0, 0.0])

    molar_mass = 2.01588  # [g mol^-1]


# class Water(Gas):
#     # H20
#     # https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Mask=1&Type=JANAFG&Table=on#JANAFG

#     T_max = 6000

#     # Temperature (K)	500. to 1700.	1700. to 6000.
#     T = np.array([500, 1700, 6000])
#     A = np.array([30.09200, 41.96426, 41.96426])
#     B = np.array([6.832514, 8.622053, 8.622053])
#     C = np.array([6.793435, -1.499780, -1.499780])
#     D = np.array([-2.534480, 0.098119, 0.098119])
#     E = np.array([0.082139, -11.15764, -11.15764])
#     F = np.array([-250.8810, -272.1797, -272.1797])
#     G = np.array([223.3967, 219.7809, 219.7809])
#     H = np.array([-241.8264, -241.8264, -241.8264])

    # molar_mass = 18.0153  # [g mol^-1]
    # Water only in gas phase with nist data

class Water(Gas):
    # H20
    # https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Mask=1&Type=JANAFG&Table=on#JANAFG

    T_max = 6000

    # Temperature (K)	500. to 1700.	1700. to 6000.
    T = np.array([298, 500, 1700, 6000])
    a = np.array([-203.6060, 30.09200, 41.96426, 41.96426])
    b = np.array([1523.290, 6.832514, 8.622053, 8.622053])
    c = np.array([-3196.413, 6.793435, -1.499780, -1.499780])
    d = np.array([ 2474.455, -2.534480, 0.098119, 0.098119])
    e = np.array([3.855326, 0.082139, -11.15764, -11.15764])
    f = np.array([ -256.5478, -250.8810, -272.1797, -272.1797])
    g = np.array([-488.7163, 223.3967, 219.7809, 219.7809])
    h = np.array([-285.8304, -241.8264, -241.8264, -241.8264])

    molar_mass = 18.0153  # [g mol^-1]

    # water with liquid phase too from 
    # https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Mask=2&Type=JANAFL&Table=on#JANAFL


# Temperature (K)	298. to 500.
# A	-203.6060
# B	1523.290
# C	-3196.413
# D	2474.455
# E	3.855326
# F	-256.5478
# G	-488.7163
# H	-285.8304


class Nitrogen(Gas):
    # N2
    # https://webbook.nist.gov/cgi/cbook.cgi?ID=C7727379&Mask=1&Type=JANAFG&Table=on#JANAFG

    T_max = 6000

    # Temperature (K)	100. to 500.	500. to 2000.	2000. to 6000.
    T = np.array([100, 500, 2000, 6000])
    a = np.array([28.98641, 19.50583, 35.51872, 35.51872])
    b = np.array([1.853978, 19.88705, 1.128728, 1.128728])
    c = np.array([-9.647459, -8.598535, -0.196103, -0.196103])
    d = np.array([16.63537, 1.369784, 0.014662, 0.014662])
    e = np.array([0.000117, 0.527601, -4.553760, -4.553760])
    f = np.array([-8.671914, -4.935202, -18.97091, -18.97091])
    g = np.array([226.4168, 212.3900, 224.9810, 224.9810])
    h = np.array([0.0, 0.0, 0.0, 0.0])

    molar_mass = 28.0134  # [g mol^-1]


class Iron(Solid):
    # Formula: Fe

    # Shomate equation parameters:
    # https://webbook.nist.gov/cgi/cbook.cgi?ID=C7439896&Units=SI&Mask=2&Type=JANAFS&Table=on#JANAFS

    T_max = 1809  # [K]

    # T_shomate = 298. to 700.	700. to 1042.	1042. to 1100.	1100. to 1809.	298. to 1809. # [K]
    T = np.array([298, 700, 1042, 1100, 1809])  # alpha-delta phase
    a = np.array([18.42868, -57767.65, -325.8859, -776.7387, -776.7387])
    b = np.array([24.64301, 137919.7, 28.92876, 919.4005, 919.4005])
    c = np.array([-8.913720, -122773.2, 0.000000, -383.7184, -383.7184])
    d = np.array([9.664706, 38682.42, 0.000000, 57.08148, 57.08148])
    e = np.array([-0.012643, 3993.080, 411.9629, 242.1369, 242.1369])
    f = np.array([-6.573022, 24078.67, 745.8231, 697.6234, 697.6234])
    g = np.array([42.51488, -87364.01, 241.8766, -558.3674, -558.3674])
    h = np.array([0.000000, 0.000000, 0.000000, 0.000000, 0.000000])

    molar_mass = 55.845  # [g mol^-1]


class IronOxide(Solid):
    # FeO

    # https://webbook.nist.gov/cgi/inchi?ID=C1345251&Mask=2&Type=JANAFS&Plot=on#JANAFS

    T_max = 1650
    T = np.array([298, 1650])

    a = np.array([45.75120, 45.75120])
    b = np.array([18.78553, 18.78553])
    c = np.array([-5.952201, -5.952201])
    d = np.array([0.852779, 0.852779])
    e = np.array([-0.081265, -0.081265])
    f = np.array([-286.7429, -286.7429])
    g = np.array([110.3120, 110.3120])
    h = np.array([-272.0441, -272.0441])

    molar_mass = 71.844  # [g mol^-1]


class Hematite(Solid):
    # Fe2 O3

    T_max = 2500  # [C]

    # Temperature (K)	298. to 950.	950. to 1050.	1050. to 2500.
    T = np.array([298, 950, 1050, 2500])
    a = np.array([93.43834, 150.6240, 110.9362, 110.9362])
    b = np.array([108.3577, 0.000000, 32.04714, 32.04714])
    c = np.array([-50.86447, 0.000000, -9.192333, -9.192333])
    d = np.array([25.58683, 0.000000, 0.901506, 0.901506])
    e = np.array([-1.611330, 0.000000, 5.433677, 5.433677])
    f = np.array([-863.2094, -875.6066, -843.1471, -843.1471])
    g = np.array([161.0719, 252.8814, 228.3548, 228.3548])
    h = np.array([-825.5032, -825.5032, -825.5032, -825.5032])

    molar_mass = 159.688  # [g mol^-1]


class Quartz(Solid):
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

    molar_mass = 60.0843 # [g mol^-1]