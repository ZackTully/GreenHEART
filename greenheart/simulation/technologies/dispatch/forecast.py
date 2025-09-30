import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as si

class Forecast:

    generation_profile: np.ndarray
    # resource_profile: np.ndarray # Ignore for now

    forecast_horizon: int  # number of timesteps to
    forecast_method: str

    def __init__(self, config, true_forecast, greenheart_config=None):
        self.config = config
        # self.forecast_horizon = config["horizon"]
        self.forecast_method = config["method"]

        self.step_index = 0

        if greenheart_config is not None:
            self.greenheart_config = greenheart_config
            mpc_horizon = self.greenheart_config.greenheart_config[
                "realtime_simulation"
            ]["dispatch"]["mpc"]["horizon"]
            # if mpc_horizon != self.forecast_horizon:
                # self.forecast_horizon = mpc_horizon
            self.forecast_horizon = mpc_horizon
        else:
            self.forecast_horizon = config["horizon"]

        self.true_forecast = true_forecast
        self.perfect_forecast_profile = np.concatenate(
            [
                self.true_forecast,
                self.true_forecast[-1] * np.ones(self.forecast_horizon * 10),
            ]
        )
        # getattr(self, f"_setup_{self.forecast_method}")(self.config["method_config"])

        self.make_forecast = getattr(self, f"_make_forecast_{self.forecast_method}")

        if hasattr(self, f"_setup_{self.forecast_method}"):
            getattr(self, f"_setup_{self.forecast_method}")()

        self.setup_storage()

    def setup_storage(self):
        self.measurement_history = []

    def get_forecast(self, measurement, step_index):
        self.step_index = step_index
        forecast = self.make_forecast(measurement, step_index)
        return forecast

    def _make_forecast_perfect_method(self, measurement, step_index):
        # forecast = self.perfect_forecast_profile[step_index: step_index + self.forecast_horizon]
        # return forecast
        return self._forecast_perfect(measurement, step_index)

    def _make_forecast_persistence_method(self, measurement, step_index):
        # return measurement * np.ones(self.forecast_horizon)
        return self._forecast_persistence(measurement, step_index)

    def _make_forecast_perfect_persistence_interp(self, measurement, step_index):

        perfect_fraction = self.config["method_config"]["perfect_fraction"]

        return (perfect_fraction) * self._forecast_perfect(measurement, step_index) + (
            1 - perfect_fraction
        ) * self._forecast_persistence(measurement, step_index)

    def _make_forecast_perfect_persistence_interp_horizon(
        self, measurement, step_index
    ):

        args = (measurement, step_index)

        # perfect fraction
        pf = self.config["method_config"]["perfect_fraction"]
        pd = self.config["method_config"]["perfect_duration"]

        # interp = np.linspace(pf, 0, self.forecast_horizon)
        interp = np.concatenate(
            [np.linspace(pf, 0, pd), np.zeros(self.forecast_horizon - pd)]
        )

        forecast = interp * self._forecast_perfect(*args) + (
            1 - interp
        ) * self._forecast_persistence(*args)
        return forecast

    def _forecast_perfect(self, measurement, step_index):
        return self.perfect_forecast_profile[
            step_index : step_index + self.forecast_horizon
        ]

    def _forecast_persistence(self, measurement, step_index):
        return measurement * np.ones(self.forecast_horizon)

    def _setup_filter_method(self):

        wc = self.config["method_config"]["w_cutoff"]
        Ts = self.config["method_config"]["Ts"]

        # Number of samples 
        self.n_cutoff = int(2 * np.pi / wc / Ts)



        K = 1
        alpha =  wc
        tau = 1 / alpha

        # Denominator
        a_s = np.array([tau, 1])

        # Numerator
        b_s = np.array([K * 1])

        TF_s = si.TransferFunction(b_s, a_s)

        TF_z = TF_s.to_discrete(dt=Ts)

        a_z, b_z = TF_z.den, TF_z.num

        self.filter_den = a_z
        self.filter_num = b_z

    def _make_forecast_filter_method(self, measurement, step_index):

        # d_perfect = self._forecast_perfect(measurement, step_index)

        d_perfect = self.perfect_forecast_profile[
            step_index : step_index + max(self.forecast_horizon, self.n_cutoff)
        ]


        # d_filtered = si.lfilter(self.filter_num, self.filter_den, d_perfect)
        d_filtered = si.filtfilt(self.filter_num, self.filter_den, d_perfect)

        # TODO May need to set the first value of the forecast to be the same as the measured disturbance. 


        return d_filtered[0:self.forecast_horizon]



def plot_interp_options():
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "serif",
            # "font.family": "sans-serif",
            # "font.sans-serif": "Helvetica",
            "font.sans-serif": "computer modern",
        }
    )
    fig, ax = plt.subplots(1, 1, layout="constrained", figsize=(4, 2.75))
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_visible(False)

    alpha = 0.5
    horizon = 10
    short_horizon = 6

    interp_perfect = np.ones(horizon)

    interp_persistence = np.zeros(horizon)

    interp_interp = alpha * np.ones(horizon)

    interp_interp_horizon_weighted = np.linspace(alpha, 0, horizon)

    interp_interp_horizon_weighted_short = np.concatenate(
        [np.linspace(alpha, 0, short_horizon), np.zeros(horizon - short_horizon)]
    )

    ax.plot(interp_perfect, label="perfect")
    ax.plot(interp_interp, label="interpolated")
    ax.plot(interp_interp_horizon_weighted, label="interp. horizon-weighted")
    ax.plot(interp_interp_horizon_weighted_short, label="interp. short-weighted")
    ax.plot(interp_persistence, label="persistence")

    ax.legend(
        frameon=False, labelspacing=0.125, loc="upper right", bbox_to_anchor=(1, 0.94)
    )

    ax.set_xticks(np.arange(horizon), np.arange(horizon))

    ax.text(0.15, 0.7, f"$\\alpha_0 = {alpha}$")

    ax.set_xlabel("Timestep $k$")
    ax.set_ylabel("Interp. value $\\alpha_k$")

    []


if __name__ == "__main__":

    from pathlib import Path

    # plot_interp_options()

    perfect_method_config = {}

    average_method_config = dict(history_window=10)

    persistence_method_config = dict()

    perfect_persistence_interp_config = dict(perfect_fraction=0.75)

    perfect_persistence_interp_horizon_config = dict(
        perfect_fraction=0.75,
        perfect_duration=12,
    )

    filter_config = dict(
        type="lpf",
        order=1,
        Ts = 3600,
        w_cutoff=2 * np.pi / (40 * 3600),
    )

    horizon = 24

    config1 = dict(
        horizon=horizon,
        method="perfect_method",
        method_config=perfect_method_config,
    )
    config2 = dict(
        horizon=horizon,
        method="persistence_method",
        method_config=persistence_method_config,
    )
    config3 = dict(
        horizon=horizon,
        method="perfect_persistence_interp",
        method_config=perfect_persistence_interp_config,
    )
    config4 = dict(
        horizon=horizon,
        method="perfect_persistence_interp_horizon",
        method_config=perfect_persistence_interp_horizon_config,
    )
    config5 = dict(
        horizon=horizon,
        method="filter_method",
        method_config=filter_config,
    )

    configs = [
        # config1,
        # config2,
        # config3,
        # config4,
        config5,
    ]

    t = np.arange(0, 8760, 1)
    # t = np.arange(0, 8769, 1)
    hybrid_profile = np.zeros(len(t))

    hybrid_profile += 50

    for i in range(10):
        hybrid_profile += (
            1
            * np.random.rand(1)
            * np.cos(
                t * (2 * np.pi) / (np.random.randn(1) * 6 + 6)
                + np.random.rand(1) * 2 * np.pi
            )
        )
    for i in range(10):
        hybrid_profile += (
            15
            * np.random.rand(1)
            * np.cos(
                t * (2 * np.pi) / (np.random.randn(1) * 6 + 24)
                + np.random.rand(1) * 2 * np.pi
            )
        )
    for i in range(10):
        hybrid_profile += (
            10
            * np.random.rand(1)
            * np.cos(
                t * (2 * np.pi) / (np.random.randn(1) * 5000 + 8769)
                + np.random.rand(1) * 2 * np.pi
            )
        )

    hybrid_profile = np.where(hybrid_profile < 0, 0, hybrid_profile)


    # data_root = Path(__file__).parent / "data"
    data_root = Path("/Users/ztully/Documents/hybrids_code/GH_scripts/greenheart_scripts/experimental_code/control_theory/forecasting/filtering/data")

    # Load some data files
    solar_data = np.load(data_root / "solar_data.npz")
    solar_generation = np.load(data_root / "solar_generation.npz")
    wind_data = np.load(data_root / "wind_data.npz")
    wind_generation = np.load(data_root / "wind_generation.npz")


    hybrid_profile = wind_generation["generation"] + solar_generation["generation"]


    def plot_forecaster(forecaster, ax):
        ax.plot(t, hybrid_profile, alpha=0.75, color="black", linewidth=2, label="Data")
        sim_length = 100
        ax.set_xlim([0, sim_length + forecaster.forecast_horizon])


        for i in range(sim_length):
            if i % 12:
                continue

            prediction = forecaster.get_forecast(hybrid_profile[i], step_index=i)
            t_forecast = np.arange(i, i + forecaster.forecast_horizon, 1)

            ax.plot(t_forecast, prediction, linewidth=1.25, color="orange")

        ax.set_ylabel(forecaster.forecast_method)

    # forecaster = Forecast(config, hybrid_profile)
    # prediction = forecaster.get_forecast(hybrid_profile[0], step_index=0)

    fig, ax = plt.subplots(len(configs), 1, layout="constrained", sharex="all")

    ax = np.atleast_1d(ax)

    for i in range(len(configs)):
        forecaster = Forecast(configs[i], hybrid_profile)
        plot_forecaster(forecaster, ax[i])

    []
