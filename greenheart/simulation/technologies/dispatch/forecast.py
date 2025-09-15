import numpy as np
import matplotlib.pyplot as plt


class Forecast:

    generation_profile: np.ndarray
    # resource_profile: np.ndarray # Ignore for now

    forecast_horizon: int  # number of timesteps to
    forecast_method: str

    def __init__(self, config, true_forecast, greenheart_config=None):
        self.config = config
        self.forecast_horizon = config["horizon"]
        self.forecast_method = config["method"]

        self.step_index = 0


        if greenheart_config is not None:
            self.greenheart_config = greenheart_config
            mpc_horizon = self.greenheart_config.greenheart_config[
                "realtime_simulation"
            ]["dispatch"]["mpc"]["horizon"]
            if mpc_horizon != self.forecast_horizon:
                self.forecast_horizon = mpc_horizon

        self.true_forecast = true_forecast
        self.perfect_forecast_profile = np.concatenate(
            [
                self.true_forecast,
                self.true_forecast[-1] * np.ones(self.forecast_horizon*2),
            ]
        )
        # getattr(self, f"_setup_{self.forecast_method}")(self.config["method_config"])

        self.make_forecast = getattr(self, f"_make_forecast_{self.forecast_method}")

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

        # f_persist = measurement * np.ones(self.forecast_horizon)
        # f_perfect = self.perfect_forecast_profile[step_index: step_index + self.forecast_horizon]
        # f_mixed = (perfect_fraction) * f_perfect + (1 - perfect_fraction) * f_persist
        # return f_mixed
        return (perfect_fraction) * self._forecast_perfect(measurement, step_index) + (
            1 - perfect_fraction
        ) * self._forecast_persistence(measurement, step_index)

    def _make_forecast_perfect_persistence_interp_horizon(self, measurement, step_index):

        args = (measurement, step_index)


        # perfect fraction 
        pf = self.config["method_config"]["perfect_fraction"]
        pd = self.config["method_config"]["perfect_duration"]



        # interp = np.linspace(pf, 0, self.forecast_horizon)
        interp = np.concatenate([np.linspace(pf, 0, pd), np.zeros(self.forecast_horizon - pd)])






        forecast = interp * self._forecast_perfect(*args) + (1 - interp) * self._forecast_persistence(*args)
        return forecast








    def _forecast_perfect(self, measurement, step_index):
        return self.perfect_forecast_profile[
            step_index : step_index + self.forecast_horizon
        ]

    def _forecast_persistence(self, measurement, step_index):
        return measurement * np.ones(self.forecast_horizon)

    # def _make_forecast_average_method(self ):
    #     pass

    # def _make_forecast_naive_method(self):
    #     pass

    # def _make_forecast_seasonal_naive_method(self):
    #     pass

    # def _make_forecast_drift_method(self):
    #     pass

    # @property
    # def forecast_perfect(self):
    #     return self.perfect_forecast_profile[self.step_index: self.step_index + self.forecast_horizon]

    # @property
    # def forecast_persistence(self):
    #     return

def plot_interp_options():
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        # "font.family": "sans-serif",
        # "font.sans-serif": "Helvetica",
        "font.sans-serif": "computer modern",
    })
    fig, ax = plt.subplots(1, 1, layout="constrained", figsize=(4, 2.75))
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_visible(False)

    alpha = 0.5
    horizon = 10
    short_horizon = 6


    interp_perfect = np.ones(horizon)

    interp_persistence = np.zeros(horizon)

    interp_interp = alpha * np.ones(horizon)

    interp_interp_horizon_weighted = np.linspace(alpha, 0, horizon)

    interp_interp_horizon_weighted_short = np.concatenate([np.linspace(alpha, 0, short_horizon), np.zeros(horizon - short_horizon)])

    ax.plot(interp_perfect, label="perfect")
    ax.plot(interp_interp, label="interpolated")
    ax.plot(interp_interp_horizon_weighted, label="interp. horizon-weighted")
    ax.plot(interp_interp_horizon_weighted_short, label="interp. short-weighted")
    ax.plot(interp_persistence, label="persistence")
    
    ax.legend(frameon=False, labelspacing=.125, loc = "upper right", bbox_to_anchor=(1, 0.94))

    ax.set_xticks(np.arange(horizon), np.arange(horizon))


    ax.text( 0.15, 0.7, f"$\\alpha_0 = {alpha}$")

    ax.set_xlabel("Timestep $k$")
    ax.set_ylabel("Interp. value $\\alpha_k$")


    []





if __name__ == "__main__":

    plot_interp_options()



    perfect_method_config = {}

    average_method_config = dict(history_window=10)

    persistence_method_config = dict()

    perfect_persistence_interp_config = dict(perfect_fraction=0.75)

    perfect_persistence_interp_horizon_config = dict(
        perfect_fraction = 0.75,
        perfect_duration = 12,
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


    configs = [config1, config2, config3, config4]


    t = np.arange(0, 8769, 1)
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

    def plot_forecaster(forecaster, ax):
        ax.plot(t, hybrid_profile, alpha=0.75, color="black", linewidth=2, label="Data")

        sim_length = 100

        for i in range(sim_length):
            if i % 12:
                continue

            prediction = forecaster.get_forecast(hybrid_profile[i], step_index=i)
            t_forecast = np.arange(i, i + forecaster.forecast_horizon, 1)

            ax.plot(t_forecast, prediction, linewidth=.75, color="orange")

        ax.set_xlim([0, sim_length + forecaster.forecast_horizon])
        ax.set_ylabel(forecaster.forecast_method)







    # forecaster = Forecast(config, hybrid_profile)
    # prediction = forecaster.get_forecast(hybrid_profile[0], step_index=0)





    fig, ax = plt.subplots(len(configs), 1, layout="constrained")

    # ax[0].plot(t, hybrid_profile, alpha=0.5, color="black", label="Data")

    # sim_length = 100

    # for i in range(sim_length):
    #     if i % 24:
    #         continue

    #     prediction = forecaster.get_forecast(hybrid_profile[i], step_index=i)
    #     t_forecast = np.arange(i, i + forecaster.forecast_horizon, 1)

    #     ax[0].plot(t_forecast, prediction, color="orange")

    # ax[0].set_xlim([0, sim_length + forecaster.forecast_horizon])


    for i in range(len(configs)):
        forecaster = Forecast(configs[i], hybrid_profile)
        plot_forecaster(forecaster, ax[i])

    []
