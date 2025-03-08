# validation params
# precleaning
# regressors
# metric_weightings
# prediction intervals
# model_lists

"""
LOAD DATA
"""
import json
import datetime
import os
import numpy as np
import pandas as pd
from autots import AutoTS, create_regressor
from autots.evaluator.metrics import rps
from m6_functions import (
    load_data, forecast_returns_to_investment_agreement,
    prob_forecast_returns_to_investments, forecast_returns_to_investments,
    returns_to_obs_matrix
)

fred_key = "93873d40f10c20fe6f6e75b1ad0aed4d"  # https://fred.stlouisfed.org/docs/api/api_key.html
gsa_key = "zBYMGPb0bom4BEDsD8V0d3mwR6DnabT7Q9vkF1hz"  # https://open.gsa.gov/api/dap/
forecast_name = "m6_loop"
graph = False  # whether to plot graphs
csv_load = False
period = "4"
train_test = True
highlevel_validations = 2
# https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects
frequency = (
    "4W-FRI"  # "infer" for automatic alignment, but specific offsets are most reliable
)
if frequency == "4W-FRI":
    forecast_length = 1
elif frequency == "W-FRI":
    forecast_length = 4
if frequency == "D":
    forecast_length = 28
drop_most_recent = 0  # whether to discard the n most recent records (as incomplete)

n_jobs = "auto"  # or set to number of CPU cores
initial_training = "auto"  # set this to True on first run, or on reset, 'auto' looks for existing template, if found, sets to False.
evolve = False  # allow time series to progressively evolve on each run, if False, uses fixed template
archive_templates = False  # save a copy of the model template used with a timestamp
save_location = None  # "C:/Users/Colin/Downloads"  # directory to save templates to. Defaults to working dir
template_filename = f"autots_forecast_template_{forecast_name}.csv"
gens = 10

if save_location is not None:
    template_filename = os.path.join(save_location, template_filename)

if initial_training == "auto":
    initial_training = not os.path.exists(template_filename)
    if initial_training:
        print("No existing template found.")
    else:
        print("Existing template found.")

"""
Begin dataset retrieval
"""
df, regr_df_weekly = load_data(csv_load, train_test, forecast_length, fred_key, gsa_key, period=period)

regr_df_weekly = regr_df_weekly.reindex(df.index)

run_list = [
    {
        "name": "baseline_metric_mle",
        "prediction_interval": 0.9,
        "prediction_interval2": 0.6,
        "ensemble": ["dist", "simple", "mosaic", "horizontal-max", "subsample"],
        "transformer_list": "all",  # 'superfast'
        "transformer_max_depth": 8,
        "models_mode": "default",  # "deep", "regressor"
        "preclean": None,
        "preclean2": {
            "fillna": "ffill",  # "mean" or "median" are most consistent
            "transformations": {"0": "EWMAFilter"},
            "transformation_params": {
                "0": {"span": 3},
            },
        },
        "num_validations": 2,
        "models_to_validate": 0.35,
        "validation_method": "similarity",  # "similarity", "backwards", "seasonal 364",
        "model_list": [
            "ZeroesNaive",
            "LastValueNaive",
            "AverageValueNaive",
            "GLS",
            "SeasonalNaive",
            "GLM",
            "ETS",
            "FBProphet",
            # 'GluonTS',
            "UnobservedComponents",
            "VAR",
            "VECM",
            "WindowRegression",
            "DatepartRegression",
            # "MultivariateRegression",
            "UnivariateMotif",
            "MultivariateMotif",
            "SectionalMotif",
            "NVAR",
            "Theta",
            "ARDL",
            # additions
            # "ARIMA",
            # 'VARMAX',
            # "UnivariateRegression",
            # 'DynamicFactor',
            # "RollingRegression",
            # "DynamicFactorMQ",
        ],
        "similarity_validation_params": {
            "stride_size": 1,
            "distance_metric": "nan_euclidean",
            "include_differenced": True,
            "window_size": 30,
        },
        "regr_kwargs": {
            "scale": True,
            "summarize": "auto",
            "backfill": "bfill",
            "fill_na": "spline",
            "holiday_countries": ["US"],
            "datepart_method": "simple_2",
        },
        "metric_weighting": {
            'smape_weighting': 2,
            'mae_weighting': 2,
            'rmse_weighting': 1,
            'made_weighting': 0,
            'mage_weighting': 0,
            'mle_weighting': 3,
            'imle_weighting': 0,
            'spl_weighting': 2,
            'containment_weighting': 0,
            'contour_weighting': 0,
            'runtime_weighting': 0,
        }
    },
    # LINE BREAK
    {
        "name": "baseline_metric_shapes",
        "prediction_interval": 0.9,
        "prediction_interval2": 0.6,
        "ensemble": ["dist", "simple", "mosaic", "horizontal-max", "subsample"],
        "transformer_list": "all",  # 'superfast'
        "transformer_max_depth": 8,
        "models_mode": "default",  # "deep", "regressor"
        "preclean": None,
        "preclean2": {
            "fillna": "ffill",  # "mean" or "median" are most consistent
            "transformations": {"0": "EWMAFilter"},
            "transformation_params": {
                "0": {"span": 3},
            },
        },
        "num_validations": 2,
        "models_to_validate": 0.35,
        "validation_method": "similarity",  # "similarity", "backwards", "seasonal 364",
        "model_list": [
            "ZeroesNaive",
            "LastValueNaive",
            "AverageValueNaive",
            "GLS",
            "SeasonalNaive",
            "GLM",
            "ETS",
            "FBProphet",
            # 'GluonTS',
            "UnobservedComponents",
            "VAR",
            "VECM",
            "WindowRegression",
            "DatepartRegression",
            # "MultivariateRegression",
            "UnivariateMotif",
            "MultivariateMotif",
            "SectionalMotif",
            "NVAR",
            "Theta",
            "ARDL",
            # additions
            # "ARIMA",
            # 'VARMAX',
            # "UnivariateRegression",
            # 'DynamicFactor',
            # "RollingRegression",
            # "DynamicFactorMQ",
        ],
        "similarity_validation_params": {
            "stride_size": 1,
            "distance_metric": "nan_euclidean",
            "include_differenced": True,
            "window_size": 30,
        },
        "regr_kwargs": {
            "scale": True,
            "summarize": "auto",
            "backfill": "bfill",
            "fill_na": "spline",
            "holiday_countries": ["US"],
            "datepart_method": "simple_2",
        },
        "metric_weighting": {
            'smape_weighting': 2,
            'mae_weighting': 2,
            'rmse_weighting': 1,
            'made_weighting': 2,
            'mage_weighting': 0,
            'mle_weighting': 0,
            'imle_weighting': 0.1,
            'spl_weighting': 2,
            'containment_weighting': 0,
            'contour_weighting': 1,
            'runtime_weighting': 0,
        }
    },
    # LINE BREAK
    {
        "name": "superfast_preclean",
        "prediction_interval": 0.9,
        "prediction_interval2": 0.6,
        "ensemble": None,
        # "ensemble": ["dist", "simple", "mosaic", "horizontal-max", "subsample"],
        "transformer_list": "all",  # 'superfast'
        "transformer_max_depth": 8,
        "models_mode": "default",  # "deep", "regressor"
        "preclean": {
            "fillna": "ffill",  # "mean" or "median" are most consistent
            "transformations": {"0": "EWMAFilter"},
            "transformation_params": {
                "0": {"span": 2},
            },
        },
        "num_validations": 2,
        "models_to_validate": 0.35,
        "validation_method": "similarity",  # "similarity", "backwards", "seasonal 364",
        "model_list": "superfast",
        "similarity_validation_params": {
            "stride_size": 1,
            "distance_metric": "nan_euclidean",
            "include_differenced": True,
            "window_size": 30,
        },
        "regr_kwargs": {
            "scale": True,
            "summarize": "auto",
            "backfill": "bfill",
            "fill_na": "spline",
            "holiday_countries": ["US"],
            "datepart_method": "simple_2",
        },
        "metric_weighting": {
            'smape_weighting': 2,
            'mae_weighting': 2,
            'rmse_weighting': 1,
            'made_weighting': 0,
            'mage_weighting': 0,
            'mle_weighting': 0,
            'imle_weighting': 0.1,
            'spl_weighting': 2,
            'containment_weighting': 0,
            'contour_weighting': 0,
            'runtime_weighting': 0,
        }
    },
    # LINE BREAK
    {
        "name": "fast_preclean",
        "prediction_interval": 0.9,
        "prediction_interval2": 0.6,
        "ensemble": None,
        # "ensemble": ["dist", "simple", "mosaic", "horizontal-max", "subsample"],
        "transformer_list": "all",  # 'superfast'
        "transformer_max_depth": 8,
        "models_mode": "default",  # "deep", "regressor"
        "preclean": {
            "fillna": "ffill",  # "mean" or "median" are most consistent
            "transformations": {"0": "EWMAFilter"},
            "transformation_params": {
                "0": {"span": 2},
            },
        },
        "num_validations": 2,
        "models_to_validate": 0.35,
        "validation_method": "similarity",  # "similarity", "backwards", "seasonal 364",
        "model_list": "fast",
        "similarity_validation_params": {
            "stride_size": 1,
            "distance_metric": "nan_euclidean",
            "include_differenced": True,
            "window_size": 30,
        },
        "regr_kwargs": {
            "scale": True,
            "summarize": "auto",
            "backfill": "bfill",
            "fill_na": "spline",
            "holiday_countries": ["US"],
            "datepart_method": "simple_2",
        },
        "metric_weighting": {
            'smape_weighting': 2,
            'mae_weighting': 2,
            'rmse_weighting': 1,
            'made_weighting': 0,
            'mage_weighting': 0,
            'mle_weighting': 0,
            'imle_weighting': 0.1,
            'spl_weighting': 2,
            'containment_weighting': 0,
            'contour_weighting': 0,
            'runtime_weighting': 0,
        }
    },
    # LINE BREAK
    {
        "name": "fast_preclean_motifs",
        "prediction_interval": 0.9,
        "prediction_interval2": 0.6,
        "ensemble": None,
        # "ensemble": ["dist", "simple", "mosaic", "horizontal-max", "subsample"],
        "transformer_list": "all",  # 'superfast'
        "transformer_max_depth": 8,
        "models_mode": "default",  # "deep", "regressor"
        "preclean": {
            "fillna": "ffill",  # "mean" or "median" are most consistent
            "transformations": {"0": "EWMAFilter"},
            "transformation_params": {
                "0": {"span": 2},
            },
        },
        "num_validations": 2,
        "models_to_validate": 0.35,
        "validation_method": "similarity",  # "similarity", "backwards", "seasonal 364",
        "model_list": "motifs",
        "similarity_validation_params": {
            "stride_size": 1,
            "distance_metric": "nan_euclidean",
            "include_differenced": True,
            "window_size": 30,
        },
        "regr_kwargs": {
            "scale": True,
            "summarize": "auto",
            "backfill": "bfill",
            "fill_na": "spline",
            "holiday_countries": ["US"],
            "datepart_method": "simple_2",
        },
        "metric_weighting": {
            'smape_weighting': 2,
            'mae_weighting': 2,
            'rmse_weighting': 1,
            'made_weighting': 0,
            'mage_weighting': 0,
            'mle_weighting': 0,
            'imle_weighting': 0.1,
            'spl_weighting': 2,
            'containment_weighting': 0,
            'contour_weighting': 0,
            'runtime_weighting': 0,
        }
    },
    # LINE BREAK
    {
        "name": "fast_preclean_ODA_MLE",
        "prediction_interval": 0.9,
        "prediction_interval2": 0.6,
        "ensemble": None,
        # "ensemble": ["dist", "simple", "mosaic", "horizontal-max", "subsample"],
        "transformer_list": "all",  # 'superfast'
        "transformer_max_depth": 8,
        "models_mode": "default",  # "deep", "regressor"
        "preclean": {
            "fillna": "ffill",  # "mean" or "median" are most consistent
            "transformations": {"0": "EWMAFilter"},
            "transformation_params": {
                "0": {"span": 2},
            },
        },
        "num_validations": 2,
        "models_to_validate": 0.35,
        "validation_method": "similarity",  # "similarity", "backwards", "seasonal 364",
        "model_list": "fast",
        "similarity_validation_params": {
            "stride_size": 1,
            "distance_metric": "nan_euclidean",
            "include_differenced": True,
            "window_size": 30,
        },
        "regr_kwargs": {
            "scale": True,
            "summarize": "auto",
            "backfill": "bfill",
            "fill_na": "spline",
            "holiday_countries": ["US"],
            "datepart_method": "simple_2",
        },
        "metric_weighting": {
            'smape_weighting': 2,
            'mae_weighting': 2,
            'rmse_weighting': 1,
            'made_weighting': 0,
            'mage_weighting': 0,
            'mle_weighting': 3,
            'imle_weighting': 0.1,
            'spl_weighting': 2,
            'containment_weighting': 0,
            'contour_weighting': 0,
            'runtime_weighting': 0,
            'oda_weighting': 5,
        }
    },
    # LINE BREAK
    {
        "name": "fast_preclean_moreval",
        "prediction_interval": 0.9,
        "prediction_interval2": 0.6,
        "ensemble": None,
        # "ensemble": ["dist", "simple", "mosaic", "horizontal-max", "subsample"],
        "transformer_list": "all",  # 'superfast'
        "transformer_max_depth": 8,
        "models_mode": "default",  # "deep", "regressor"
        "preclean": {
            "fillna": "ffill",  # "mean" or "median" are most consistent
            "transformations": {"0": "EWMAFilter"},
            "transformation_params": {
                "0": {"span": 2},
            },
        },
        "num_validations": 8,
        "models_to_validate": 0.35,
        "validation_method": "similarity",  # "similarity", "backwards", "seasonal 364",
        "model_list": "fast",
        "similarity_validation_params": {
            "stride_size": 1,
            "distance_metric": "nan_euclidean",
            "include_differenced": True,
            "window_size": 30,
        },
        "regr_kwargs": {
            "scale": True,
            "summarize": "auto",
            "backfill": "bfill",
            "fill_na": "spline",
            "holiday_countries": ["US"],
            "datepart_method": "simple_2",
        },
        "metric_weighting": {
            'smape_weighting': 2,
            'mae_weighting': 2,
            'rmse_weighting': 1,
            'made_weighting': 0,
            'mage_weighting': 0,
            'mle_weighting': 0,
            'imle_weighting': 0.1,
            'spl_weighting': 2,
            'containment_weighting': 0,
            'contour_weighting': 0,
            'runtime_weighting': 0,
        }
    },
    # LINE BREAK
    {
        "name": "oda",
        "prediction_interval": 0.9,
        "prediction_interval2": 0.6,
        "ensemble": ['horizontal-max', 'mosaic', 'horizontal'],
        "transformer_list": "all",  # 'superfast'
        "transformer_max_depth": 8,
        "models_mode": "default",  # "deep", "regressor"
        "preclean": {
            "fillna": "ffill",  # "mean" or "median" are most consistent
            "transformations": {"0": "EWMAFilter"},
            "transformation_params": {
                "0": {"span": 3},
            },
        },
        "num_validations": 12,
        "models_to_validate": 0.35,
        "validation_method": "backwards",  # "similarity", "backwards", "seasonal 364",
        "model_list": [
            "ZeroesNaive",
            "LastValueNaive",
            "AverageValueNaive",
            "GLS",
            "SeasonalNaive",
            "GLM",
            "ETS",
            "UnobservedComponents",
            "VAR",
            "VECM",
            "WindowRegression",
            "DatepartRegression",
            "UnivariateMotif",
            "MultivariateMotif",
            "SectionalMotif",
            "NVAR",
            "Theta",
            "ARDL",
        ],
        "similarity_validation_params": {
            "stride_size": 1,
            "distance_metric": "nan_euclidean",
            "include_differenced": False,
            "window_size": 30,
        },
        "regr_kwargs": {
            "scale": True,
            "summarize": "auto",
            "backfill": "bfill",
            "fill_na": "spline",
            "holiday_countries": ["US"],
            "datepart_method": "simple_2",
        },
        "metric_weighting": {
            'smape_weighting': 2,
            'mae_weighting': 2,
            'rmse_weighting': 1,
            'made_weighting': 2,
            'mage_weighting': 0,
            'mle_weighting': 0,
            'imle_weighting': 0.1,
            'spl_weighting': 2,
            'containment_weighting': 0,
            'contour_weighting': 1,
            'runtime_weighting': 0,
            'oda_weighting': 5,
        }
    },
    # LINE BREAK
    {
        "name": "oda_mle",
        "prediction_interval": 0.9,
        "prediction_interval2": 0.6,
        "ensemble": ['horizontal-max', 'mosaic', 'horizontal'],
        "transformer_list": "all",  # 'superfast'
        "transformer_max_depth": 8,
        "models_mode": "default",  # "deep", "regressor"
        "preclean": {
            "fillna": "ffill",  # "mean" or "median" are most consistent
            "transformations": {"0": "EWMAFilter"},
            "transformation_params": {
                "0": {"span": 3},
            },
        },
        "num_validations": 12,
        "models_to_validate": 0.35,
        "validation_method": "backwards",  # "similarity", "backwards", "seasonal 364",
        "model_list": [
            "ZeroesNaive",
            "LastValueNaive",
            "AverageValueNaive",
            "GLS",
            "SeasonalNaive",
            "GLM",
            "ETS",
            "UnobservedComponents",
            "VAR",
            "VECM",
            "WindowRegression",
            "DatepartRegression",
            "UnivariateMotif",
            "MultivariateMotif",
            "SectionalMotif",
            "NVAR",
            "Theta",
            "ARDL",
        ],
        "similarity_validation_params": {
            "stride_size": 1,
            "distance_metric": "nan_euclidean",
            "include_differenced": False,
            "window_size": 30,
        },
        "regr_kwargs": {
            "scale": True,
            "summarize": "auto",
            "backfill": "bfill",
            "fill_na": "spline",
            "holiday_countries": ["US"],
            "datepart_method": "simple_2",
        },
        "metric_weighting": {
            'smape_weighting': 2,
            'mae_weighting': 2,
            'rmse_weighting': 1,
            'made_weighting': 2,
            'mage_weighting': 0,
            'mle_weighting': 2,
            'imle_weighting': 0.1,
            'spl_weighting': 2,
            'containment_weighting': 0,
            'contour_weighting': 1,
            'runtime_weighting': 0,
            'oda_weighting': 5,
        }
    },
    # LINE BREAK
    {
        "name": "oda_noensemble",
        "prediction_interval": 0.9,
        "prediction_interval2": 0.6,
        "ensemble": None,
        "transformer_list": "all",  # 'superfast'
        "transformer_max_depth": 8,
        "models_mode": "default",  # "deep", "regressor"
        "preclean": {
            "fillna": "ffill",  # "mean" or "median" are most consistent
            "transformations": {"0": "EWMAFilter"},
            "transformation_params": {
                "0": {"span": 3},
            },
        },
        "num_validations": 12,
        "models_to_validate": 0.35,
        "validation_method": "backwards",  # "similarity", "backwards", "seasonal 364",
        "model_list": [
            "ZeroesNaive",
            "LastValueNaive",
            "AverageValueNaive",
            "GLS",
            "SeasonalNaive",
            "GLM",
            "ETS",
            "UnobservedComponents",
            "VAR",
            "VECM",
            "WindowRegression",
            "DatepartRegression",
            "UnivariateMotif",
            "MultivariateMotif",
            "SectionalMotif",
            "NVAR",
            "Theta",
            "ARDL",
        ],
        "similarity_validation_params": {
            "stride_size": 1,
            "distance_metric": "nan_euclidean",
            "include_differenced": False,
            "window_size": 30,
        },
        "regr_kwargs": {
            "scale": True,
            "summarize": "auto",
            "backfill": "bfill",
            "fill_na": "spline",
            "holiday_countries": ["US"],
            "datepart_method": "simple_2",
        },
        "metric_weighting": {
            'smape_weighting': 2,
            'mae_weighting': 2,
            'rmse_weighting': 1,
            'made_weighting': 2,
            'mage_weighting': 0,
            'mle_weighting': 0,
            'imle_weighting': 0.1,
            'spl_weighting': 2,
            'containment_weighting': 0,
            'contour_weighting': 1,
            'runtime_weighting': 0,
            'oda_weighting': 5,
        }
    },
]

metric_df_list = []
investment_df_list = []
result_df_list = []

for current_run in run_list:
    run = current_run["name"]
    for val in range(highlevel_validations):
        print(f"Starting {run} in validation {val + 1}")
        current_slice = df.head(
            df.shape[0] - (val * forecast_length)
        )
        df_train = current_slice[: -forecast_length].copy()
        df_test = current_slice[-forecast_length: ].copy()
        regr_train, regr_fcst = create_regressor(
            regr_df_weekly.reindex(current_slice.index),
            forecast_length=forecast_length,
            frequency=frequency,
            drop_most_recent=drop_most_recent,
            **current_run.get("regr_kwargs", {})
        )
        df_train = df_train.iloc[forecast_length:]
        regr_train = regr_train.iloc[forecast_length:]

        model = AutoTS(
            forecast_length=forecast_length,
            frequency=frequency,
            prediction_interval=current_run["prediction_interval"],
            ensemble=current_run.get("ensemble", None),
            model_list=current_run.get("model_list", "fast"),
            transformer_list=current_run.get("transformer_list", "fast"),
            transformer_max_depth=current_run.get("transformer_max_depth", 8),
            max_generations=gens,
            metric_weighting=current_run["metric_weighting"],
            initial_template="random",
            aggfunc="sum",
            models_to_validate=current_run.get("models_to_validate", 0.35),
            model_interrupt=True,
            num_validations=current_run.get("num_validations", 3),
            validation_method=current_run.get("validation_method", "similarity"),
            constraint=None,
            drop_most_recent=drop_most_recent,
            preclean=current_run.get("preclean", None),
            models_mode=current_run.get("models_mode", "random"),
            current_model_file=f"current_model_{forecast_name}",
            n_jobs=n_jobs,
            verbose=0,
        )
        model.similarity_validation_params = current_run["similarity_validation_params"]

        if not initial_training:
            if evolve:
                model.import_template(template_filename, method="addon")
            else:
                model.import_template(template_filename, method="only")

        model = model.fit(
            df_train,
            future_regressor=regr_train,
        )

        interval_list = [current_run["prediction_interval"], current_run["prediction_interval2"]]
        prediction = model.predict(
            future_regressor=regr_fcst,
            prediction_interval=interval_list,
            verbose=1,
            fail_on_forecast_nan=True,
        )

        forecasts_df = prediction[str(interval_list[0])].forecast
        forecasts_upper_df = prediction[str(interval_list[0])].upper_forecast
        forecasts_lower_df = prediction[str(interval_list[0])].lower_forecast
        forecasts_mid_upper_df = prediction[str(interval_list[1])].upper_forecast
        forecasts_mid_lower_df = prediction[str(interval_list[1])].lower_forecast
        forecast_return = (forecasts_df.iloc[-1] - df_train.iloc[-1]) / df_train.iloc[-1]
        uforecast_return = (forecasts_upper_df.iloc[-1] - df_train.iloc[-1]) / df_train.iloc[-1]
        lforecast_return = (forecasts_lower_df.iloc[-1] - df_train.iloc[-1]) / df_train.iloc[-1]
        muforecast_return = (forecasts_mid_upper_df.iloc[-1] - df_train.iloc[-1]) / df_train.iloc[-1]
        mlforecast_return = (forecasts_mid_lower_df.iloc[-1] - df_train.iloc[-1]) / df_train.iloc[-1]

        forecasts_obs = returns_to_obs_matrix(forecast_return)
        u_forecasts_obs = returns_to_obs_matrix(uforecast_return)
        l_forecasts_obs = returns_to_obs_matrix(lforecast_return)
        mu_forecasts_obs = returns_to_obs_matrix(muforecast_return)
        ml_forecasts_obs = returns_to_obs_matrix(mlforecast_return)
        ranked_prediction = (forecasts_obs + u_forecasts_obs + l_forecasts_obs) / 3
        ranked_prediction_2int = (
            forecasts_obs + u_forecasts_obs + l_forecasts_obs
            + mu_forecasts_obs + ml_forecasts_obs
        ) / 5
        weighted_ranked = (forecasts_obs * 2 + u_forecasts_obs + l_forecasts_obs) / 4
        weighted_ranked_2int = (
            forecasts_obs + u_forecasts_obs * 2 + l_forecasts_obs * 2
            + mu_forecasts_obs + ml_forecasts_obs
        ) / 7
        naive_prediction = pd.DataFrame(
            0.2, index=ranked_prediction.index, columns=ranked_prediction.columns
        )
        naive_prediction5 = pd.DataFrame(
            np.tile(np.array([0.15, 0.2, 0.3, 0.2, 0.15]).reshape(1, -1), (df.shape[1], 1)),
            index=ranked_prediction.index,
            columns=ranked_prediction.columns,
        )
        ranked_w_naive_obs = (
            forecasts_obs + u_forecasts_obs + l_forecasts_obs + naive_prediction
        ) / 4
        weighted_ranked_2int_naive = (
            forecasts_obs + u_forecasts_obs * 2 + l_forecasts_obs * 2
            + mu_forecasts_obs + ml_forecasts_obs + naive_prediction * 3
        ) / 10
        weighted_ranked_naive_2 = (
            forecasts_obs * 2 + u_forecasts_obs + l_forecasts_obs
            + naive_prediction5 * 10
        ) / 14
        weighted_ranked_naive_3 = (
            forecasts_obs * 2 + u_forecasts_obs + l_forecasts_obs
            + naive_prediction5 * 4
        ) / 8
        if train_test:
            df_forecast = prediction[str(interval_list[0])]
            model_error = df_forecast.evaluate(df_test)
            result = pd.DataFrame(
                {
                    'run': run,
                    'Model': df_forecast.model_name,
                    'ModelParameters': json.dumps(df_forecast.model_parameters),
                    'TransformationParameters': json.dumps(
                        df_forecast.transformation_parameters
                    ),
                    'TransformationRuntime': df_forecast.transformation_runtime,
                    'FitRuntime': df_forecast.fit_runtime,
                    'PredictRuntime': df_forecast.predict_runtime,
                    'Exceptions': np.nan,
                    'ValidationRound': val,
                },
                index=[0],
            )
            a = pd.DataFrame(
                model_error.avg_metrics_weighted.rename(lambda x: x + '_weighted')
            ).transpose()
            result = pd.concat(
                [result, pd.DataFrame(model_error.avg_metrics).transpose(), a], axis=1
            )
            result.index=[run]
            metric_df = pd.DataFrame(index=[run])
            metric_df["ValidationRound"] = val
            investment_df = pd.DataFrame(index=[run])
            investment_df["ValidationRound"] = val
            actual_returns = (df_test.iloc[-1] - df_train.iloc[-1]) / df_train.iloc[-1]
            actuals = returns_to_obs_matrix(actual_returns)
            rps_forecast = rps(ranked_prediction, actuals)
            # print(model)
            metric_df['RPS_Forecasts'] = rps_forecast.sum()
            # print(f"RPS of forecasts: {rps_forecast.sum()}")
            rps_forecast_2int = rps(ranked_prediction_2int, actuals)
            metric_df['RPS_Forecasts_2pred_int'] = rps_forecast_2int.sum()
            # print(f"RPS of forecasts (2 pred int): {rps_forecast_2int.sum()}")
            rps_forecast_pt = rps(forecasts_obs, actuals)
            metric_df['RPS_point_forecast_only'] = rps_forecast_pt.sum()
            # print(f"RPS of point forecasts: {rps_forecast_pt.sum()}")
            rps_hinge = rps((u_forecasts_obs + l_forecasts_obs) / 2, actuals)
            metric_df['RPS_hinge_forecast'] = rps_hinge.sum()
            # print(f"RPS of hinge forecasts: {rps_hinge.sum()}")
            rps_rank_w_naive = rps(ranked_w_naive_obs, actuals)
            metric_df['RPS_forecast+naive'] = rps_rank_w_naive.sum()
            # print(f"RPS of rank+naive forecasts: {rps_rank_w_naive.sum()}")
            rps_rank_weighted = rps(weighted_ranked, actuals)
            metric_df['RPS_forecasts_weighted'] = rps_rank_weighted.sum()
            # print(f"RPS of weighted forecasts: {rps_rank_weighted.sum()}")
            rps_rank_weighted_2int = rps(weighted_ranked_2int, actuals)
            metric_df['RPS_forecasts_weighted_2int'] = rps_rank_weighted_2int.sum()
            # print(f"RPS of weighted 2 interval forecasts: {rps_rank_weighted_2int.sum()}")
            rps_rank_weighted_2int_naive = rps(weighted_ranked_2int_naive, actuals)
            metric_df['RPS_forecasts_weighted_2int_naive'] = rps_rank_weighted_2int_naive.sum()
            # print(f"RPS of weighted 2 interval forecasts + naive: {rps_rank_weighted_2int_naive.sum()}")
            rps_naive = rps(naive_prediction, actuals)
            metric_df['RPS_naive'] = rps_naive.sum()
            # print(f"RPS of naive: {rps_naive.sum()}")
            rps_naive2 = rps(weighted_ranked_naive_2, actuals)
            metric_df['RPS_weighted_ranked_naive_2'] = rps_naive2.sum()
            # print(f"RPS of naive2: {rps_naive2.sum()}")
            rps_naive3 = rps(weighted_ranked_naive_3, actuals)
            metric_df['RPS_weighted_ranked_naive_3'] = rps_naive3.sum()
            rps_naive5 = rps(naive_prediction5, actuals)
            metric_df['RPS_naive5'] = rps_naive5.sum()
            print(metric_df.iloc[0])
            investment = forecast_returns_to_investments(forecast_return)
            prob_investment = prob_forecast_returns_to_investments(
                [forecast_return, uforecast_return, lforecast_return]
            )
            prob_investment_2int = prob_forecast_returns_to_investments(
                [
                    forecast_return,
                    uforecast_return,
                    lforecast_return,
                    muforecast_return,
                    mlforecast_return,
                ]
            )
            hinge_investment = prob_forecast_returns_to_investments(
                [forecast_return, uforecast_return, lforecast_return], "hinge"
            )
            hinge_rounded = (hinge_investment * 100).apply(np.trunc) / 100
            hinge_rounded.name = "Decision"
            hinge_rounded_ls = (hinge_investment * 1000).apply(np.trunc) / 1000
            hinge_rounded_ls.name = "Decision"
            hinge_investment_other_int = prob_forecast_returns_to_investments(
                [
                    forecast_return,
                    muforecast_return,
                    mlforecast_return,
                ],
                "hinge",
            )
            agree_investment = forecast_returns_to_investment_agreement(
                [forecast_return, uforecast_return, lforecast_return]
            )
            agree_investment_other_int = forecast_returns_to_investment_agreement(
                [forecast_return, muforecast_return, mlforecast_return]
            )
            investment_df['return_point_forecast'] = (investment * actual_returns).sum()
            # print(f"investment return: {(investment * actual_returns).sum()}")
            prob_x_act = prob_investment * actual_returns
            investment_df['return_prob_forecast'] = prob_x_act.sum()
            # print(f"prob investment return: {prob_x_act.sum()}")
            investment_df['return_prob_forecast_2int'] = (prob_investment_2int * actual_returns).sum()
            # print(f"prob investment return (2 pred int): {(prob_investment_2int * actual_returns).sum()}")
            investment_df['return_hinge_forecast'] = (hinge_investment * actual_returns).sum()
            # print(f"hinge investment return: {(hinge_investment * actual_returns).sum()}")
            investment_df['return_hinge_rounded'] = (hinge_rounded * actual_returns).sum()
            # print(f"hinge rnded investment return: {(hinge_rounded * actual_returns).sum()}")
            investment_df['return_hinge_rounded_less'] = (hinge_rounded_ls * actual_returns).sum()
            # print(f"hinge rnded investment return: {(hinge_rounded_ls * actual_returns).sum()}")
            investment_df['return_hinge_2int'] = (hinge_investment_other_int * actual_returns).sum()
            # print(f"hinge investment return (2nd pred int): {(hinge_investment_other_int * actual_returns).sum()}")
            investment_df['return_agreement'] = (agree_investment * actual_returns).sum()
            # print(f"agreement investment return: {(agree_investment * actual_returns).sum()}")
            investment_df['return_agreement_other_int'] = (agree_investment_other_int * actual_returns).sum()
            # print(f"agreement investment return (other interval): {(agree_investment_other_int * actual_returns).sum()}")
            investment_df['return_mrkt_basket'] = actual_returns.sum()
            # print(f"market basket return: {(actual_returns).sum()}")
            investment_df['greatest_loss_investment'] = (prob_x_act.sort_values().head(5).round(4) * 100).iloc[0]
            print(investment_df.iloc[0])
            print(f"greatest loss investment return for prob forecast: \n{prob_x_act.sort_values().head(5).round(4) * 100}")
            metric_df_list.append(metric_df)
            investment_df_list.append(investment_df)
            result_df_list.append(result)

    result_df = pd.concat(result_df_list)
    overall_result = pd.concat(
        [
            pd.concat(metric_df_list),
            pd.concat(investment_df_list),
            result_df,
        ],
        axis = 1
    )
    agg_result = overall_result.select_dtypes("number").groupby(level=0).mean()
    agg_result.to_csv("agg_m6_trial_results.csv")
