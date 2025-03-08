# -*- coding: utf-8 -*-
"""
The north remembers.
"""
import json
import datetime
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # required only for graphs
from autots import AutoTS, load_live_daily, create_regressor
from autots.evaluator.metrics import rps

csv_load = True
train_test = False
evolve = True
fred_key = "93873d40f10c20fe6f6e75b1ad0aed4d"  # https://fred.stlouisfed.org/docs/api/api_key.html
gsa_key = "zBYMGPb0bom4BEDsD8V0d3mwR6DnabT7Q9vkF1hz"  # https://open.gsa.gov/api/dap/
forecast_name = "m6_2"
graph = True  # whether to plot graphs
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
num_validations = (
    2  # number of cross validation runs. More is better but slower, usually
)
validation_method = "backwards"  # "similarity", "backwards", "seasonal 364"
n_jobs = "auto"  # or set to number of CPU cores
prediction_interval = 0.9  # 0.8 or 0.6 seem most useful
initial_training = "auto"  # set this to True on first run, or on reset, 'auto' looks for existing template, if found, sets to False.
archive_templates = True  # save a copy of the model template used with a timestamp
save_location = None  # "C:/Users/Colin/Downloads"  # directory to save templates to. Defaults to working dir
template_filename = f"autots_forecast_template_{forecast_name}.csv"
forecast_csv_name = None  # f"autots_forecast_{forecast_name}.csv"  # or None, point forecast only is written
transformer_list = "fast"  # 'superfast'
transformer_max_depth = 4
models_mode = "random"  # "deep", "regressor"
preclean = None
{  # preclean this or None
    "fillna": "ffill",  # "mean" or "median" are most consistent
    "transformations": {"0": "EWMAFilter"},
    "transformation_params": {
        "0": {"span": 2},
    },
}
submission_month = "2301"
model_list = "fast_parallel"
constraint = None
{"constraint_method": "stdev", "constraint_regularization": 1, "upper_constraint": 2, "lower_constraint": 2, "bounds": False}
metric_weighting = {
    'smape_weighting': 1,
    'mae_weighting': 3,
    'rmse_weighting': 3,
    'made_weighting': 0,
    'mage_weighting': 0,
    'mle_weighting': 1,
    'imle_weighting': 0,
    'spl_weighting': 5,
    'containment_weighting': 0,
    'contour_weighting': 0,
    'runtime_weighting': 0.05,
    'oda_weighting': 0,
    'dwae_weighting': 1,
}

if save_location is not None:
    template_filename = os.path.join(save_location, template_filename)
    if forecast_csv_name is not None:
        forecast_csv_name = os.path.join(save_location, forecast_csv_name)

if initial_training == "auto":
    initial_training = not os.path.exists(template_filename)
    if initial_training:
        print("No existing template found.")
    else:
        print("Existing template found.")

# set max generations based on settings, increase for slower but greater chance of highest accuracy
if initial_training:
    gens = 10
    models_to_validate = 0.35
    ensemble = None  # ['horizontal-max']
elif evolve:
    gens = 100
    models_to_validate = 0.15
    # you can include "simple" and "distance" but they can nest, and may get huge as time goes on...
    ensemble = None  # ['horizontal-max']
else:
    gens = 0
    models_to_validate = 0.35
    ensemble = ['horizontal-max']

# only save the very best model if not evolve
if evolve:
    n_export = 30
else:
    n_export = 1  # wouldn't be a bad idea to do > 1, allowing some future adaptability

"""
Begin dataset retrieval
"""

if csv_load:
    df = pd.read_csv(f"M6_month{submission_month}.csv", index_col=0)
    df.index = pd.DatetimeIndex(df.index)
    print("csv data loaded")
else:
    short_list = ["AMZN", "PG", "AVB", "EWY", "IEF"]
    long_list = [
        'ABBV', 'ACN', 'AEP', 'AIZ', 'ALLE', 'AMAT', 'AMP', 'AMZN', 'AVB', 'AVY',
        'AXP', 'BDX', 'BF-B', 'BMY', 'BR', 'CARR', 'CDW', 'CE', 'CHTR', 'CNC', 'CNP',
        'COP', 'CTAS', 'CZR', 'DG', 'DPZ', 'DRE', 'DXC', 'META', 'FTV', 'GOOG', 'GPC',
        'HIG', 'HST', 'JPM', 'KR', 'OGN', 'PG', 'PPL', 'PRU', 'PYPL', 'RE', 'ROL',
        'ROST', 'UNH', 'URI', 'V', 'VRSK', 'WRK', 'XOM', 'IVV', 'IWM', 'EWU', 'EWG',
        'EWL', 'EWQ', 'IEUS', 'EWJ', 'EWT', 'MCHI', 'INDA', 'EWY', 'EWA', 'EWH', 'EWZ',
        'EWC', 'IEMG', 'LQD', 'HYG', 'SHY', 'IEF', 'TLT', 'SEGA.L', 'IEAA.L', 'HIGH.L',
        'JPEA.L', 'IAU', 'SLV', 'GSG', 'REET', 'ICLN', 'IXN', 'IGF', 'IUVL.L', 'IUMO.L',
        'SPMV.L', 'IEVL.L', 'IEFM.L', 'MVEU.L', 'XLK', 'XLF', 'XLV', 'XLE', 'XLY', 'XLI',
        'XLC', 'XLU', 'XLP', 'XLB', 'VXX',
    ]
    df = load_live_daily(
        long=False,
        fred_key=fred_key,
        fred_series=None,  # ["DGS10", "T5YIE", "SP500", "DCOILWTICO", "DEXUSEU", "WPU0911", "DEXUSUK"]
        tickers=long_list,
        trends_list=None,  # ["forecasting", "msft", "p&g"]
        earthquake_min_magnitude=None,
        weather_stations=None,
        weather_years=3,
        london_air_stations=None,
        london_air_days=700,
        gsa_key=gsa_key,
        gov_domain_list=None,  # ['usajobs.gov', 'usps.com', 'weather.gov']
        gov_domain_limit=1000,
        weather_event_types=None,
        sleep_seconds=20,
    )
    df_backup = df.copy()
    df.to_csv(f"M6_month{submission_month}.csv")
# sample to full daily data
df = df.resample("D").fillna(method="ffill")
# remove data from before most recent Friday
df = df[df.index <= df.index[df.index.dayofweek.isin([4])].max()]
# keep only the closing price
df = df[[x for x in df.columns if "_close" in x]]
df.columns = [x.replace("_close", "").upper() for x in df.columns]
# resample to weekly Friday
# df = df.resample("W-FRI").last()
# make divisible by 4 weeks and resample to 4 weeks
df = df.iloc[df.shape[0] % 21:].resample("4W-FRI", label="right").last()
start_time = datetime.datetime.now()
while df.index[-1] > start_time:
    df.index = df.index.shift(-1, "W-FRI")


# shifted_df = df.shift(4)
# return_df = (df - shifted_df) / shifted_df

regr_df = load_live_daily(
    long=False,
    fred_key=fred_key,
    fred_series=["DGS10", "T5YIE", "DCOILWTICO", "DEXUSEU", "DEXUSUK"],
    tickers=["^IXIC"],
    trends_list=["p&g", "AMZN", "China", "house prices", "federal reserve"],
    earthquake_min_magnitude=None,
    weather_stations=["USW00014732", "UKM00003772"],
    weather_data_types=["PRCP", "TAVG"],
    weather_years=5,
    london_air_stations=["CT3"],
    london_air_days=1000,
    gsa_key=gsa_key,
    gov_domain_list=None,  # ["usajobs.gov", "usps.com"]
    gov_domain_limit=1000,
    weather_event_types=None,
    timeout=900,
    sleep_seconds=20,
)
regr_df_weekly = regr_df.resample("D").fillna(method="ffill").resample("W-FRI").last()
regr_df_weekly_backup = regr_df_weekly.copy()

df = df[df.index.year > 1999]
# df = df[df.index.year < 2022]
start_time = datetime.datetime.now()
# remove any data from the future
# df = df[df.index <= start_time]
# regr_df_weekly = regr_df_weekly[regr_df_weekly.index <= start_time]
# remove series with no recent data
regr_df_weekly = regr_df_weekly.dropna(axis="columns", how="all")
min_cutoff_date = start_time - datetime.timedelta(days=21)
most_recent_date = regr_df_weekly.notna()[::-1].idxmax()
drop_cols = most_recent_date[most_recent_date < min_cutoff_date].index.tolist()
regr_df_weekly = regr_df_weekly.drop(columns=drop_cols)

if train_test:
    df_test = df[df.shape[0] - forecast_length:]
    df = df[:-forecast_length]
    regr_df_weekly = regr_df_weekly.reindex(df.index)
else:
    regr_df_weekly = regr_df_weekly.reindex(df.index)

# example regressor with some things we can glean from data and datetime index
# note this only accepts `wide` style input dataframes
regr_train, regr_fcst = create_regressor(
    regr_df_weekly,
    forecast_length=forecast_length,
    frequency=frequency,
    drop_most_recent=drop_most_recent,
    scale=True,
    summarize="auto",
    backfill="bfill",
    fill_na="spline",
    holiday_countries=["US"],  # requires holidays package
    datepart_method="simple_2",
)

# remove the first forecast_length rows (because those are lost in regressor)
df = df.iloc[forecast_length:]
regr_train = regr_train.iloc[forecast_length:]

print("data setup completed, beginning modeling")
"""
Begin modeling
"""

model = AutoTS(
    forecast_length=forecast_length,
    frequency=frequency,
    prediction_interval=prediction_interval,
    ensemble=ensemble,
    model_list=model_list,
    transformer_list=transformer_list,
    transformer_max_depth=transformer_max_depth,
    max_generations=gens,
    metric_weighting=metric_weighting,
    initial_template="random",
    aggfunc="sum",
    models_to_validate=models_to_validate,
    model_interrupt=True,
    num_validations=num_validations,
    validation_method=validation_method,
    drop_most_recent=drop_most_recent,  # if newest data is incomplete, also remember to increase forecast_length
    preclean=preclean,
    models_mode=models_mode,
    constraint=constraint,
    # no_negatives=True,
    # subset=100,
    # prefill_na=0,
    # remove_leading_zeroes=True,
    # current_model_file=f"current_model_{forecast_name}",
    introduce_na=False,
    n_jobs=n_jobs,
    verbose=2,
)
model.similarity_validation_params = {
    "stride_size": 1,
    "distance_metric": "nan_euclidean",
    "include_differenced": True,
    "window_size": 30,
}


if not initial_training:
    if evolve:
        model.import_template(template_filename, method="addon")
    else:
        model.import_template(template_filename, method="only")

model = model.fit(
    df,
    future_regressor=regr_train,
)

interval_list = [prediction_interval, 0.6]
prediction = model.predict(
    future_regressor=regr_fcst,
    prediction_interval=interval_list,
    verbose=2,
    fail_on_forecast_nan=True,
)

# Print the details of the best model
print(model)

"""
Process results
"""


def returns_to_obs_matrix(returns, q=5):
    quintiles = pd.qcut(returns, q=q, labels=range(1, q + 1))
    observed = pd.get_dummies(quintiles)
    observed.reindex(columns=sorted(observed.columns), fill_value=0)
    return observed


def forecast_returns_to_investments(fore):
    total_returns = fore.abs().sum()
    return fore / total_returns


def prob_forecast_returns_to_investments(fore_list, agg_func="mean"):
    """Create investment decisions.

    Args:
        fore_list (list): list of forecast returns in pd.Series, point forecast as first one in list
    """
    all_fore = pd.concat(fore_list, axis=1)
    if agg_func == "hinge":
        return forecast_returns_to_investments(all_fore.iloc[:, 1:].mean(axis=1))
    else:
        return forecast_returns_to_investments(all_fore.mean(axis=1))


def forecast_returns_to_investment_agreement(fore_list):
    all_fore = pd.concat(fore_list, axis=1)
    agreement = all_fore[(all_fore >= 0).all(axis=1) | (all_fore <= 0).all(axis=1)]
    if agreement.empty:
        print("agreement investment FAILED")
        return prob_forecast_returns_to_investments(fore_list)
    else:
        return forecast_returns_to_investments(agreement.median(axis=1))


# point forecasts dataframe
forecasts_df = prediction[str(interval_list[0])].forecast  # .fillna(0).round(0)
if forecast_csv_name is not None:
    forecasts_df.to_csv(forecast_csv_name)
forecasts_upper_df = prediction[str(interval_list[0])].upper_forecast
forecasts_lower_df = prediction[str(interval_list[0])].lower_forecast
forecasts_mid_upper_df = prediction[str(interval_list[1])].upper_forecast
forecasts_mid_lower_df = prediction[str(interval_list[1])].lower_forecast
forecast_return = (forecasts_df.iloc[-1] - df.iloc[-1]) / df.iloc[-1]
uforecast_return = (forecasts_upper_df.iloc[-1] - df.iloc[-1]) / df.iloc[-1]
lforecast_return = (forecasts_lower_df.iloc[-1] - df.iloc[-1]) / df.iloc[-1]
muforecast_return = (forecasts_mid_upper_df.iloc[-1] - df.iloc[-1]) / df.iloc[-1]
mlforecast_return = (forecasts_mid_lower_df.iloc[-1] - df.iloc[-1]) / df.iloc[-1]

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
naive_prediction2 = pd.DataFrame(
    np.tile(np.array([0, 0.25, 0.5, 0.25, 0]).reshape(1, -1), (df.shape[1], 1)),
    index=ranked_prediction.index,
    columns=ranked_prediction.columns,
)
naive_prediction3 = pd.DataFrame(
    np.tile(np.array([0, 0.1, 0.8, 0.1, 0]).reshape(1, -1), (df.shape[1], 1)),
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
weighted_ranked_naive2 = (
    forecasts_obs * 3 + u_forecasts_obs + l_forecasts_obs
    + mu_forecasts_obs + ml_forecasts_obs + naive_prediction * 15
) / 22
run = 0
if train_test:
    print(prediction[str(interval_list[0])].evaluate(df_test).avg_metrics)
    metric_df = pd.DataFrame(index=[run])
    investment_df = pd.DataFrame(index=[run])
    actual_returns = (df_test.iloc[-1] - df.iloc[-1]) / df.iloc[-1]
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
    rps_naive2 = rps(naive_prediction2, actuals)
    metric_df['RPS_naive2'] = rps_naive2.sum()
    # print(f"RPS of naive2: {rps_naive2.sum()}")
    rps_naive3 = rps(naive_prediction3, actuals)
    metric_df['RPS_naive3'] = rps_naive3.sum()
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
    investment_df['return_mrkt_basket'] = actual_returns.mean()
    # print(f"market basket return: {(actual_returns).sum()}")
    investment_df['greatest_loss_investment'] = (prob_x_act.sort_values().head(5).round(4) * 100).iloc[0]
    print(investment_df.iloc[0])
    print(f"greatest loss investment return for prob forecast: \n{prob_x_act.sort_values().head(5).round(4) * 100}")

# accuracy of all tried model results
model_results = model.results()
validation_results = model.results("validation")

# save a template of best models
if initial_training or evolve:
    model.export_template(
        template_filename, models="best", n=n_export, max_per_model_class=5
    )
    if archive_templates:
        arc_file = f"{template_filename.split('.csv')[0]}_{start_time.strftime('%Y%m%d%H%M')}.csv"
        model.export_template(arc_file, models="best", n=1)

print(f"Model failure rate is {model.failure_rate() * 100:.1f}%")
print(f"The following model types failed completely {model.list_failed_model_types()}")
print("Slowest models:")
try:
    model_results['TotalRuntime'] = model_results['TotalRuntime'].dt.seconds
except Exception:
    pass
print(
    model_results[model_results["Ensemble"] < 1]
    .groupby("Model")
    .agg({"TotalRuntime": ["mean", "max"]})
    .idxmax()
)

hinge_investment = prob_forecast_returns_to_investments(
    [forecast_return, uforecast_return, lforecast_return], "hinge"
)
hinge_rounded = (hinge_investment * 100).apply(np.trunc) / 100
hinge_rounded.name = "Decision"
hinge_rounded_ls = (hinge_investment * 1000).apply(np.trunc) / 1000
hinge_rounded_ls.name = "Decision"
result = pd.concat([weighted_ranked_naive2, hinge_rounded_ls], axis=1)
result.columns = ["Rank1", "Rank2", "Rank3", "Rank4", "Rank5", "Decision"]
result.index.name = "ID"
result = (np.trunc(result * 10000) / 10000)
result["Rank3"] = (result['Rank3'] + (1 - result.iloc[:, 0:5].sum(axis=1))).round(5)
result.to_csv(f"m6_submission{submission_month}.csv")
model_parameters = json.loads(model.best_model["ModelParameters"].iloc[0])

if graph:
    column_indices = [0, 1, 2]  # change column here
    for plt_col in column_indices:
        col = model.df_wide_numeric.columns[plt_col]
        plot_df = pd.DataFrame(
            {
                col: model.df_wide_numeric[col],
                "up_forecast": forecasts_upper_df[col],
                "low_forecast": forecasts_lower_df[col],
                "forecast": forecasts_df[col],
            }
        )
        if train_test:
            plot_df = plot_df.merge(df_test[col].rename("actual"), right_index=True, left_index=True, how="left")
        plot_df[plot_df == 0] = np.nan
        plot_df[plot_df < 0] = np.nan
        plot_df[plot_df > 100000] = np.nan
        plot_df[col] = plot_df[col].interpolate(method="linear", limit_direction="backward")
        fig, ax = plt.subplots(dpi=300, figsize=(8, 6))
        plot_df[plot_df.index.year >= 2021].plot(ax=ax, kind="line")
        # plt.savefig("model.png", dpi=300)
        plt.show()

    model.plot_generation_loss()
    plt.show()

    model.plot_per_series_smape()
    plt.show()

    if model.best_model["Ensemble"].iloc[0] == 2:
        plt.subplots_adjust(bottom=0.5)
        model.plot_horizontal_transformers()
        plt.savefig("transformers_0819.png", dpi=300)
        plt.show()

        if series.shape[0] > 25:
            series = series.sample(25, replace=False)
        series[["log(Volatility)", "log(Mean)"]] = np.log(
            series[["Volatility", "Mean"]]
        )

        fig, ax = plt.subplots(figsize=(6, 4.5))
        cmap = plt.get_cmap("tab10")  # 'Pastel1, 'cividis', 'coolwarm', 'spectral'
        names = series["Model"].unique()
        colors = dict(zip(names, cmap(np.linspace(0, 1, len(names)))))
        grouped = series.groupby("Model")
        for key, group in grouped:
            group.plot(
                ax=ax,
                kind="scatter",
                x="log(Mean)",
                y="log(Volatility)",
                label=key,
                color=colors[key].reshape(1, -1),
            )
        plt.title("Horizontal Ensemble: models choosen by series")
        plt.show()
        # plt.savefig("horizontal.png", dpi=300)

        if str(model_parameters["model_name"]).lower() in ["mosaic", "mosaic-window"]:
            mosaic_df = model.mosaic_to_df()
            print(mosaic_df[mosaic_df.columns[0:5]].head(5))
            print(mosaic_df.iloc[-1])

        # plot the SMAPE error sources of point forecast vs test
        if train_test:
            ((forecasts_df - df_test).abs().mean() * 100 / df_test.mean()).round(
                2
            ).sort_values(ascending=False).head(10).plot(
                kind="bar", title="highest MAPE series on test", color="#ff6666"
            )
