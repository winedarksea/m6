import datetime
import pandas as pd
from autots import load_live_daily

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


def load_data(csv_load, train_test, forecast_length, fred_key=None, gsa_key=None, period=0):
    if csv_load:
        df = pd.read_csv(f"M6_month{period}.csv", index_col=0)
        df.index = pd.DatetimeIndex(df.index)
        print("csv data loaded")
    else:
        # short_list = ["AMZN", "PG", "AVB", "EWY", "IEF"]
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
        df.to_csv(f"M6_month{period}.csv")
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

    df = df[df.index.year > 1999]
    # df = df[df.index.year < 2022]
    start_time = datetime.datetime.now()
    # remove any data from the future
    df = df[df.index <= start_time]
    regr_df_weekly = regr_df_weekly[regr_df_weekly.index <= start_time]
    # remove series with no recent data
    regr_df_weekly = regr_df_weekly.dropna(axis="columns", how="all")
    min_cutoff_date = start_time - datetime.timedelta(days=21)
    most_recent_date = regr_df_weekly.notna()[::-1].idxmax()
    drop_cols = most_recent_date[most_recent_date < min_cutoff_date].index.tolist()
    regr_df_weekly = regr_df_weekly.drop(columns=drop_cols)

    return df, regr_df_weekly
