from pathlib import Path
import re
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import datetime
import traceback
from mychecks import expect_columns


here = Path(__file__).parent

SNAFFLELAX = 'Snafflelax'

product_name_aliases = {
    'Globberin': ['Globbrin', 'Globberin', 'Globberin'],
    SNAFFLELAX: ['Snaffleflax', 'Snafulopromazide-b (Snaffleflax)'],
    'Vorbulon': ['Vorbulon', 'vorbulon.'],
    'Beeblizox': ['Beebliz%C3%B6x', 'Beeblizox']
}


def normalize_product_name(name: str) -> str:
    for normalized_name, aliases in product_name_aliases.items():
        if name.strip() in aliases:
            return normalized_name
    raise ValueError(f"no normalized name found for '{name}'")


def df_of_read_json_files() -> pd.DataFrame:
    """
    reads the json files and return one dataframe

    note that the json files are not json files, so they need to be fixed.
    this is not done because we assume these are input files
    which are read only

    :return:
    """
    dir = here / "Data-Engineer-Case" / "data"
    df = pd.DataFrame()
    pattern = re.compile('\[\"(.*?)\"\]')  # noqa : W605
    for file in dir.iterdir():
        if file.suffix != '.json':
            continue
        data = file.read_text()
        m = pattern.match(data)
        if m is None:
            raise ValueError(f"file {file} has not the expected format")
        data = m.group(1)
        data = data.replace("\\\"", "\"")
        df = df.append(pd.read_json(data))

    # we fix the names, by trying to replace known typos
    # this will raise an exception if a name cannot be fixed
    # the product names are categorized, the values are
    # product_name_aliases.keys()
    df.product_name = df.product_name.map(normalize_product_name)

    return df


@expect_columns(input=['acct_id', 'product_name', 'date',
                       'unit_sales', 'created_at'],
                output=['date', 'all_product_unit_sales'])
def df_sum_of_sales(df: pd.DataFrame) -> pd.DataFrame:
    '''
    :param df:
    the original sales dataframe from json
    :return:
    a dataframe with
        index : date (month)
        cols : all_product_sales :
               the sum of of all sales for this month
    '''

    df = df.drop(labels=['acct_id', 'created_at'], axis=1)

    df = pd.pivot_table(df, index=['date'], columns=['product_name'], values=[
                        'unit_sales'], aggfunc=np.sum)
    df['total'] = 0
    for key in product_name_aliases.keys():
        df['total'] += df[('unit_sales'), key]
    df.columns = list(map("_".join, df.columns))
    df = df.rename(columns={"total_": "all_product_unit_sales"})

    for key in product_name_aliases.keys():
        df = df.drop(f"unit_sales_{key}", axis=1)

    return df


@expect_columns(
    input=['acct_id', 'product_name', 'date', 'unit_sales', 'created_at'],
    output=['acct_id', 'date', 'market_share', "lagged_unit_sales",
            "market_share", "lagged_market_share"])
def build_df_market_share(df, ms_average) -> pd.DataFrame:
    '''

    :param df: the original dataset
    :param ms_average: the number of month  for the X-lagged mean of rate
    :return:
    '''

    sum_per_month = df_sum_of_sales(df)

    # for debugging and discussing the exercise
    sum_per_month.to_csv(str(here / "sums.csv"))

    # in the sales data, we are only interested in SNAFFLELAX sales
    df = df[df.product_name == SNAFFLELAX]

    df = df.drop(labels=['product_name', 'created_at'], axis=1)

    df.date = df.date.map(np.datetime64)

    acct_ids = df.acct_id.unique()

    @expect_columns(input=["date", "acct_id", "unit_sales"],
                    output=["date", "acct_id", "lagged_unit_sales",
                            "market_share", "lagged_market_share"])
    def get_df_for_acct_id(df, acct_id):
        '''
        calculates the means for a given client
        :param df: the original data set
        :param acct_id: the client
        :return:
        a dataframe with market share and lagged_market_sahre
        '''
        df = df[df.acct_id == acct_id]
        # merge with the sums, this will populate the missing months
        df = pd.merge(df, sum_per_month, on='date', how='right').fillna(0)
        df = df.set_index(df.date)
        df = df.sort_index()
        df = df.resample(rule='M', on='date').sum()

        df['market_share'] = df.unit_sales / df.all_product_unit_sales

        # x-month average of market share
        df['lagged_unit_sales'] = df.unit_sales.rolling(
            window=ms_average).mean()
        df['lagged_market_share'] = df['lagged_unit_sales'] / \
            df.all_product_unit_sales

        df = df.reset_index()
        # the windowing labels with the right value of the window,
        # we like the first day of the month better
        df['date'] = df['date'].apply(lambda x: x.replace(day=1))
        # add the client id because we will concatenate all results
        df['acct_id'] = acct_id
        return df

    new_df = pd.DataFrame()
    for acct_id in acct_ids:
        new_df = pd.concat([new_df, get_df_for_acct_id(df, acct_id)])

    new_df.reset_index()

    # for debug and discuss
    new_df.to_csv('newdf.csv')

    # clean up
    # df = df.drop(labels=['product_name','created_at','unit_sales','all_sums'],axis=1)
    # df = df.drop(labels=['product_name','created_at','all_sums'],axis=1)

    return new_df


@expect_columns(input=None, output=['acct_id', 'date', 'sum_events',
                                    'lagged_sum_events', 'lagged_weighted_sum_events'])
def build_df_crm(events_average, events_weights) -> pd.DataFrame:
    '''

    :param events_average: the size of the window of lagged_sum_events -> lagged_sum_events
    :param events_weights: the vector of weights for sum_of_events -> lagged_weighted_sum_events
    :return:
    '''
    df = pd.read_csv(str(here / "Data-Engineer-Case" / "data" / "crm_data.csv"))

    # convert string to date
    df.date = df.date.map(lambda s: datetime.datetime.strptime(s, '%Y-%m-%d'))

    # we project the date on the first day of the month
    df.date = df.date.map(lambda d: np.datetime64(
        datetime.date(year=d.year, month=d.month, day=1)))

    # we dont't care about the type of event : we replace it by 1, so we can count
    df.event_type = 1

    df = df.groupby(['acct_id', 'date']).sum()
    df = df.reset_index()
    df = df.rename(columns={'event_type': 'sum_events'})

    @expect_columns(input=['date','acct_id','sum_events'],
                    output=['date','acct_id','sum_events','lagged_sum_events','lagged_weighted_sum_events'])
    def get_df_for_acct_id(df, acct_id):
        df = df[df.acct_id == acct_id]
        df = df.set_index(df.date)
        df = df.sort_index()
        df = df.resample(rule='M', on='date').sum()

        # events_average is the size of the window in monthgs
        df['lagged_sum_events'] = df.sum_events.rolling(window=events_average).mean()

        df['lagged_weighted_sum_events'] = df.sum_events.rolling(window=len(events_weights)).apply(
            func=lambda x: np.dot(x, events_weights))

        df = df.reset_index()
        # the windowing labels with the right value of the window,
        # we like the first day of the month better
        df['date'] = df['date'].apply(lambda x: x.replace(day=1))

        # add the client id because we will concatenate all results
        df['acct_id'] = acct_id

        return df

    acct_ids = df.acct_id.unique()

    new_df = pd.DataFrame()
    for acct_id in acct_ids:
        new_df = pd.concat([new_df, get_df_for_acct_id(df, acct_id)])

    return new_df


def main(ms_average, events_average, events_weight):
    try:
        # read all json files.
        # if the set of data was really big, we would use parquet format,
        # and read only once the json for past months, and reload the current month
        df_raw = df_of_read_json_files()

        # for debugging and discussing
        df_raw.to_csv("raw.csv")

        # ms average is the X-month average of market share, as per question 1.a and 1.b
        df_ms = build_df_market_share(df_raw, ms_average)

        # for debugging and discussing
        df_ms.to_csv('market-share.csv')

        # the crm data with the sum of events, lagged and weighted as per question 1.c and 1.d
        df_crm = build_df_crm(events_average, events_weights)

        # for debugging and discussing
        df_crm.to_csv('crm.csv')

        # merge with the sales dataframe
        # this data set has columns that can be used by the data scientist,
        # to correlate crm action (sum of events) and sales
        df_join = pd.merge(df_ms, df_crm, on=["acct_id", "date"])
        df_join.to_csv('join.csv')
    except Exception:
        traceback.print_exc()


if __name__ == "__main__":
    ms_average = 3
    events_average = 3
    events_weights = [.2, .2, .6]
    main(ms_average, events_average, events_weights)
