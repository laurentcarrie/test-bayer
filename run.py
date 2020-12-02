import json
from pathlib import Path
import re
from pprint import pp
import pandas as pd # type: ignore
import numpy as np # type: ignore
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



def normalize_product_name(name:str) -> str :
    for normalized_name,aliases in product_name_aliases.items():
        if name.strip() in aliases:
            return normalized_name
    raise ValueError(f"no normalized name found for '{name}'")


def df_of_read_json_files() -> pd.DataFrame :
    """
    reads the json files and return one dataframe

    note that the json files are not json files, so they need to be fixed.
    this is not done because we assume these are input files which are read only

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
        j = json.loads(data)
        df = df.append(pd.read_json(data))
    return df




@expect_columns(input=['acct_id','product_name','date','unit_sales','created_at'],
                output=['date','all_product_unit_sales'])
def df_sum_of_sales(df:pd.DataFrame) -> pd.DataFrame:
    '''
    :param df:
    the original sales dataframe from json
    :return:
    a dataframe with
        index : date (month)
        cols : all_product_sales : the sum of of all sales for this month
    '''


    df = df.drop(labels=['acct_id', 'created_at'], axis=1)

    df = pd.pivot_table(df, index=['date'], columns=['product_name'], values=['unit_sales'], aggfunc=np.sum)
    df['total'] = 0
    for key in product_name_aliases.keys():
        df['total'] += df[('unit_sales'), key]
    df.columns = list(map("_".join, df.columns))
    df = df.rename(columns={"total_": "all_product_unit_sales"})

    for key in product_name_aliases.keys():
        df = df.drop(f"unit_sales_{key}", axis=1)

    return df



#@expect_columns(input=None,output=['acct_id','date','market_share'])
def build_df_market_share() -> pd.DataFrame:

    # read all json files.
    # if the set of data was really big, we would use parquet format,
    # and read only once the json for past months, and reload the current month
    df = df_of_read_json_files()

    # we fix the names, by trying to replace known typos
    # this will raise an exception if a name cannot be fixed
    # the product names are categorized, the values are product_name_aliases.keys()
    df.product_name = df.product_name.map(normalize_product_name)

    # for debug and discuss
    df.to_csv("brut.csv")


    sum_per_month = df_sum_of_sales(df)

    # for debugging and discussing the exercise
    sum_per_month.to_csv(str(here / "sums.csv"))

    # in the sales data, we are only interested in SNAFFLELAX sales
    df = df[df.product_name == SNAFFLELAX]

    df.date = df.date.map(np.datetime64)

    df = df[df.acct_id == '933bc0ad-9764-4035-abf8-f8fb3ebeeaff']
    df = df.drop(labels=['acct_id','product_name','created_at'],axis=1)

    df = pd.merge(df,sum_per_month,on='date',how='outer').fillna(0)
    df = df.set_index(df.date)
    df = df.sort_index()

    #df['xmonth'] = df.unit_sales.rolling(window=3,min_periods=1).mean().fillna(0)
    df['x_mean_unit_sales'] = df.unit_sales.rolling(window=4).mean().fillna(0)
    df.to_csv('toto.csv')

    # we add a column : for this month, the total of sales
    #
    #df['all_sums'] = df['date'].apply(lambda x : sum_per_month['all_product_unit_sales'][x]).fillna(0)

    # to get the market share
    df['market_share'] = df['unit_sales'] / df['all_product_unit_sales']
    df['x_mean_market_share'] = df['x_mean_unit_sales'] / df['all_product_unit_sales']
    df.to_csv('toto.csv')

    # clean up
    #df = df.drop(labels=['product_name','created_at','unit_sales','all_sums'],axis=1)
    #df = df.drop(labels=['product_name','created_at','all_sums'],axis=1)

    return df


@expect_columns(input=None,output=['acct_id','date','nb_event'])
def build_df_crm() -> pd.DataFrame :
    df= pd.read_csv(str(here/ "Data-Engineer-Case" /"data"/"crm_data.csv"))

    # convert string to date
    df.date = df.date.map(lambda s : datetime.datetime.strptime(s,'%Y-%m-%d'))

    # we project the date on the first day of the month
    df.date = df.date.map(lambda d:np.datetime64(datetime.date(year=d.year,month=d.month,day=1)))

    # we dont't care about the type of event : we replace it by 1, so we can count
    df.event_type = 1

    df = df.groupby(['acct_id','date']).sum()
    df = df.reset_index()
    df = df.rename(columns={'event_type':'nb_event'})

    return df

if __name__ == "__main__":
    try:
        df_ms = build_df_market_share()
        df_ms.to_csv('market-share.csv')

        df_crm = build_df_crm()
        df_crm.to_csv('crm.csv')

        #df_join = pd.merge(df_ms,df_crm,on=["acct_id","date"])
        #df_join.to_csv('join.csv')
    except Exception :
        traceback.print_exc()
