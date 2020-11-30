import json
from pathlib import Path
import re
from pprint import pp
import pandas as pd
import numpy as np

here = Path(__file__).parent

pattern = re.compile('\[\"(.*?)\"\]')  # noqa : W605

product_name_aliases = {
    'Globberin': ['Globbrin', 'Globberin', 'Globberin'],
    'Snafflelax': ['Snaffleflax', 'Snafulopromazide-b (Snaffleflax)'],
    'Vorbulon': ['Vorbulon', 'vorbulon.'],
    'Beeblizox': ['Beebliz%C3%B6x', 'Beeblizox']
}

def normalize_product_name(name):
    for normalized_name,aliases in product_name_aliases.items():
        if name.strip() in aliases:
            return normalized_name
    raise ValueError(f"no normalized name found for '{name}'")


def df_of_read_json_files():
    dir = here / "Data-Engineer-Case" / "data"
    df = pd.DataFrame()
    for file in dir.iterdir():
        if file.suffix != '.json':
            continue
        data = file.read_text()
        m = pattern.match(data)
        data = m.group(1)
        print(data)
        data = data.replace("\\\"", "\"")
        print(data)
        j = json.loads(data)
        df = df.append(pd.read_json(data))
    return df

def f():


    df_event = pd.read_csv(str(here/ "Data-Engineer-Case" /"data"/"crm_data.csv"))

    df = df_of_read_json_files()
    df.drop(labels=['acct_id', 'created_at'], axis=1, inplace=True)
    names = df.product_name.map(normalize_product_name)
    df.product_name = names
    #df.product_name = names.astype("category")
    #df.product_name.cat.set_categories(product_name_aliases.keys(), inplace=True)

    names = pd.unique(df.product_name)
    print(names)

    total_sales = df.unit_sales.sum()
    print(total_sales)

    #df = df.groupby(['product_name','date']).sum().apply(lambda x:x/total_sales)
    #df = df.rename(columns={"unit_sales":"share"})

    df = pd.pivot_table(df,index=['date'],columns=['product_name'],values=['unit_sales'],aggfunc=np.sum)
    df['total'] = 0
    for key in product_name_aliases.keys():
        df['total'] += df[('unit_sales'),key]
    for key in product_name_aliases.keys():
        df[('unit_sales',key)] = df[('unit_sales',key)] / df['total']
    df.to_csv(str(here / "out.csv"))

    return


if __name__ == "__main__":
    f()
