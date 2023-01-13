import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.api import OLS
import warnings
import tqdm
import statsmodels.api as sm


def winsorize(series, p=0.05):
    up = np.quantile(series, 1-p/2)
    down = np.quantile(series,p/2)
    series_winsorized = np.clip(series, down, up)
    return series_winsorized

def standardize():
    pass

def neutralize():
    pass

def get_sheet_names():
    with pd.ExcelFile('./data/宏观数据.xlsx') as p:
        sheet_names = p.sheet_names
    return sheet_names

def get_macro_data(sheet_name):
    #get macro data
    date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")
    macro_data = pd.read_excel('data/宏观数据.xlsx', sheet_name=sheet_name, skipfooter=2, index_col=0,
                                date_parser=date_parser)
    macro_data.index.name = None
    if macro_data.index[1].day - macro_data.index[0].day == 1 or macro_data.index[1].month - macro_data.index[
        0].month == 1:
        macro_data = macro_data.resample('m').mean()
    else:
        macro_data = macro_data.resample('3m').mean()
    macro_data.fillna(method='ffill', inplace=True)
    return macro_data

def get_fundamental_data(name):
    """
    name : EPS_TTM, PE_TTM
    """
    date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")

    data = pd.read_excel(f'./data/{name}.xlsx', skiprows=[0, 1, 3], index_col=0, date_parser=date_parser).iloc[:,
               :31]
    for column in data.columns:
        data.rename(columns={column: column.split('.')[0]}, inplace=True)
    data.fillna(method='ffill', inplace=True)
    data.index.name = None
    data = (data - data.shift(12)) / data.shift(12)
    data = data[data.index >= datetime(2000, 1, 1)]
    return data
# --------------------------------------------------------------------------------

def cal_ICseries():
    pass


