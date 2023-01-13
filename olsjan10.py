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

def get_ols_result(y, x, weighted=False):
    X = pd.DataFrame(x)
    X = sm.add_constant(X)
    Y = pd.DataFrame(y)
    alldata = pd.concat([Y,X],axis=1)
    alldata.dropna(inplace=True, axis=0)
    pri_model = sm.OLS(alldata.iloc[:,0], alldata.iloc[:,1:])
    pri_result = pri_model.fit()
    if not weighted:
        return pri_result.tvalues[1]
    else:
        weights = 1 / np.abs(pri_result.resid)
        sec_model = sm.WLS(alldata.iloc[:,0], alldata.iloc[:,1:], weights)
        sec_result = sec_model.fit()
        return sec_result.tvalues[1]


def main_ols_varselect(**kwargs):
    """
    Parameters:
    s: 宏观因子的滞后阶数
    """
    label_names = kwargs['label_names']
    filename = kwargs['filename']
    sheet_names = kwargs['sheet_names']
    s = kwargs['s']
    weighted = kwargs['weighted']
    
    labels = get_fundamental_data(label_names)
    with pd.ExcelWriter(f'{filename}') as writer:
        for j in tqdm.tqdm(sheet_names):
            _ = get_macro_data(j)
            x = _.apply(lambda x:winsorize(x), axis=0)
            x = x.shift(s)
            res = pd.concat([x.apply(lambda x:get_ols_result(labels[col], x, weighted),axis=0) for col in labels.columns], axis=1) 
            res.columns = labels.columns
            res.index.name = 'industry'
            res.to_excel(writer, sheet_name=f'{j}')

def main_var_varselect(**kwargs):
    pass

def cal_ICseries():
    pass


if __name__ == '__main__':
    with pd.ExcelFile('./data/宏观数据.xlsx') as p:
        sheet_names = p.sheet_names
    
    # 更改params里的参数就行
    params = {
        'label_names': 'EPS_TTM',
        'filename': 'eps_result_varselect.xlsx',
        'sheet_names':  sheet_names,
        's': 0,
        'weighted': False,
    }
    main_varselect(**params)
    
    
            
            

