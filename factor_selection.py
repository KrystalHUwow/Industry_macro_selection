from dataloading import *

from statsmodels.tsa.stattools import grangercausalitytests, adfuller
from statsmodels.tsa.api import VAR
from datetime import datetime
from statsmodels.api import OLS
import statsmodels.api as sm


# 利用OLS进行变量删选
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

def ols_select(**kwargs):
    """利用OLS对宏观变量进行选择
    
    Parameters:
    --------------------------------
    s: 宏观因子的滞后阶数
    
    
    Returns:
    -------------------------------
    tvalue of beta
    """
    label_names = kwargs['label_names']
    filename = kwargs['filename']
    sheet_names = kwargs['sheet_names']
    s = kwargs['s']
    weighted = kwargs['weighted']
    
    labels = get_fundamental_data(label_names)
    with pd.ExcelWriter(f'./result/{filename}') as writer:
        for j in tqdm.tqdm(sheet_names):
            _ = get_macro_data(j) 
            x = _.apply(lambda x:winsorize(x), axis=0) # 缩尾处理, 在现阶段standarize没有意义
            x = x.shift(s)
            res = pd.concat([x.apply(lambda x:get_ols_result(labels[col], x, weighted),axis=0) for col in labels.columns], axis=1) 
            res.columns = labels.columns
            res.index.name = 'industry'
            res.to_excel(writer, sheet_name=f'{j}')

# 在进行VAR前, 先定义一些数据处理函数

def stationaize_series(series: pd.Series):
    p_value = adfuller(series)[1]
    d = 0
    while p_value > 0.05:
        d += 1
        if d > 2: break
        series = series.diff().finall(series.mean()) # 用均值填补数据, 保持两个序列的支撑相同
        p_value = adfuller(series)[1]
    if d > 2:
        return None
    else:
        return series, d


def var_select(**kwargs):
    pass