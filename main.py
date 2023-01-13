import pandas as pd
import numpy as np
from datetime import datetime
from statsmodels.api import OLS
import warnings
import tqdm
import statsmodels.api as sm

#---------------------------------------------------------------------------
from dataloading import *
from factor_selection import ols_select

            
if __name__ == '__main__':
    # 载入sheet_names 
    sheet_names = get_sheet_names()

    # 更改params里的参数就行
    params = {
        'label_names': 'EPS_TTM',
        'filename': 'eps_result_varselect.xlsx', # 这个写保存的文件名字, 结果去result里查看
        'sheet_names':  sheet_names,
        's': 0, # 滞后的阶段数
        'weighted': False,
    }
    ols_select(**params)
    
    
    