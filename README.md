# 

## 可能存在的问题

1. 宏观因子的winsorize应该是从时序角度
2. ICIR部分, 学姐是把宏观因子的因子暴露作为新的"因子吗" ? 如果是这样的话，OLS结果中应该选择第二个参数
3. RanK_IC 其实是Spearman_rho相关系数，可以直接一行代码就搞定了
4. 为什么因子暴露的ICIR是需要和eps作相关系数呢

## 补充代码问题
1. dataloading--get_fundamental_data中data.rename(columns={column: column.split('.')[0]}, inplace=True)，如果是要把“(申万)"删掉，应为column.split('(')
