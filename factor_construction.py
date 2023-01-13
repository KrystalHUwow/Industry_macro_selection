from dataloading import *


def cal_implicit_factors(asset_name, n=4, method=None):
    """
    Parameters
    -----------
    asset_name: str
        EPS_TTM, PE_TTM
    n: int
    """
    X = get_fundamental_data(f'{asset_name}')
    rowindex = X.index
    colindex = X.columns
    X.dropna(axis=0,inplace=True)
    eigen_vals, eigen_vectors = np.linalg.eig(np.cov(X,rowvar=False))
    Variance_interpretation_ratio = eigen_vals.cumsum() / eigen_vals.sum()
    Factors = X @ eigen_vectors
    Factors.columns = [f"pca{i+1}" for i in range(len(X.columns))]
    return Factors.iloc[:,:n], Variance_interpretation_ratio

