import numpy as np
from scipy.stats import pearsonr, spearmanr
from lifelines.utils import concordance_index
from sklearn.metrics import average_precision_score
# =========================
# basic metrics
# =========================
def get_mse(y, f):
    y = np.asarray(y)
    f = np.asarray(f)
    return np.mean((y - f) ** 2)

def get_rmse(y, f):
    return np.sqrt(get_mse(y, f))

def get_pearson(y, f):
    return pearsonr(y, f)[0]

def get_spearman(y, f):
    return spearmanr(y, f).correlation

def get_aupr(Y, P, threshold=7.0):
    Y = (np.asarray(Y) >= threshold).astype(int)
    P = (np.asarray(P) >= threshold).astype(int)
    return average_precision_score(Y, P)


def get_ci(y, f):
    y = np.asarray(y)
    f = np.asarray(f)
    return concordance_index(y, f)

def r_squared_error(y_obs, y_pred):

    y_obs = np.asarray(y_obs)
    y_pred = np.asarray(y_pred)
    y_obs_mean = np.mean(y_obs)
    y_pred_mean = np.mean(y_pred)
    mult = np.sum((y_pred - y_pred_mean) *
                  (y_obs - y_obs_mean)) ** 2
    y_obs_sq = np.sum((y_obs - y_obs_mean) ** 2)
    y_pred_sq = np.sum((y_pred - y_pred_mean) ** 2)
    return mult / (y_obs_sq * y_pred_sq)

def get_k(y_obs, y_pred):
    y_obs = np.asarray(y_obs, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    numerator = np.dot(y_obs, y_pred)
    denominator = np.dot(y_pred, y_pred) + 1e-12

    k = numerator / denominator

    return k


def squared_error_zero(y_obs, y_pred):

    y_obs = np.asarray(y_obs)
    y_pred = np.asarray(y_pred)

    k = get_k(y_obs, y_pred)

    y_obs_mean = np.mean(y_obs)

    upp = np.sum((y_obs - k * y_pred) ** 2)
    down = np.sum((y_obs - y_obs_mean) ** 2)

    return 1 - upp / down


def get_rm2(y, f):

    r2 = r_squared_error(y, f)
    r02 = squared_error_zero(y, f)

    return r2 * (1 - np.sqrt(abs(r2 ** 2 - r02 ** 2)))

def get_rm2(y, f):

    r2 = r_squared_error(y, f)
    r02 = squared_error_zero(y, f)
    rm2 = r2 * (1 - np.sqrt(abs(r2-r02)))

    return rm2
def calculate_metrics(Y, P, dataset='davis',type = 'test'):
    # aupr = get_aupr(Y, P)
    # cindex = get_cindex(Y, P)  # DeepDTA
    cindex2 = get_ci(Y, P)  # GraphDTA
    rm2 = get_rm2(Y, P)  # DeepDTA
    mse = get_mse(Y, P)
    pearson = get_pearson(Y, P)
    spearman = get_spearman(Y, P)
    rmse = get_rmse(Y, P)

    result_file_name = './results/result_' + '_' + dataset + '.txt'
    result_str = ''
    result_str += '\n'+type+' '+dataset + '\r\n'
    result_str += 'rmse:' + str(rmse) + ' ' + ' mse:' + str(mse) + ' ' + ' pearson:' + str(
        pearson) + ' ' + 'spearman:' + str(spearman) + ' ' + 'ci:' + str(cindex2) + ' ' + 'rm2:' + str(rm2)
    print(result_str)
    open(result_file_name, 'a').writelines(result_str)
    return mse