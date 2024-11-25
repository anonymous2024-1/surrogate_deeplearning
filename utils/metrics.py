import numpy as np
from scipy.stats import pearsonr


def calc_nse(sim: np.ndarray, obs: np.ndarray) -> float:
    """Nash-Sutcliffe-Effiency (NSE)
    """
    sim = sim.flatten() # make sure that metric is calculated over the same dimension
    obs = obs.flatten()

    if obs.shape != sim.shape:
        raise RuntimeError("obs and sim must be of the same length.")

    denominator = np.sum((obs - np.mean(obs))**2) # denominator of the fraction term

    # this would lead to a division by zero error and nse is defined as -inf
    if denominator == 0:
        msg = ["The Nash-Sutcliffe-Efficiency coefficient is not defined ",
            "for the case, that all values in the observations are equal.",
            " Maybe you should use the Mean-Squared-Error instead."]
        # raise RuntimeError("".join(msg))
        return 0

    numerator = np.sum((sim - obs)**2)  # numerator of the fraction term
    nse_val = 1 - numerator / denominator
    return nse_val


def NSE(sim: np.ndarray, obs: np.ndarray) -> float:
    """
        Nash-Sutcliffe-Effiency (NSE)
        sim, obs: [batch, length, 1]
    """
    N, L, D = sim.shape
    nse = []
    for dim in range(D):
        pred = sim[:, :, dim]
        true = obs[:, :, dim]
        numerator = np.sum((pred - true) ** 2) # (B,)

        true_mean = np.mean(true, axis=-1)
        print(true_mean)
        denominator = np.sum((true - true_mean.reshape(N, 1)) ** 2)

        nse_val = 1 - numerator / denominator
        # print(nse_val)
        nse_val = np.mean(nse_val)
        nse.append(nse_val)

    nse = np.array(nse_val)

    return nse

# sim = np.array([[[1.1], [2.1], [3.1], [4.1], [5.1], [6.1], [7.1]],
#        [[1.2], [2.2], [3.2], [4.2], [5.2], [6.2], [7.2]],
#        [[1.5], [2.5], [3.5], [4.5], [5.5], [6.5], [7.5]]])
#
# obs = np.array([[[1], [2], [3], [4], [5], [6], [7]],
#        [[2], [3], [4], [5], [6], [7], [8]],
#        [[0], [1], [2], [3], [4], [5], [6]]])
# #
# print(calc_nse(sim, obs)) # 0.975
# #
# # print(calc_nse(sim[0], obs[0]))
# # print(calc_nse(sim[1], obs[1]))
# # print(calc_nse(sim[2], obs[2]))
# print(NSE(sim, obs))



def calc_fdc_fhv(sim: np.ndarray, obs: np.ndarray, h: float = 0.02) -> float:
    """Peak flow bias of the flow duration curve
    h : float, optional
        Fraction of the flows considered as peak flows. Has to be in range(0,1), by default 0.02
    """
    sim = sim.flatten() # make sure that metric is calculated over the same dimension
    obs = obs.flatten()

    if obs.shape != sim.shape:
        raise RuntimeError("obs and sim must be of the same length.")

    if (h <= 0) or (h >= 1):
        raise RuntimeError("h has to be in the range (0,1)")

    # sort both in descending order
    obs = -np.sort(-obs)
    sim = -np.sort(-sim)

    # subset data to only top h flow values
    obs = obs[:np.round(h * len(obs)).astype(int)]
    sim = sim[:np.round(h * len(sim)).astype(int)]

    fhv = np.sum(sim - obs) / (np.sum(obs) + 1e-6)
    return fhv * 100


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred: np.ndarray, true: np.ndarray):
    """
    :param pred: [B, L, num_features]
    :param true: [B, L, num_features]
    :return: [num_features]
    """
    # B, L, num_features = pred.shape
    # corr = np.full(shape=(num_features), fill_value=np.nan)
    # for i in range(num_features):
    #     pred_1 = pred[:, :, i]
    #     true_1 = true[:, :, i]
    #     pred_mean = pred_1.mean(1)[:, np.newaxis]
    #     true_mean = true_1.mean(1)[:, np.newaxis]
    #
    #     u = ((true_1 - true_mean) * (pred_1 - pred_mean)).sum(1)
    #     d = np.sqrt(((true_1 - true_mean) ** 2).sum(1) * ((pred_1 - pred_mean) ** 2).sum(1))
    #     corr[i] = (u / d).mean(-1)
    # return corr

    pred = pred.flatten()
    true = true.flatten()

    u = ((true - true.mean()) * (pred - pred.mean())).sum()
    d = np.sqrt(((true - true.mean()) ** 2).sum() * ((pred - pred.mean()) ** 2).sum())
    return u / d


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def KGE(sim: np.ndarray, obs: np.ndarray) -> float:
    ''' Calculate the Kling-Gupta Efficiency (KGE) coefficient.
    '''
    sim = sim.flatten()  # make sure that metric is calculated over the same dimension
    obs = obs.flatten()

    if len(sim) != len(obs):
        raise RuntimeError("obs and sim must be of the same length.")

    # Calculate the correlation coefficient
    r = np.corrcoef(obs, sim)[0, 1]

    # Calculate the ratio of the means
    beta = np.mean(sim) / np.mean(obs)

    # Calculate the ratio of the standard deviations
    gamma = np.std(sim) / np.std(obs)

    # Calculate the KGE
    kge = 1 - np.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)

    return kge

def Corr(pred, true):
    pred = pred.flatten()
    true = true.flatten()

    corr, p = pearsonr(pred, true)
    # print(p)
    return corr



def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    # rmse = RMSE(pred, true)
    # mape = MAPE(pred, true)
    # mspe = MSPE(pred, true)
    nse = calc_nse(pred, true)
    kge = KGE(pred, true)
    corr = Corr(pred, true)

    return mae, mse, nse, kge, corr





