"""
Loss functions for PyTorch.
"""
import torch
import torch.nn as nn
import numpy as np


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self,
                insample: torch.Tensor,
                freq: int,
                forecast: torch.Tensor,
                target: torch.Tensor,
                mask: torch.Tensor) -> torch.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return torch.mean(torch.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self,
                insample: torch.Tensor,
                freq: int,
                forecast: torch.Tensor,
                target: torch.Tensor,
                mask: torch.Tensor) -> torch.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * torch.mean(divide_no_nan(torch.abs(forecast - target),
                                              torch.abs(forecast.data) + torch.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self,
                insample: torch.Tensor,
                freq: int,
                forecast: torch.Tensor,
                target: torch.Tensor,
                mask: torch.Tensor) -> torch.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = torch.mean(torch.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])

        return torch.mean(torch.abs(target - forecast) * masked_masep_inv)


class nse_loss(nn.Module):
    """Calculate (batch-wise) NSE Loss.
    Each sample i is weighted by 1 / (std_i + eps)^2, where std_i is the standard deviation of the
    discharge from the basin, to which the sample belongs.
    Parameters:
    -----------
    eps : float
        Constant, added to the weight for numerical stability and smoothing, default to 0.1
    """
    def __init__(self, eps: float = 0.01):
        super(nse_loss, self).__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, q_stds: torch.Tensor):
        """Calculate the batch-wise NSE Loss function.
        Parameters
        ----------
        y_pred : torch.Tensor
            Tensor containing the network prediction.
        y_true : torch.Tensor
            Tensor containing the true discharge values
        q_stds : torch.Tensor
            Tensor containing the discharge std (calculate over training period) of each sample
        Returns
        -------
        torch.Tenor
            The (batch-wise) NSE Loss
        """
        squared_error = (y_pred - y_true)**2
        weights = 1 / (q_stds + self.eps)**2
        scaled_loss = weights * squared_error

        return torch.mean(scaled_loss)
