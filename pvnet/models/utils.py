"""Utility functions"""

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)

import torch
from torch.distributions import Categorical, Normal

def gmm_loss(pi: torch.Tensor, sigma: torch.Tensor, mu: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Calculates the negative log-likelihood loss for a Gaussian Mixture Model.

    Args:
        pi: The mixture coefficients (probabilities) for each Gaussian component.
            Shape: (batch_size, num_forecast_horizons, num_gaussians)
        sigma: The standard deviations of each Gaussian component.
            Shape: (batch_size, num_forecast_horizons, num_gaussians)
        mu: The means of each Gaussian component.
            Shape: (batch_size, num_forecast_horizons, num_gaussians)
        y: The ground truth target values.
            Shape: (batch_size, num_forecast_horizons)

    Returns:
        The mean negative log-likelihood loss.
    """
    # We need to expand the target y to match the shape of the GMM parameters
    # y becomes: (batch_size, num_forecast_horizons, 1)
    y = y.unsqueeze(-1)

    # Create the mixture distribution (the weights for each Gaussian)
    mixture_distribution = Categorical(probs=pi)

    # Create the component distributions (the Gaussians themselves)
    component_distribution = Normal(loc=mu, scale=sigma)

    # Create the full GMM
    # This object represents the entire probability distribution for each forecast horizon
    gmm = torch.distributions.mixture_same_family.MixtureSameFamily(
        mixture_distribution, component_distribution
    )

    # Calculate the log probability of the true 'y' values under the GMM
    # This tells us how likely our target values were, given the predicted distribution
    log_prob = gmm.log_prob(y)

    # We want to maximize the log-likelihood, which is equivalent to
    # minimizing the negative log-likelihood. We take the mean over the batch.
    return -log_prob.mean()


class PredAccumulator:
    """A class for accumulating y-predictions using grad accumulation and small batch size.

    Attributes:
        _y_hats (list[torch.Tensor]): List of prediction tensors
    """

    def __init__(self):
        """Prediction accumulator"""
        self._y_hats = []

    def __bool__(self):
        return len(self._y_hats) > 0

    def append(self, y_hat: torch.Tensor):
        """Append a sub-batch of predictions"""
        self._y_hats.append(y_hat)

    def flush(self) -> torch.Tensor:
        """Return all appended predictions as single tensor and remove from accumulated store."""
        y_hat = torch.cat(self._y_hats, dim=0)
        self._y_hats = []
        return y_hat


class DictListAccumulator:
    """Abstract class for accumulating dictionaries of lists"""

    @staticmethod
    def _dict_list_append(d1, d2):
        for k, v in d2.items():
            d1[k].append(v)

    @staticmethod
    def _dict_init_list(d):
        return {k: [v] for k, v in d.items()}


class MetricAccumulator(DictListAccumulator):
    """Dictionary of metrics accumulator.

    A class for accumulating, and finding the mean of logging metrics when using grad
    accumulation and the batch size is small.

    Attributes:
        _metrics (Dict[str, list[float]]): Dictionary containing lists of metrics.
    """

    def __init__(self):
        """Dictionary of metrics accumulator."""
        self._metrics = {}

    def __bool__(self):
        return self._metrics != {}

    def append(self, loss_dict: dict[str, float]):
        """Append lictionary of metrics to self"""
        if not self:
            self._metrics = self._dict_init_list(loss_dict)
        else:
            self._dict_list_append(self._metrics, loss_dict)

    def flush(self) -> dict[str, float]:
        """Calculate mean of all accumulated metrics and clear"""
        mean_metrics = {k: np.mean(v) for k, v in self._metrics.items()}
        self._metrics = {}
        return mean_metrics


class BatchAccumulator(DictListAccumulator):
    """A class for accumulating batches when using grad accumulation and the batch size is small.

    Attributes:
        _batches (Dict[str, list[torch.Tensor]]): Dictionary containing lists of metrics.
    """

    def __init__(self, key_to_keep: str = "gsp"):
        """Batch accumulator"""
        self._batches = {}
        self.key_to_keep = key_to_keep

    def __bool__(self):
        return self._batches != {}

    # @staticmethod
    def _filter_batch_dict(self, d):
        keep_keys = [
            self.key_to_keep,
            f"{self.key_to_keep}_id",
            f"{self.key_to_keep}_t0_idx",
            f"{self.key_to_keep}_time_utc",
        ]
        return {k: v for k, v in d.items() if k in keep_keys}

    def append(self, batch: dict[str, list[torch.Tensor]]):
        """Append batch to self"""
        if not self:
            self._batches = self._dict_init_list(self._filter_batch_dict(batch))
        else:
            self._dict_list_append(self._batches, self._filter_batch_dict(batch))

    def flush(self) -> dict[str, list[torch.Tensor]]:
        """Concatenate all accumulated batches, return, and clear self"""
        batch = {}
        for k, v in self._batches.items():
            if k == f"{self.key_to_keep}_t0_idx":
                batch[k] = v[0]
            else:
                batch[k] = torch.cat(v, dim=0)
        self._batches = {}
        return batch
