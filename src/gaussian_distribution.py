import numpy as np
import torch


class GaussianDistribution:
    """
    ## Gaussian Distribution
    """

    def __init__(self, parameters):
        """
        :param parameters: are the means and log of variances of the embedding of shape
            `[batch_size, z_channels * 2, z_height, z_height]`
        """
        self.parameters = parameters
        # Split mean and log of variance
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        # Clamp the log of variances
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        # Calculate standard deviation
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)

    def sample(self):
        # Sample from the distribution
        x = self.mean + self.std * torch.randn_like(
            self.std, dtype=self.std.dtype, device=self.std.device
        )
        return x

    def kl(self, other=None):
        if other is None:
            return 0.5 * torch.sum(
                torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                dim=[1, 2, 3],
            )
        else:
            return 0.5 * torch.sum(
                torch.pow(self.mean - other.mean, 2) / other.var
                + self.var / other.var
                - 1.0
                - self.logvar
                + other.logvar,
                dim=[1, 2, 3],
            )

    def nll(self, sample, dims=[1, 2, 3]):
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self):
        return self.mean
