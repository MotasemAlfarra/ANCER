import torch
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint
from torch.distributions.normal import Normal


class Certificate():
    def compute_proxy_gap(self, logits: torch.Tensor):
        raise NotImplementedError

    def sample_noise(self, batch: torch.Tensor, repeated_theta: torch.Tensor):
        raise NotImplementedError

    def compute_gap(self, pABar: float):
        raise NotImplementedError


class L2Certificate(Certificate):
    def __init__(self, batch_size: int, device: str = "cuda:0"):
        self.m = Normal(torch.zeros(batch_size).to(device),
                        torch.ones(batch_size).to(device))
        self.device = device
        self.norm = "l2"

    def compute_proxy_gap(self, logits: torch.Tensor):
        return self.m.icdf(logits[:, 0].clamp_(0.001, 0.999)) - \
            self.m.icdf(logits[:, 1].clamp_(0.001, 0.999))

    def sample_noise(self, batch: torch.Tensor, repeated_theta: torch.Tensor):
        return torch.randn_like(batch, device=self.device) * repeated_theta

    def compute_gap(self, pABar: float):
        return norm.ppf(pABar)


class L1Certificate(Certificate):
    def __init__(self, device="cuda:0"):
        self.device = device
        self.norm = "l1"

    def compute_proxy_gap(self, logits: torch.Tensor):
        return logits[:, 0] - logits[:, 1]

    def sample_noise(self, batch: torch.Tensor, repeated_theta: torch.Tensor):
        return 2 * (torch.rand_like(batch, device=self.device) - 0.5) * repeated_theta

    def compute_gap(self, pABar: float):
        return 2 * (pABar - 0.5)


class Smooth(object):
    """A smoothed classifier g

    Adapted from: https://github.com/locuslab/smoothing/blob/master/code/core.py
    """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, sigma: torch.Tensor, certificate: Certificate):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma
        self.certificate = certificate

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int, device: torch.device = torch.device('cuda:0')) -> (int, float):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2/L1 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2/L1 ball of radius R around x.

        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, n0, batch_size, device=device)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size, device=device)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0
        else:
            return cAHat, self.certificate.compute_gap(pABar)

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int, device: torch.device = torch.device('cuda:0')) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).

        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.

        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size, device=device)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]

    def _sample_noise(self, x: torch.tensor, num: int, batch_size, device: torch.device = torch.device('cuda:0')) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            # counts = np.zeros(self.num_classes, dtype=int)
            counts = torch.zeros(self.num_classes, dtype=float, device=device)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = self.certificate.sample_noise(batch, self.sigma)
                # noise = torch.randn_like(batch, device='cuda') * self.sigma
                predictions = self.base_classifier(batch + noise).argmax(1)
                counts += self._count_arr(predictions,
                                          device, self.num_classes)
            return counts.cpu().numpy()

    def _count_arr(self, arr: torch.tensor, device: torch.device, length: int) -> torch.tensor:
        counts = torch.zeros(length, dtype=torch.long, device=device)
        unique, c = arr.unique(sorted=False, return_counts=True)
        counts[unique] = c
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.

        This function uses the Clopper-Pearson method.

        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]
