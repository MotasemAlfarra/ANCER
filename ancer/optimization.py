import torch
import numpy as np
from torch.autograd import Variable
from ddsmoothing.certificate import Certificate


def optimize_ancer(
        model: torch.nn.Module, batch: torch.Tensor,
        certificate: Certificate, learning_rate: float,
        isotropic_theta: torch.Tensor, iterations: int,
        samples: int, kappa: float, device: str = "cuda:0"
) -> torch.Tensor:
    """Optimize batch using ANCER, assuming isotropic initialization point.

    Args:
        model (torch.nn.Module): trained network
        batch (torch.Tensor): inputs to certify around
        certificate (Certificate): instance of desired certification object
        learning_rate (float): optimization learning rate for ANCER
        isotropic_theta (torch.Tensor): initialization isotropic value per
            input in batch
        iterations (int): number of iterations to run the optimization
        samples (int): number of samples per input and iteration
        kappa (float): relaxation hyperparameter
        device (str, optional): device on which to perform the computations

    Returns:
        torch.Tensor: optimized anisotropic thetas
    """
    batch_size = batch.shape[0]
    img_size = np.prod(batch.shape[1:])

    # define a variable, the optimizer, and the initial sigma values
    theta = Variable(isotropic_theta, requires_grad=True).to(device)
    optimizer = torch.optim.Adam([theta], lr=learning_rate)
    initial_theta = theta.detach().clone()

    # reshape vectors to have ``samples`` per input in batch
    new_shape = [batch_size * samples]
    new_shape.extend(batch[0].shape)
    new_batch = batch.repeat((1, samples, 1, 1)).view(new_shape)

    # solve iteratively by projected gradient ascend
    for _ in range(iterations):
        theta_repeated = theta.repeat(1, samples, 1, 1).view(new_shape)

        # Reparameterization trick
        noise = certificate.sample_noise(new_batch, theta_repeated)
        out = model(
            new_batch + noise
        ).reshape(batch_size, samples, -1).mean(dim=1)

        vals, _ = torch.topk(out, 2)
        gap = certificate.compute_proxy_gap(vals)

        prod = torch.prod(
            (theta.reshape(batch_size, -1))**(1/img_size), dim=1)
        proxy_radius = prod * gap

        radius_maximizer = - (
            proxy_radius.sum() +
            kappa *
            (torch.min(theta.view(batch_size, -1), dim=1).values*gap).sum()
        )
        radius_maximizer.backward()
        optimizer.step()
        optimizer.zero_grad()

        # project to the initial theta
        with torch.no_grad():
            torch.max(theta, initial_theta, out=theta)

    return theta
