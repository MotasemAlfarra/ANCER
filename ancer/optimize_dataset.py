import torch
from tqdm import tqdm

from ddsmoothing import OptimizeSmoothingParameters, Certificate
from .optimization import optimize_ancer


class OptimizeANCERSmoothingParameters(OptimizeSmoothingParameters):
    def __init__(
            self, model: torch.nn.Module,
            test_loader: torch.utils.data.DataLoader, device: str = "cuda:0",
    ):
        """Optimize anisotropic smoothing parameters over a dataset given by
        the loader in test_loader. 

        Args:
            model (torch.nn.Module): trained base model
            test_loader (torch.utils.data.DataLoader): dataset of inputs
            device (str, optional): device on which to perform the computations
        """
        super().__init__()
        self.model = model
        self.device = device
        self.loader = test_loader
        self.num_samples = 0

        # Getting the number of samples in the testloader
        for _, _, idx in self.loader:
            self.num_samples += len(idx)

        self.log(
            "There are in total {} instances in the testloader".format(
                self.num_samples
            )
        )

    def save_theta(
            self, thetas: torch.Tensor, idx: torch.Tensor, path: str,
            flag: str = 'test'
    ):
        """Save the optimized thetas at idx

        Args:
            thetas (torch.Tensor): optimized thetas
            idx (torch.Tensor): indices of the thetas of thetas to be
                saved
            path (str): path to the folder where the thetas should be
                saved
            flag (str, optional): flag to add to the output name
        """
        for i, j in enumerate(idx):
            torch.save(
                thetas[i],
                path + '/theta_' + flag + '_' + str(j.item()) + '.pt'
            )

        self.log(f'Optimized parameters at indices {idx} saved.')

    def run_optimization(
        self, isotropic_thetas: torch.Tensor, output_folder: str,
        iterations: int, certificate: Certificate, lr: float = 0.04,
        num_samples: int = 100, regularization_weight: float = 2
    ):
        """Run the ANCER optimization for the dataset

        Args:
            isotropic_theta (torch.Tensor): initialization isotropic value per
                input in batch
            output_folder (str): path to the folder where the thetas should be
                saved
            iterations (int): number of iterations to run the optimization
            certificate (Certificate): instance of desired certification object
            lr (float, optional): optimization learning rate for ANCER
            num_samples (int): number of samples per input and iteration
            regularization_weight (float, optional): relaxation hyperparameter
        """
        for batch, _, idx in tqdm(self.loader):
            batch = batch.to(self.device)
            thetas = torch.ones_like(batch) * \
                isotropic_thetas[idx].reshape(-1, 1, 1, 1)

            thetas = optimize_ancer(
                self.model, batch, certificate, lr, thetas, iterations,
                num_samples, kappa=regularization_weight,
                device=self.device
            )

            # save the optimized thetas
            self.save_theta(thetas.detach(), idx, output_folder)
