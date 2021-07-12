import argparse

import torch
import numpy as np
from tqdm import tqdm
from torch.distributions import Categorical

from ddsmoothing.utils.datasets import cifar10, get_num_classes
from ddsmoothing.utils.models import get_model


def sample_uniform_noise(x: torch.Tensor, sigma: float) -> torch.Tensor:
    lam = sigma * (3 ** -0.5)
    return (torch.rand_like(x, device=device) - 0.5) * 2 * lam + x


def direct_train_log_lik(
        model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor,
        sigma: float, sample_size: int = 16):
    """
    Log-likelihood for direct training (numerically stable with logusmexp
    trick).
    """
    samples_shape = torch.Size([x.shape[0], sample_size]) + x.shape[1:]
    samples = x.unsqueeze(1).expand(samples_shape)
    samples = samples.reshape(torch.Size([-1]) + samples.shape[2:])
    samples = sample_uniform_noise(samples, sigma)
    thetas = model.forward(samples).view(x.shape[0], sample_size, -1)

    predicted = thetas.mean(1).argmax(1)
    loss = -(torch.logsumexp(
        thetas[torch.arange(x.shape[0]), :, y] -
        torch.logsumexp(thetas, dim=2), dim=1
    ) - torch.log(torch.tensor(sample_size, dtype=torch.float, device=device)))

    return predicted, loss


def train_epoch(
        epoch: int, model: torch.nn.Module, sigma: float,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer, n_noisy_samples: float = 1,
        mode: str = "direct"
):
    model = model.train()
    train_loss = 0
    total = 0
    correct = 0
    for batch_idx, (batch, targets, idx) in enumerate(train_loader):
        optimizer.zero_grad()

        batch_size = len(idx)
        batch, targets = batch.to(device), targets.to(device)

        # follow the implementation in
        # https://github.com/tonyduan/rs4a/blob/master/src/train.py
        # if direct, just compute the loss directly
        if mode == "direct":
            predicted, loss = direct_train_log_lik(
                model,
                batch,
                targets,
                sigma,
                sample_size=n_noisy_samples
            )
        elif "stability":
            # if using stability training, use this
            x = sample_uniform_noise(batch, sigma)
            x_tilde = sample_uniform_noise(batch, sigma)

            pred_x = Categorical(logits=model.forward(x))
            pred_x_tilde = Categorical(logits=model.forward(x_tilde))

            loss = -pred_x.log_prob(targets) + 6.0 * \
                torch.distributions.kl_divergence(pred_x, pred_x_tilde)

            predicted = pred_x.probs.argmax()
        else:
            raise ValueError("training mode not recognized")

        loss = loss.mean()

        # update parameters
        loss.backward()
        optimizer.step()

        train_loss += loss.item()*len(batch)

        total += batch_size * n_noisy_samples
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 100 == 0:
            print(
                "+ Epoch: {}. Iter: [{}/{} ({:.0f}%)]. Loss: {}. Accuracy: {}".format(
                    epoch,
                    batch_idx * batch_size,
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    train_loss / total,
                    100.*correct / total
                )
            )


def test(
        epoch: int, model: torch.nn.Module, sigma: float,
        test_loader: torch.utils.data.DataLoader
) -> float:
    model = model.eval()
    total = 0
    correct = 0
    correct_corrupted = 0
    for batch_idx, (batch, targets, idx) in enumerate(test_loader):
        batch, targets = batch.to(device), targets.to(device)

        with torch.no_grad():
            batch_corrupted = sample_uniform_noise(batch, sigma)

            # forward pass through the base classifier
            outputs_softmax = model(batch)
            outputs_corrputed_softmax = model(batch_corrupted)

        predicted = outputs_softmax.argmax(1)
        predicted_corrupted = outputs_corrputed_softmax.argmax(1)

        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        correct_corrupted += predicted_corrupted.eq(targets).sum().item()

    print('===> Test Accuracy: {}. Test Accuracy Corrupted: {}'.format(
        100.*correct / total,
        100.*correct_corrupted / total
    ))

    return 100.*correct_corrupted / total


if __name__ == "__main__":
    torch.manual_seed(0)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)

    # Training settings
    parser = argparse.ArgumentParser(
        description='Train WideResNet40 on CIFAR-10 following the ' +
        'procedure from RS4A'
    )
    parser.add_argument(
        '-o', '--output-file', type=str,
        required=True, help='file to save the model'
    )
    parser.add_argument(
        '-s', '--sigma', type=float,
        required=True, help='sigma to train with'
    )
    parser.add_argument(
        '--n-noisy-samples', type=int, default=1,
        help='number of noisy samples per training iteration'
    )
    parser.add_argument(
        "--direct", action="store_true",
        help='direct training uses simply data augmentation'
    )

    # training arguments
    parser.add_argument(
        '--batch-sz', type=int, default=128,
        help='input batch size for training'
    )
    parser.add_argument(
        '--epochs', type=int, default=120,
        help='number of epochs to train'
    )
    parser.add_argument(
        '--lr', type=float, default=0.01,
        help='initial learning rate'
    )
    parser.add_argument(
        '--momentum', type=int, default=0.9,
        help='momentum for optimizer'
    )
    parser.add_argument(
        '--weight-decay', type=int, default=1e-4,
        help='weight decay for optimizer'
    )
    parser.add_argument(
        '--step-sz', type=int, default=30,
        help='learning rate drop every step_sz epochs for optimizer'
    )
    parser.add_argument(
        '--gamma', type=int, default=0.1,
        help='gamma factor to drop learning rate for optimizer'
    )

    args = parser.parse_args()

    # load dataset and prepare the model
    train_loader, test_loader, img_sz, _, _ = cifar10(args.batch_sz)
    model = get_model(
        "wideresnet40",
        num_classes=get_num_classes("cifar10"),
        device=device
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.step_sz,
        gamma=args.gamma
    )

    mode = "direct" if args.direct else "stability"
    print(f"Mode: {mode}")

    best_acc = 0.0
    for epoch in tqdm(range(args.epochs)):
        train_epoch(
            epoch, model, args.sigma, train_loader, optimizer,
            n_noisy_samples=args.n_noisy_samples, mode=mode
        )

        test_acc = test(
            epoch, model, args.sigma, test_loader
        )

        scheduler.step()

    torch.save(
        {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer_param': optimizer.state_dict(),
        },
        args.ouput_file
    )
