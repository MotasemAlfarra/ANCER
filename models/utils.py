import torch
from torchvision.models.resnet import resnet50

from models import resnet
from models.wideresnet import WideResNet


def load_model(
        path: str, model_type: str, num_classes: int, device: str = "cuda"
        ):
    # crate model base on model type
    if model_type == "resnet18":
        model = resnet.resnet18(num_classes=num_classes).to(device)
    elif model_type == "wideresnet40":
        model = WideResNet(
            depth=40,
            widen_factor=2,
            num_classes=num_classes
        ).to(device)
    elif model_type == "resnet50":
        model = torch.nn.DataParallel(resnet50(pretrained=False)).to(device)
    else:
        raise ValueError("model_type requested not available")

    # load the checkpoint
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    return model
