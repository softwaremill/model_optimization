# pylint: disable = (missing-module-docstring)

import os

import numpy as np
import torch
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    ResNet18_Weights,
    Swin_T_Weights,
    ViT_B_16_Weights,
    mobilenet_v3_large,
    resnet18,
    swin_t,
    vit_b_16,
)

from src.model import T5, Bert, CustomCNN, CustomFCN, CustomLSTM, GPTNeo


def get_model_name(
    model_name: str,
    device: torch.device,  # pylint: disable = (no-member)
    batch_size: int,
) -> str:
    model = load_model(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )
    return model.__class__.__name__


def load_model(
    model_name: str,
    device: torch.device,  # pylint: disable = (no-member)
    batch_size: int,
) -> torch.nn.Module:
    if model_name == "swin_t":
        model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
    elif model_name == "vit":
        model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    elif model_name == "resnet":
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    elif model_name == "mobilenet":
        model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    elif model_name == "fcn":
        model = CustomFCN(input_size=3 * 224 * 224, hidden_size=224, num_classes=1000)
    elif model_name == "cnn":
        model = CustomCNN(num_classes=1000)
    elif model_name == "rnn":
        model = CustomLSTM(
            input_size=224,
            hidden_size=100,
            layer_size=100,
            num_classes=1000,
            batch_size=batch_size,
            device=device,
        )
    elif model_name == "bert":
        model = Bert()
    elif model_name == "t5":
        model = T5()
    elif model_name == "gptneo":
        model = GPTNeo()

    model.eval()
    return model


def save_torchscript_model(
    model: torch.nn.Module,
    model_torchscript_path: str,
    example_inputs=None,
) -> None:
    model_dir = os.path.dirname(model_torchscript_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if example_inputs is not None:
        traced_model = torch.jit.trace(
            model, example_inputs=example_inputs
        )  # it doesn't work with empty example_inputs
    else:
        traced_model = torch.jit.script(model, example_inputs=example_inputs)

    torch.jit.save(traced_model, model_torchscript_path)


def load_torchscript_model(
    model_torchscript_path: str,
    device: torch.device,  # pylint: disable = (no-member)
) -> torch.ScriptModule:  # pylint: disable = (no-member)
    model = torch.jit.load(model_torchscript_path, map_location=device).eval()
    # essential line
    # https://pytorch.org/docs/stable/generated/torch.jit.optimize_for_inference.html#torch.jit.optimize_for_inference
    model = torch.jit.optimize_for_inference(model)
    return model


def save_torchscript(
    model_name: str,
    device: torch.device,  # pylint: disable = (no-member)
    batch_size: int,
    model_torchscript_path: str,
    example_inputs=None,
) -> None:
    model = load_model(model_name=model_name, device=device, batch_size=batch_size)
    save_torchscript_model(
        model=model,
        model_torchscript_path=model_torchscript_path,
        example_inputs=example_inputs,
    )
    del model


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )
