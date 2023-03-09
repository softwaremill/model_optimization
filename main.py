# pylint: disable = (missing-module-docstring)

import argparse
import json
import os
from typing import Dict, Union

import torch

# to install version 1.3.0 follow
# https://github.com/pytorch/TensorRT/issues/1371#issuecomment-1256035010
import torch_tensorrt

from src.benchmark import (
    BenchmarkCPU,
    BenchmarkCUDA,
    BenchmarkONNX,
    BenchmarkTensorDynamicQuantization,
    BenchmarkTensorPruning,
    BenchmarkTensorPTQ,
    BenchmarkTensorRT,
)
from src.dataset_utils import (
    DatasetFactory,
    DatasetImagenetMiniFactory,
    DatasetIMDBFactory,
)
from src.memory import vram_monitor_factory
from src.model_utils import save_torchscript

torch_tensorrt.logging.set_reportable_log_level(
    torch_tensorrt.logging.Level(torch_tensorrt.logging.Level.Error)
)


def append_results_to_log_file(
    path: str, model_name: str, data: Dict[str, Union[str, float]]
) -> None:
    benchmark_results = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as file:
            benchmark_results = json.load(file)

    if model_name in benchmark_results.keys():
        benchmark_results[model_name].append(data)
    else:
        benchmark_results[model_name] = [data]

    with open(path, "w", encoding="utf-8") as file:
        json.dump(benchmark_results, file, indent=4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Benchmark model optimization techniques")
    parser.add_argument(
        "--type",
        choices=[
            "cpu",
            "cuda",
            "tensorrt",
            "quantization",
            "dynamic_quantization",
            "pruning",
            "onnx_cpu",
            "onnx_gpu",
        ],
        required=True,
        help="Model's operation type.",
    )
    parser.add_argument(
        "--model_name",
        choices=[
            "swin_t",
            "vit",
            "resnet",
            "mobilenet",
            "fcn",
            "cnn",
            "rnn",
            "bert",
            "t5",
            "gptneo",
        ],
        required=True,
        help="Model's name.",
    )
    parser.add_argument(
        "--use_fp16", action="store_true", help="Use half precision model."
    )
    parser.add_argument("--use_jit", action="store_true", help="Use JIT model.")
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Size of processed batch."
    )
    parser.add_argument(
        "--n_runs",
        type=int,
        default=1,
        help="Number of runs to compute mean of inference times.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="saved_models",
        help="Directory with saved JIT models.",
    )
    parser.add_argument(
        "--model_filename",
        type=str,
        default="model_jit.pth",
        help="JIT model file name.",
    )
    parser.add_argument(
        "--pruning_ratio",
        type=float,
        default=0.2,
        help="Ratio of model's pruned weights.",
    )
    parser.add_argument(
        "--pretrained_model_name",
        type=str,
        help="Name of a model to load from huggingface.",
    )
    parser.add_argument(
        "--structural_pruning", action="store_true", help="Use structural pruning."
    )

    parser.add_argument(
        "--max_length",
        type=int,
        default=100,
        help="Max processed text in number of tokens.",
    )
    parser.add_argument(
        "--data_dir", type=str, default="data/", help="ImageNet-Mini dataset root dir."
    )
    parser.add_argument(
        "--subset_name",
        type=str,
        default="val",
        help="Subset of ImageNet-Mini dataset: val or train.",
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=150,
        help="Number of samples from IMDB dataset to use.",
    )
    parser.add_argument(
        "--result_file",
        type=str,
        default="benchmark_log.json",
        help="Filename of a file where all benchmark results will be stored.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # has influence on performance on CNNs:
    # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#enable-cudnn-auto-tuner
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # https://pytorch.org/docs/stable/amp.html

    # For now, we suggest to disable the Jit Autocast Pass,
    # As the issue: https://github.com/pytorch/pytorch/issues/75956
    if args.use_jit:
        torch._C._jit_set_autocast_mode(  # pylint: disable = (protected-access,c-extension-no-member)
            False
        )

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device detected. Exiting...")

    cuda_device = torch.device("cuda:0")  # pylint: disable = (no-member)
    cpu_device = torch.device("cpu:0")  # pylint: disable = (no-member)

    dataset_factory: DatasetFactory
    if args.model_name in ["bert", "t5", "gptneo"]:
        dataset_factory = DatasetIMDBFactory(
            pretrained_model_name=args.pretrained_model_name,
            dataset_size=args.dataset_size,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
    else:
        dataset_factory = DatasetImagenetMiniFactory(
            data_dir=args.data_dir,
            subset_name=args.subset_name,
        )

    example_inputs = dataset_factory.get_example_inputs()

    # save model's torchscript .pth file
    model_torchscript_path: str = os.path.join(args.model_dir, args.model_filename)
    if args.use_jit:
        save_torchscript(
            model_name=args.model_name,
            device=cpu_device,
            batch_size=args.batch_size,
            model_torchscript_path=model_torchscript_path,
            example_inputs=example_inputs,
        )

    vram_monitor_factory(interval=1e-3, device="cuda:0")

    result_dict: Dict[str, Union[int, float]]
    # compute inference time, CUDA memory usage and F1 score
    if args.type == "cpu":
        result_dict = BenchmarkCPU().benchmark(
            model_name=args.model_name,
            device=cpu_device,
            batch_size=args.batch_size,
            dataset_factory=dataset_factory,
            model_torchscript_path=model_torchscript_path,
            use_jit=args.use_jit,
            use_fp16=args.use_fp16,
            n_runs=args.n_runs,
        )
    elif args.type == "cuda":
        result_dict = BenchmarkCUDA().benchmark(
            model_name=args.model_name,
            device=cuda_device,
            batch_size=args.batch_size,
            dataset_factory=dataset_factory,
            model_torchscript_path=model_torchscript_path,
            use_jit=args.use_jit,
            use_fp16=args.use_fp16,
            n_runs=args.n_runs,
        )
    elif args.type == "tensorrt":
        result_dict = BenchmarkTensorRT().benchmark(
            model_name=args.model_name,
            device=cuda_device,
            batch_size=args.batch_size,
            dataset_factory=dataset_factory,
            model_torchscript_path=model_torchscript_path,
            use_jit=args.use_jit,
            use_fp16=args.use_fp16,
            n_runs=args.n_runs,
        )
    elif args.type == "quantization":
        result_dict = BenchmarkTensorPTQ().benchmark(
            model_name=args.model_name,
            device=cuda_device,
            batch_size=args.batch_size,
            dataset_factory=dataset_factory,
            use_jit=args.use_jit,
            use_fp16=args.use_fp16,
            n_runs=args.n_runs,
            model_torchscript_path=model_torchscript_path,
        )
    elif args.type == "dynamic_quantization":
        result_dict = BenchmarkTensorDynamicQuantization().benchmark(
            model_name=args.model_name,
            device=cpu_device,
            dataset_factory=dataset_factory,
            batch_size=args.batch_size,
            model_torchscript_path=model_torchscript_path,
            use_jit=args.use_jit,
            use_fp16=args.use_fp16,
            n_runs=args.n_runs,
        )
    elif args.type == "pruning":
        result_dict = BenchmarkTensorPruning().benchmark(
            model_name=args.model_name,
            device=cpu_device,
            batch_size=args.batch_size,
            model_torchscript_path=model_torchscript_path,
            dataset_factory=dataset_factory,
            use_jit=args.use_jit,
            use_fp16=args.use_fp16,
            n_runs=args.n_runs,
            name="weight",
            amount=args.pruning_ratio,
            structural_pruning=args.structural_pruning,
        )
    elif args.type == "onnx_cpu":
        result_dict = BenchmarkONNX().benchmark(
            model_name=args.model_name,
            device=cpu_device,
            dataset_factory=dataset_factory,
            batch_size=args.batch_size,
            model_torchscript_path=model_torchscript_path,
            use_jit=args.use_jit,
            use_fp16=args.use_fp16,
            n_runs=args.n_runs,
            use_cuda=False,
        )
    elif args.type == "onnx_gpu":
        result_dict = BenchmarkONNX().benchmark(
            model_name=args.model_name,
            device=cpu_device,
            dataset_factory=dataset_factory,
            batch_size=args.batch_size,
            model_torchscript_path=model_torchscript_path,
            use_jit=args.use_jit,
            use_fp16=args.use_fp16,
            n_runs=args.n_runs,
            use_cuda=True,
        )

    append_results_to_log_file(
        path=args.result_file,
        model_name=args.model_name,
        data=result_dict,
    )


if __name__ == "__main__":
    main()
