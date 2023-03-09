# pylint: disable = (missing-module-docstring)

import argparse
import json
import os
from typing import Dict, List, Union

import pandas as pd


def load_benchmark_result(
    path: str,
) -> Dict[str, List[Dict[str, Union[float, int, str]]]]:
    benchmark_results: Dict[str, List[Dict[str, Union[float, int, str]]]] = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as file:
            benchmark_results = json.load(file)
    else:
        raise RuntimeError(f"File doesn't exist: {path}")

    return benchmark_results


def parse_model_results(
    data: List[Dict[str, Union[float, int, str]]],
    value_key: str,
) -> str:
    inference_time_dict: Dict[str, Dict[str, Union[float, int]]] = {}
    for entry in data:
        benchmark_name = str(entry["benchmark_name"])
        if benchmark_name == "BenchmarkONNX":
            if entry["use_cuda"]:
                benchmark_name = f"{benchmark_name} GPU"
            else:
                benchmark_name = f"{benchmark_name} CPU"

        if entry["use_jit"]:
            benchmark_name = f"{benchmark_name} JIT"

        if benchmark_name == "BenchmarkTensorPTQ":
            benchmark_name = f"{benchmark_name} GPU INT8"
        elif benchmark_name == "BenchmarkTensorDynamicQuantization":
            benchmark_name = f"{benchmark_name} CPU INT8"
        else:
            if entry["use_fp16"]:
                benchmark_name = f"{benchmark_name} FP16"
            else:
                benchmark_name = f"{benchmark_name} FP32"

        value = entry[value_key]
        batch_size = f"batch size {entry['batch_size']}"
        try:
            inference_time_dict[benchmark_name][batch_size] = value
        except KeyError:
            inference_time_dict[benchmark_name] = {batch_size: value}

    return pd.read_json(
        json.dumps(inference_time_dict),
        orient="index",
    ).to_markdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Benchmark model optimization techniques")
    parser.add_argument(
        "--result_file",
        type=str,
        default="benchmark_log.json",
        help="Filename of a file where all benchmark results will be stored.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_result = load_benchmark_result(path=args.result_file)
    for model_name, results in benchmark_result.items():
        print(model_name)
        print("\nInference time [ms/sample]")
        print(parse_model_results(data=results, value_key="inference_time"))
        print("\nGPU Memory Peak usage [MB] - max_memory_allocated")
        print(parse_model_results(data=results, value_key="memory_usage"))
        print("\nF1 score")
        print(parse_model_results(data=results, value_key="f1"))


if __name__ == "__main__":
    main()
