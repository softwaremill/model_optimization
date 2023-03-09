# Model's optimizations

This repository contains scripts to perform a benchmark of different ways to optimize
neural network model's inference. The benchmark reports inference time per single
sample, VRAM requirements and precision of a model.

## Install

To install all dependencies run `poetry install` command.

## pre-commit

To install `pre-commit` hooks run `poetry run pre-commit install` command. After that
code changes will be checked by hooks after every commit. You can also trigger them
without commiting changes using `pre-commit run` command.

## Dataset

The benchmark uses the ImageNet-mini dataset, which can be downloaded from the site:
https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000. After downloading, extract
the archive in to a directory of a directory named `data/` located in the directory
with the cloned repository.

## Supported models

At the moment repository supports only a few of models from `torchvision` and `transformer`
libraries. Supported models are:
- ResNet18
- MobileNetV3 Large
- BERT
- T5
- GPTNeo
- custom FCN
- custom CNN
- custom LSTM

## Run

To run benchmark run:
```bash
bash run_benchmark.sh
```

If You want to modify a number of iterations or the neural network model modify a variable
at the top of the `run_benchmark.sh` script. The content of this script is as follows:

```bash
#!/bin/bash

MODEL_NAME="resnet"
PRETRAINED_MODEL_NAME="textattack/bert-base-uncased-imdb"
N_RUNS="5"
...
```

## Parse results

To convert result `JSON` file to markdown table run
`poetry run python3 convert_results_json_to_markdown.py`.

## Conclusions

### VRAM memory usage

To minimize the model size on the GPU use `ONNX`.

### Quantization

The TensorRT `INT8` quantization gives the greatest acceleration of the model inference,
but results in a noticeable decrease in the model accuracy. The decrease is greater
the smaller the neural network.


## Results

 Benchmark environment:
* Torch-TensorRT Version (e.g. 1.0.0): 1.3.0
* PyTorch Version (e.g. 1.0): 1.13.1
* CPU Architecture: AMD® Ryzen 9 5950x 16-core processor × 32
* OS (e.g., Linux): Ubuntu 22.04.2 LTS
* Python version: 3.9.16
* CUDA version: 11.6
* GPU models and configuration: GeForce RTX 3080 Ti

<details>
<summary>MobileNetV3 Large</summary>

Inference time [ms/sample]
|                                             |   batch size 1 |   batch size 16 |   batch size 32 |   batch size 64 |
|:--------------------------------------------|---------------:|----------------:|----------------:|----------------:|
| BenchmarkCPU FP32                           |          6.285 |           2.333 |           2.644 |           3.535 |
| BenchmarkCPU JIT FP32                       |          4.189 |           1.885 |           2.363 |           3.456 |
| BenchmarkCUDA FP32                          |          4.798 |           0.296 |           0.181 |           0.168 |
| BenchmarkCUDA FP16                          |          4.536 |           0.526 |           0.425 |           0.338 |
| BenchmarkCUDA JIT FP32                      |          2.379 |           0.239 |           0.221 |           0.214 |
| BenchmarkTensorRT FP32                      |          0.71  |           0.128 |           0.111 |           0.105 |
| BenchmarkTensorRT FP16                      |          0.541 |           0.078 |           0.066 |           0.058 |
| BenchmarkTensorRT JIT FP32                  |          1.994 |           0.171 |           0.204 |           0.146 |
| BenchmarkTensorRT JIT FP16                  |          1.923 |           0.149 |           0.131 |           0.124 |
| BenchmarkTensorPTQ GPU INT8                 |          0.523 |           0.086 |           0.047 |           0.04  |
| BenchmarkTensorDynamicQuantization CPU INT8 |          6.407 |           2.468 |           2.742 |           3.229 |
| BenchmarkONNX CPU FP32                      |         10.285 |           8.629 |           9.265 |          10.289 |
| BenchmarkONNX GPU FP32                      |          1.354 |           0.332 |           0.333 |           0.343 |

GPU Memory Peak usage [MB] - max_memory_allocated
|                                             |   batch size 1 |   batch size 16 |   batch size 32 |   batch size 64 |
|:--------------------------------------------|---------------:|----------------:|----------------:|----------------:|
| BenchmarkCPU FP32                           |         0.0625 |          0      |          0      |          0      |
| BenchmarkCPU JIT FP32                       |         0      |          0      |         94.9375 |          0      |
| BenchmarkCUDA FP32                          |      4216.19   |       4051      |       4219      |       4387      |
| BenchmarkCUDA FP16                          |      3773      |       3693      |       3848.06   |       3857      |
| BenchmarkCUDA JIT FP32                      |      4175      |       4131      |       3356.69   |       4221      |
| BenchmarkTensorRT FP32                      |      4067.06   |       4113.19   |       4199      |       4273      |
| BenchmarkTensorRT FP16                      |      4175      |       4051      |       4040.12   |       4099.19   |
| BenchmarkTensorRT JIT FP32                  |      4668.56   |       4415      |       4613.12   |       4797      |
| BenchmarkTensorRT JIT FP16                  |      4789.19   |       4395.19   |       4306.19   |       4633      |
| BenchmarkTensorPTQ GPU INT8                 |      4164.38   |       4029      |       4065      |       4025.19   |
| BenchmarkTensorDynamicQuantization CPU INT8 |         0      |          0      |          0      |          0      |
| BenchmarkONNX CPU FP32                      |         0.1875 |          0.1875 |          0      |          0.1875 |
| BenchmarkONNX GPU FP32                      |      1461      |       1937      |       1937      |       2009      |

F1 score
|                                             |   batch size 1 |   batch size 16 |   batch size 32 |   batch size 64 |
|:--------------------------------------------|---------------:|----------------:|----------------:|----------------:|
| BenchmarkCPU FP32                           |          0.734 |           0.734 |           0.734 |           0.734 |
| BenchmarkCPU JIT FP32                       |          0.734 |           0.734 |           0.734 |           0.734 |
| BenchmarkCUDA FP32                          |          0.734 |           0.734 |           0.734 |           0.734 |
| BenchmarkCUDA FP16                          |          0.735 |           0.735 |           0.734 |           0.734 |
| BenchmarkCUDA JIT FP32                      |          0.734 |           0.734 |           0.734 |           0.734 |
| BenchmarkTensorRT FP32                      |          0.734 |           0.734 |           0.734 |           0.734 |
| BenchmarkTensorRT FP16                      |          0.735 |           0.735 |           0.735 |           0.735 |
| BenchmarkTensorRT JIT FP32                  |          0.734 |           0.734 |           0.734 |           0.734 |
| BenchmarkTensorRT JIT FP16                  |          0.735 |           0.734 |           0.735 |           0.735 |
| BenchmarkTensorPTQ GPU INT8                 |          0.693 |           0.699 |           0.697 |           0.711 |
| BenchmarkTensorDynamicQuantization CPU INT8 |          0.734 |           0.734 |           0.734 |           0.734 |
| BenchmarkONNX CPU FP32                      |          0.734 |           0.734 |           0.734 |           0.734 |
| BenchmarkONNX GPU FP32                      |          0.734 |           0.734 |           0.734 |           0.734 |

</details>

<details>
<summary>ResNet18</summary>

Inference time [ms/sample]
|                                             |   batch size 1 |   batch size 16 |   batch size 32 |   batch size 64 |
|:--------------------------------------------|---------------:|----------------:|----------------:|----------------:|
| BenchmarkCPU FP32                           |          6.567 |           4.04  |           5.079 |           4.283 |
| BenchmarkCPU JIT FP32                       |          4.238 |           3.211 |           3.705 |           3.467 |
| BenchmarkCUDA FP32                          |          2.104 |           0.254 |           0.253 |           0.215 |
| BenchmarkCUDA FP16                          |          2.05  |           0.423 |           0.442 |           0.403 |
| BenchmarkCUDA JIT FP32                      |          1.536 |           0.239 |           0.202 |           0.174 |
| BenchmarkTensorRT FP32                      |          0.862 |           0.217 |           0.201 |    RuntimeError |
| BenchmarkTensorRT FP16                      |          0.359 |           0.074 |           0.062 |    RuntimeError |
| BenchmarkTensorRT JIT FP32                  |          1.566 |           0.279 |           0.226 |           0.203 |
| BenchmarkTensorRT JIT FP16                  |          1.564 |           0.292 |           0.223 |           0.212 |
| BenchmarkTensorPTQ GPU INT8                 |          0.289 |           0.044 |           0.033 |    RuntimeError |
| BenchmarkTensorPTQ JIT FP32                 |          1.585 |           0.299 |           0.223 |           0.208 |
| BenchmarkTensorDynamicQuantization CPU INT8 |          7.569 |           4.119 |           3.965 |           4.286 |
| BenchmarkONNX CPU FP32                      |          7.887 |           7.84  |           9.858 |           8.826 |
| BenchmarkONNX GPU FP32                      |          1.254 |           0.342 |           0.306 |           0.339 |

GPU Memory Peak usage [MB] - max_memory_allocated
|                                             |   batch size 1 |   batch size 16 |   batch size 32 |   batch size 64 |
|:--------------------------------------------|---------------:|----------------:|----------------:|----------------:|
| BenchmarkCPU FP32                           |        214.562 |          69.875 |         29.6875 |         35.0625 |
| BenchmarkCPU JIT FP32                       |        115.562 |          23.375 |         78.875  |        106.75   |
| BenchmarkCUDA FP32                          |       4098.44  |        4027     |       4255.31   |       4287      |
| BenchmarkCUDA FP16                          |       3899.88  |        3731     |       3781.94   |       3790.19   |
| BenchmarkCUDA JIT FP32                      |       3202.06  |        3168     |       3239.12   |       3318      |
| BenchmarkTensorRT FP32                      |       4268.38  |        4543.94  |       4262.31   |    RuntimeError |
| BenchmarkTensorRT FP16                      |       4219     |        4031.88  |       4145.12   |    RuntimeError |
| BenchmarkTensorRT JIT FP32                  |       5245     |        4115.38  |       4081.44   |       4306.12   |
| BenchmarkTensorRT JIT FP16                  |       5071.31  |        4156.44  |       4185      |       4359.12   |
| BenchmarkTensorPTQ GPU INT8                 |       4201.06  |        4031.31  |       3663      |    RuntimeError |
| BenchmarkTensorPTQ JIT FP32                 |       4286.5   |        4110.75  |       4177.12   |       4314.12   |
| BenchmarkTensorDynamicQuantization CPU INT8 |         83     |           0     |         65.75   |          7.25   |
| BenchmarkONNX CPU FP32                      |        183.188 |         123.812 |          0      |         20.5    |
| BenchmarkONNX GPU FP32                      |       1555     |        1708.06  |       1873      |       2036.06   |

F1 score
|                                             |   batch size 1 |   batch size 16 |   batch size 32 |   batch size 64 |
|:--------------------------------------------|---------------:|----------------:|----------------:|----------------:|
| BenchmarkCPU FP32                           |          0.691 |           0.691 |           0.691 |           0.691 |
| BenchmarkCPU JIT FP32                       |          0.691 |           0.691 |           0.691 |           0.691 |
| BenchmarkCUDA FP32                          |          0.691 |           0.691 |           0.691 |           0.691 |
| BenchmarkCUDA FP16                          |          0.69  |           0.69  |           0.69  |           0.69  |
| BenchmarkCUDA JIT FP32                      |          0.691 |           0.691 |           0.691 |           0.691 |
| BenchmarkTensorRT FP32                      |          0.691 |           0.691 |           0.691 |    RuntimeError |
| BenchmarkTensorRT FP16                      |          0.691 |           0.69  |           0.691 |    RuntimeError |
| BenchmarkTensorRT JIT FP32                  |          0.691 |           0.691 |           0.691 |           0.691 |
| BenchmarkTensorRT JIT FP16                  |          0.691 |           0.691 |           0.691 |           0.691 |
| BenchmarkTensorPTQ GPU INT8                 |          0.687 |           0.691 |           0.687 |    RuntimeError |
| BenchmarkTensorPTQ JIT FP32                 |          0.691 |           0.691 |           0.691 |           0.691 |
| BenchmarkTensorDynamicQuantization CPU INT8 |          0.691 |           0.691 |           0.691 |           0.691 |
| BenchmarkONNX CPU FP32                      |          0.691 |           0.691 |           0.691 |           0.691 |
| BenchmarkONNX GPU FP32                      |          0.691 |           0.691 |           0.691 |           0.691 |

</details>

<details>
<summary>Custom FCN</summary>

Inference time [ms/sample]
|                                             |   batch size 1 |   batch size 16 |   batch size 32 |   batch size 64 |
|:--------------------------------------------|---------------:|----------------:|----------------:|----------------:|
| BenchmarkCPU FP32                           |          3.538 |           0.481 |           0.381 |           0.396 |
| BenchmarkCPU JIT FP32                       |          7.794 |           0.72  |           0.53  |           0.517 |
| BenchmarkCUDA FP32                          |          0.254 |           0.02  |           0.012 |           0.01  |
| BenchmarkCUDA FP16                          |          0.194 |           0.014 |           0.01  |           0.006 |
| BenchmarkCUDA JIT FP32                      |          0.268 |           0.018 |           0.017 |           0.014 |
| BenchmarkCUDA JIT FP16                      |          0.183 |           0.02  |           0.015 |           0.012 |
| BenchmarkTensorRT FP32                      |          0.322 |           0.028 |           0.017 |           0.011 |
| BenchmarkTensorRT FP16                      |          0.249 |           0.023 |           0.015 |           0.011 |
| BenchmarkTensorRT JIT FP32                  |          0.332 |           0.026 |           0.016 |           0.011 |
| BenchmarkTensorRT JIT FP16                  |          0.243 |           0.022 |           0.014 |           0.01  |
| BenchmarkTensorPTQ GPU INT8                 |          0.319 |           0.026 |           0.016 |           0.011 |
| BenchmarkTensorPTQ JIT FP32                 |          0.325 |           0.026 |           0.017 |           0.011 |
| BenchmarkTensorDynamicQuantization CPU INT8 |          3.563 |           0.417 |           0.383 |           0.407 |
| BenchmarkONNX CPU FP32                      |          2.217 |           0.324 |           0.311 |           0.222 |
| BenchmarkONNX GPU FP32                      |          0.392 |           0.097 |           0.11  |           0.138 |

GPU Memory Peak usage [MB] - max_memory_allocated
|                                             |   batch size 1 |   batch size 16 |   batch size 32 |   batch size 64 |
|:--------------------------------------------|---------------:|----------------:|----------------:|----------------:|
| BenchmarkCPU FP32                           |        91.875  |         129.438 |            0    |          0      |
| BenchmarkCPU JIT FP32                       |       151.312  |         133.312 |            0    |        117.875  |
| BenchmarkCUDA FP32                          |      3940.06   |        3733     |         3735    |       3587.44   |
| BenchmarkCUDA FP16                          |      3925.94   |        3799     |         3814.25 |       3748.81   |
| BenchmarkCUDA JIT FP32                      |      2969.06   |        2656     |         2707.44 |       2483.56   |
| BenchmarkCUDA JIT FP16                      |      3094.5    |        2860.69  |         2798.69 |       2699.81   |
| BenchmarkTensorRT FP32                      |      4478.5    |        4224.75  |         4264.5  |       3170.56   |
| BenchmarkTensorRT FP16                      |      4318.69   |        4092.56  |         1737.44 |       1886.56   |
| BenchmarkTensorRT JIT FP32                  |      4519.88   |        3295     |         3442.38 |       4222.88   |
| BenchmarkTensorRT JIT FP16                  |      4624.5    |        4327.38  |         1918.94 |       1904.5    |
| BenchmarkTensorPTQ GPU INT8                 |      4362.88   |        4230.44  |         1995.12 |       2061.88   |
| BenchmarkTensorPTQ JIT FP32                 |      4555.56   |        4356.56  |         4432.25 |       2615.31   |
| BenchmarkTensorDynamicQuantization CPU INT8 |        79.0625 |           0     |            0    |         30.6875 |
| BenchmarkONNX CPU FP32                      |       145.562  |           4     |           17    |         37.125  |
| BenchmarkONNX GPU FP32                      |      1608.81   |        1755     |         1842    |       1746.69   |

</details>

<details>
<summary>Custom CNN</summary>

Inference time [ms/sample]
|                                             |   batch size 1 |   batch size 16 |   batch size 32 |   batch size 64 |
|:--------------------------------------------|---------------:|----------------:|----------------:|----------------:|
| BenchmarkCPU FP32                           |          0.638 |           0.511 |           0.481 |           0.615 |
| BenchmarkCPU JIT FP32                       |          0.57  |           0.634 |           0.633 |           0.818 |
| BenchmarkCUDA FP32                          |          0.411 |           0.07  |           0.057 |           0.062 |
| BenchmarkCUDA FP16                          |          0.38  |           0.101 |           0.095 |           0.104 |
| BenchmarkCUDA JIT FP32                      |          0.304 |           0.057 |           0.051 |           0.063 |
| BenchmarkCUDA JIT FP16                      |          0.345 |           0.061 |           0.048 |           0.062 |
| BenchmarkTensorRT FP32                      |          0.197 |           0.033 |           0.029 |           0.028 |
| BenchmarkTensorRT FP16                      |          0.188 |           0.025 |           0.019 |           0.018 |
| BenchmarkTensorRT JIT FP32                  |          0.349 |           0.061 |           0.051 |           0.055 |
| BenchmarkTensorRT JIT FP16                  |          0.348 |           0.058 |           0.055 |           0.054 |
| BenchmarkTensorPTQ GPU INT8                 |          0.191 |           0.027 |           0.022 |           0.02  |
| BenchmarkTensorPTQ JIT FP32                 |          0.357 |           0.059 |           0.059 |           0.054 |
| BenchmarkTensorDynamicQuantization CPU INT8 |          0.674 |           0.425 |           0.517 |           0.59  |
| BenchmarkONNX CPU FP32                      |          0.595 |           0.515 |           0.54  |           0.634 |
| BenchmarkONNX GPU FP32                      |          0.368 |           0.131 |           0.156 |           0.179 |

GPU Memory Peak usage [MB] - max_memory_allocated
|                                             |   batch size 1 |   batch size 16 |   batch size 32 |   batch size 64 |
|:--------------------------------------------|---------------:|----------------:|----------------:|----------------:|
| BenchmarkCPU FP32                           |        20      |          4.5625 |          9.875  |         26.0625 |
| BenchmarkCPU JIT FP32                       |       106.375  |          5.3125 |         98.3125 |          0      |
| BenchmarkCUDA FP32                          |      4043.81   |       3854.25   |       3876.88   |       3879      |
| BenchmarkCUDA FP16                          |      3722.81   |       3623      |       3669.81   |       3591      |
| BenchmarkCUDA JIT FP32                      |      3279.25   |       3072.88   |       3165.88   |       3100      |
| BenchmarkCUDA JIT FP16                      |      3244.56   |       3115.06   |       3153.69   |       3102      |
| BenchmarkTensorRT FP32                      |      4170.62   |       3098.88   |       4025      |       4063.19   |
| BenchmarkTensorRT FP16                      |      4248.88   |       3650.94   |       1684.81   |       1584.44   |
| BenchmarkTensorRT JIT FP32                  |      4135      |       4061.12   |       4069.69   |       4004.38   |
| BenchmarkTensorRT JIT FP16                  |      4245.25   |       3994      |       4066.19   |       4004.31   |
| BenchmarkTensorPTQ GPU INT8                 |      4195.25   |       2309.12   |       1804.12   |       3407      |
| BenchmarkTensorPTQ JIT FP32                 |      4148.88   |       4003.12   |       4497.62   |       4005      |
| BenchmarkTensorDynamicQuantization CPU INT8 |         4.9375 |          0      |          0      |         57.0625 |
| BenchmarkONNX CPU FP32                      |        30.4375 |          0      |          0      |          1.6875 |
| BenchmarkONNX GPU FP32                      |      1414.38   |       1513      |       1587      |       1815.12   |

</details>

<details>
<summary>Custom LSTM</summary>

Inference time [ms/sample]
|                                             |   batch size 1 |   batch size 16 |   batch size 32 |   batch size 64 |
|:--------------------------------------------|---------------:|----------------:|----------------:|----------------:|
| BenchmarkCPU FP32                           |         10.481 |           5.14  |           2.605 |           1.518 |
| BenchmarkCPU JIT FP32                       |         12.097 |           4.917 |           2.592 |           1.523 |
| BenchmarkCUDA FP32                          |          3.343 |           0.226 |           0.117 |           0.062 |
| BenchmarkCUDA FP16                          |          4.386 |           0.275 |           0.144 |           0.077 |
| BenchmarkCUDA JIT FP32                      |          4.906 |           0.322 |           0.161 |           0.083 |
| BenchmarkCUDA JIT FP16                      |          4.351 |           0.271 |           0.138 |           0.078 |
| BenchmarkTensorRT FP32                      |          3.668 |           0.233 |           0.123 |           0.071 |
| BenchmarkTensorRT FP16                      |          3.404 |           0.253 |           0.124 |           0.071 |
| BenchmarkTensorRT JIT FP32                  |          5.111 |           0.345 |           0.174 |           0.096 |
| BenchmarkTensorRT JIT FP16                  |          5.013 |           0.355 |           0.174 |           0.094 |
| BenchmarkTensorDynamicQuantization CPU INT8 |          9.749 |           5.139 |           2.619 |           1.555 |
| BenchmarkONNX CPU FP32                      |          3.772 |    RuntimeError |    RuntimeError |    RuntimeError |
| BenchmarkONNX GPU FP32                      |          5.063 |    RuntimeError |    RuntimeError |    RuntimeError |

GPU Memory Peak usage [MB] - max_memory_allocated
|                                             |   batch size 1 |   batch size 16 |   batch size 32 |   batch size 64 |
|:--------------------------------------------|---------------:|----------------:|----------------:|----------------:|
| BenchmarkCPU FP32                           |        97.1875 |         85.125  |         52.9375 |         94.1875 |
| BenchmarkCPU JIT FP32                       |       162.75   |          1.3125 |         31.0625 |        120.938  |
| BenchmarkCUDA FP32                          |      4115.25   |       3966      |       3991      |       3943.06   |
| BenchmarkCUDA FP16                          |      4227.44   |       3950.31   |       4031.56   |       3861.81   |
| BenchmarkCUDA JIT FP32                      |      3368.44   |       3165.31   |       3158      |       3098      |
| BenchmarkCUDA JIT FP16                      |      3221.06   |       3052.25   |       3039.88   |       3034      |
| BenchmarkTensorRT FP32                      |      4378.94   |       4093.56   |       4170.94   |       4146.5    |
| BenchmarkTensorRT FP16                      |      4369.19   |       4169.06   |       4183.12   |       4149.31   |
| BenchmarkTensorRT JIT FP32                  |      4442.5    |       4195.44   |       4242.06   |       4219.06   |
| BenchmarkTensorRT JIT FP16                  |      4442.38   |       4282.5    |       4245.12   |       4218.75   |
| BenchmarkTensorDynamicQuantization CPU INT8 |        96.1875 |          0      |         53.625  |        105.562  |
| BenchmarkONNX CPU FP32                      |        18.8125 |    RuntimeError |    RuntimeError |    RuntimeError |
| BenchmarkONNX GPU FP32                      |      2308.31   |    RuntimeError |    RuntimeError |    RuntimeError |

</details>

<details>
<summary>BERT</summary>

Inference time [ms/sample]
|                                             |   batch size 1 |   batch size 16 |   batch size 32 |   batch size 64 |
|:--------------------------------------------|---------------:|----------------:|----------------:|----------------:|
| BenchmarkCUDA FP32                          |          5.219 |           1.256 |           1.204 |           1.022 |
| BenchmarkCUDA FP16                          |          5.87  |           0.523 |           0.512 |           0.39  |
| BenchmarkCUDA JIT FP32                      |          4.61  |           1.737 |           1.062 |           0.894 |
| BenchmarkCUDA JIT FP16                      |          5.352 |           1.423 |           0.454 |           0.372 |
| BenchmarkTensorDynamicQuantization CPU INT8 |         50.772 |          38.479 |          37.276 |          35.526 |

GPU Memory Peak usage [MB] - max_memory_allocated
|                                             |   batch size 1 |   batch size 16 |   batch size 32 |   batch size 64 |
|:--------------------------------------------|---------------:|----------------:|----------------:|----------------:|
| BenchmarkCUDA FP32                          |       1677.25  |        1671     |        1765.25  |        1909.06  |
| BenchmarkCUDA FP16                          |       1759.69  |        1262.62  |        1229     |        2226.31  |
| BenchmarkCUDA JIT FP32                      |        323.875 |         561.25  |         538     |         632.312 |
| BenchmarkCUDA JIT FP16                      |        444     |         476.688 |         617.875 |         800     |
| BenchmarkTensorDynamicQuantization CPU INT8 |          5.25  |           8     |         418.375 |          61.875 |

F1 score
|                                             |   batch size 1 |   batch size 16 |   batch size 32 |   batch size 64 |
|:--------------------------------------------|---------------:|----------------:|----------------:|----------------:|
| BenchmarkCUDA FP32                          |          0.867 |           0.867 |           0.867 |           0.867 |
| BenchmarkCUDA FP16                          |          0.867 |           0.867 |           0.867 |           0.867 |
| BenchmarkCUDA JIT FP32                      |          0.867 |           0.867 |           0.867 |           0.867 |
| BenchmarkCUDA JIT FP16                      |          0.867 |           0.867 |           0.867 |           0.867 |
| BenchmarkTensorDynamicQuantization CPU INT8 |          0.867 |           0.867 |           0.867 |           0.867 |

</details>

<details>
<summary>GPTNeo</summary>

Inference time [ms/sample]
|                                             |   batch size 1 |   batch size 16 |   batch size 32 |   batch size 64 |
|:--------------------------------------------|---------------:|----------------:|----------------:|----------------:|
| BenchmarkCUDA FP32                          |        774.454 |          97.448 |          72.084 |     OutOfMemory |
| BenchmarkCUDA FP16                          |        853.397 |         100.885 |          73.057 |          51.257 |
| BenchmarkTensorDynamicQuantization CPU INT8 |       4277.05  |     OutOfMemory |     OutOfMemory |     OutOfMemory |

GPU Memory Peak usage [MB] - max_memory_allocated
|                                             |   batch size 1 |   batch size 16 |   batch size 32 |   batch size 64 |
|:--------------------------------------------|---------------:|----------------:|----------------:|----------------:|
| BenchmarkCUDA FP32                          |      2234.69   |         7977    |        10455.1  |     OutOfMemory |
| BenchmarkCUDA FP16                          |      2422.56   |         3851.31 |         6490.81 |         9843.62 |
| BenchmarkTensorDynamicQuantization CPU INT8 |        22.5625 |     OutOfMemory |     OutOfMemory |     OutOfMemory |

</details>
