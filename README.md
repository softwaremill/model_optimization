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

Inference time [ms/batch]
|                                             |   batch size 1 |   batch size 16 |   batch size 32 |   batch size 64 |
|:--------------------------------------------|---------------:|----------------:|----------------:|----------------:|
| BenchmarkCPU FP32                           |        6.65668 |        39.6939  |        93.6553  |       258.536   |
| BenchmarkCPU JIT FP32                       |        4.15924 |        33.0095  |        84.7393  |       215.743   |
| BenchmarkCUDA FP32                          |        4.73794 |         4.90756 |         4.98441 |         5.13826 |
| BenchmarkCUDA FP16                          |        4.24022 |         8.92531 |        13.3196  |        22.2075  |
| BenchmarkCUDA JIT FP32                      |        2.50684 |         2.57411 |         2.6444  |         2.66859 |
| BenchmarkTensorRT FP32                      |        0.37369 |         0.46744 |         0.55616 |         0.64345 |
| BenchmarkTensorRT FP16                      |        0.36989 |         0.44853 |         0.48069 |         0.65303 |
| BenchmarkTensorRT JIT FP32                  |        1.86556 |         2.29273 |         2.72782 |         3.42388 |
| BenchmarkTensorRT JIT FP16                  |        1.8543  |         2.51738 |         2.69553 |         3.50059 |
| BenchmarkTensorPTQ GPU INT8                 |        0.36075 |         0.47061 |         0.47071 |         0.59932 |
| BenchmarkTensorDynamicQuantization CPU INT8 |        6.52208 |        33.8213  |        90.519   |       248.511   |
| BenchmarkONNX CPU FP32                      |       13.1123  |       143.642   |       296.045   |       631.377   |
| BenchmarkONNX GPU FP32                      |        1.40741 |         5.55726 |        11.0792  |        23.09    |

GPU Memory Peak usage [MB] - max_memory_allocated
|                                             |   batch size 1 |   batch size 16 |   batch size 32 |   batch size 64 |
|:--------------------------------------------|---------------:|----------------:|----------------:|----------------:|
| BenchmarkCPU FP32                           |        86.5625 |          93.625 |         187.75  |        132.562  |
| BenchmarkCPU JIT FP32                       |        90.25   |          38     |         106.125 |        196.25   |
| BenchmarkCUDA FP32                          |      4280.06   |        4056.5   |        4219     |       4332.44   |
| BenchmarkCUDA FP16                          |      3738.62   |        3740.75  |        3775     |       3955.56   |
| BenchmarkCUDA JIT FP32                      |      4180.12   |        3463.25  |        3293     |       4242.5    |
| BenchmarkTensorRT FP32                      |      4191.25   |        4037.25  |        4234.88  |       4468.5    |
| BenchmarkTensorRT FP16                      |      4098.94   |        4109.94  |        4036.5   |       4096.88   |
| BenchmarkTensorRT JIT FP32                  |      4888.31   |        4367.88  |        4634.06  |       4869.31   |
| BenchmarkTensorRT JIT FP16                  |      4756.5    |        4511.44  |        4549     |       4406.38   |
| BenchmarkTensorPTQ GPU INT8                 |      4278      |        3988.62  |        4138.75  |       4129.12   |
| BenchmarkTensorDynamicQuantization CPU INT8 |         0      |         111.5   |         727.812 |         42.5625 |
| BenchmarkONNX CPU FP32                      |        70.25   |         164     |         133.062 |         69.375  |
| BenchmarkONNX GPU FP32                      |      1565.31   |        1748.44  |        2104.38  |       1972.69   |

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

Inference time [ms/batch]
|                                             |   batch size 1 |   batch size 16 |   batch size 32 |   batch size 64 |
|:--------------------------------------------|---------------:|----------------:|----------------:|----------------:|
| BenchmarkCPU FP32                           |        6.56187 |        66.63    |       148.69    |       310.237   |
| BenchmarkCPU JIT FP32                       |        3.60982 |        58.6944  |       115.064   |       264.537   |
| BenchmarkCUDA FP32                          |        2.11525 |         2.04019 |         2.1531  |         2.20068 |
| BenchmarkCUDA FP16                          |        1.96927 |         4.98496 |         8.59432 |        21.8313  |
| BenchmarkCUDA JIT FP32                      |        1.36742 |         1.37558 |         1.38043 |         1.38113 |
| BenchmarkTensorRT FP32                      |        0.18209 |         0.234   |         0.29882 |    RuntimeError |
| BenchmarkTensorRT FP16                      |        0.17422 |         0.24152 |         0.32649 |    RuntimeError |
| BenchmarkTensorRT JIT FP32                  |        1.42797 |         1.92369 |         2.40061 |         3.36516 |
| BenchmarkTensorRT JIT FP16                  |        1.43552 |         1.93637 |         2.40309 |         3.35638 |
| BenchmarkTensorPTQ GPU INT8                 |        0.17013 |         0.24272 |         0.28632 |    RuntimeError |
| BenchmarkTensorPTQ JIT FP32                 |        1.45247 |         1.96525 |         2.36605 |         3.32073 |
| BenchmarkTensorDynamicQuantization CPU INT8 |        7.10821 |        66.5728  |       152.422   |       310.828   |
| BenchmarkONNX CPU FP32                      |        7.57494 |       126.287   |       288.258   |       590.629   |
| BenchmarkONNX GPU FP32                      |        1.34674 |         5.33583 |         9.58347 |        22.3498  |

GPU Memory Peak usage [MB] - max_memory_allocated
|                                             |   batch size 1 |   batch size 16 |   batch size 32 |   batch size 64 |
|:--------------------------------------------|---------------:|----------------:|----------------:|----------------:|
| BenchmarkCPU FP32                           |        175.562 |         240.438 |         111.938 |           0     |
| BenchmarkCPU JIT FP32                       |         16     |         157.062 |           1.5   |         343.125 |
| BenchmarkCUDA FP32                          |       4164     |        4035.5   |        4155     |        4287     |
| BenchmarkCUDA FP16                          |       3880.31  |        3740.5   |        3799.06  |        3781.31  |
| BenchmarkCUDA JIT FP32                      |       3333.38  |        3325.06  |        3254     |        3318     |
| BenchmarkTensorRT FP32                      |       4465.5   |        4139.81  |        4468.31  |    RuntimeError |
| BenchmarkTensorRT FP16                      |       4239.19  |        4112.06  |        4163.81  |    RuntimeError |
| BenchmarkTensorRT JIT FP32                  |       5306.94  |        4228.81  |        4185     |        4315     |
| BenchmarkTensorRT JIT FP16                  |       4326.75  |        4193.31  |        4269     |        4315     |
| BenchmarkTensorPTQ GPU INT8                 |       4492.75  |        4019.69  |        2984.75  |    RuntimeError |
| BenchmarkTensorPTQ JIT FP32                 |       4362.19  |        4109.19  |        4161.81  |        4305     |
| BenchmarkTensorDynamicQuantization CPU INT8 |        175.438 |         119.438 |           4     |           0     |
| BenchmarkONNX CPU FP32                      |        119.625 |           5.625 |         129.125 |           0     |
| BenchmarkONNX GPU FP32                      |       1794.25  |        1669.25  |        1683     |        2058.81  |

F1 score
|                                             |   batch size 1 |   batch size 16 |   batch size 32 |   batch size 64 |
|:--------------------------------------------|---------------:|----------------:|----------------:|----------------:|
| BenchmarkCPU FP32                           |          0.691 |           0.691 |           0.691 |           0.691 |
| BenchmarkCPU JIT FP32                       |          0.691 |           0.691 |           0.691 |           0.691 |
| BenchmarkCUDA FP32                          |          0.691 |           0.691 |           0.691 |           0.691 |
| BenchmarkCUDA FP16                          |          0.69  |           0.69  |           0.69  |           0.69  |
| BenchmarkCUDA JIT FP32                      |          0.691 |           0.691 |           0.691 |           0.691 |
| BenchmarkTensorRT FP32                      |          0.691 |           0.691 |           0.691 |    RuntimeError |
| BenchmarkTensorRT FP16                      |          0.69  |           0.691 |           0.691 |    RuntimeError |
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

Inference time [ms/batch]
|                                             |   batch size 1 |   batch size 16 |   batch size 32 |   batch size 64 |
|:--------------------------------------------|---------------:|----------------:|----------------:|----------------:|
| BenchmarkCPU FP32                           |        3.64144 |         7.21241 |        12.6852  |        26.349   |
| BenchmarkCPU JIT FP32                       |        8.344   |        11.5492  |        18.1966  |        34.0335  |
| BenchmarkCUDA FP32                          |        0.13043 |         0.14393 |         0.14425 |         0.1537  |
| BenchmarkCUDA FP16                          |        0.16081 |         0.15058 |         0.15591 |         0.17274 |
| BenchmarkCUDA JIT FP32                      |        0.11584 |         0.11744 |         0.30389 |         0.50184 |
| BenchmarkCUDA JIT FP16                      |        0.14448 |         0.23365 |         0.34274 |         0.51351 |
| BenchmarkTensorRT FP32                      |        0.11961 |         0.18326 |         0.21406 |         0.31011 |
| BenchmarkTensorRT FP16                      |        0.12771 |         0.18176 |         0.22592 |         0.30709 |
| BenchmarkTensorRT JIT FP32                  |        0.12059 |         0.17736 |         0.21766 |         0.30517 |
| BenchmarkTensorRT JIT FP16                  |        0.13034 |         0.17527 |         0.22913 |         0.31921 |
| BenchmarkTensorPTQ GPU INT8                 |        0.11739 |         0.18039 |         0.21735 |         0.31279 |
| BenchmarkTensorPTQ JIT FP32                 |        0.122   |         0.17064 |         0.22594 |         0.32175 |
| BenchmarkTensorDynamicQuantization CPU INT8 |        3.74147 |         7.7936  |        12.7667  |        25.6249  |
| BenchmarkONNX CPU FP32                      |        2.48965 |         6.26884 |        10.3026  |        15.9953  |
| BenchmarkONNX GPU FP32                      |        0.3746  |         1.58999 |         3.43263 |         8.43945 |

GPU Memory Peak usage [MB] - max_memory_allocated
|                                             |   batch size 1 |   batch size 16 |   batch size 32 |   batch size 64 |
|:--------------------------------------------|---------------:|----------------:|----------------:|----------------:|
| BenchmarkCPU FP32                           |         2.6875 |           6.125 |            0    |           0     |
| BenchmarkCPU JIT FP32                       |         7.125  |           0     |            0    |           0     |
| BenchmarkCUDA FP32                          |      3869      |        3557.44  |         3596.94 |        3385     |
| BenchmarkCUDA FP16                          |      3935      |        3686.56  |         3679    |        3526.5   |
| BenchmarkCUDA JIT FP32                      |      2971.12   |        2842     |         2566    |        2567.44  |
| BenchmarkCUDA JIT FP16                      |      3044      |        2916.12  |         2920    |        2522.06  |
| BenchmarkTensorRT FP32                      |      4417      |        4242.94  |         2483    |        1843     |
| BenchmarkTensorRT FP16                      |      4355      |        4189     |         2820.88 |        1757     |
| BenchmarkTensorRT JIT FP32                  |      4549      |        4374.94  |         4357.75 |        3681     |
| BenchmarkTensorRT JIT FP16                  |      4514.56   |        2403.12  |         1879    |        1889     |
| BenchmarkTensorPTQ GPU INT8                 |      4382.75   |        4242.94  |         1999    |        3476.56  |
| BenchmarkTensorPTQ JIT FP32                 |      4680.69   |        4365.12  |         4472    |        2100.25  |
| BenchmarkTensorDynamicQuantization CPU INT8 |         4.1875 |           0     |            0    |          69.875 |
| BenchmarkONNX CPU FP32                      |         6.125  |           6.125 |            0    |           0     |
| BenchmarkONNX GPU FP32                      |      1743.25   |        1747     |         1753    |        1745     |

</details>

<details>
<summary>Custom CNN</summary>

Inference time [ms/batch]
|                                             |   batch size 1 |   batch size 16 |   batch size 32 |   batch size 64 |
|:--------------------------------------------|---------------:|----------------:|----------------:|----------------:|
| BenchmarkCPU FP32                           |        0.6041  |         6.84558 |        15.2199  |        36.631   |
| BenchmarkCPU JIT FP32                       |        0.55452 |         9.20548 |        17.5044  |        50.8534  |
| BenchmarkCUDA FP32                          |        0.36863 |         0.39096 |         0.40863 |         0.44231 |
| BenchmarkCUDA FP16                          |        0.32376 |         0.90419 |         1.43579 |         2.59567 |
| BenchmarkCUDA JIT FP32                      |        0.25777 |         0.31822 |         0.30812 |         1.04666 |
| BenchmarkCUDA JIT FP16                      |        0.29276 |         0.40917 |         0.34614 |         1.06159 |
| BenchmarkTensorRT FP32                      |        0.14719 |         0.19721 |         0.23885 |         0.32795 |
| BenchmarkTensorRT FP16                      |        0.13708 |         0.19832 |         0.24555 |         0.31692 |
| BenchmarkTensorRT JIT FP32                  |        0.29644 |         0.37561 |         0.43972 |         0.55849 |
| BenchmarkTensorRT JIT FP16                  |        0.3038  |         0.40303 |         0.4536  |         0.53448 |
| BenchmarkTensorPTQ GPU INT8                 |        0.14049 |         0.19181 |         0.23541 |         0.33495 |
| BenchmarkTensorPTQ JIT FP32                 |        0.29664 |         0.39762 |         0.47142 |         0.57725 |
| BenchmarkTensorDynamicQuantization CPU INT8 |        0.62132 |         6.90235 |        14.6365  |        37.6674  |
| BenchmarkONNX CPU FP32                      |        0.5589  |         6.61945 |        15.4145  |        44.824   |
| BenchmarkONNX GPU FP32                      |        0.32198 |         2.01974 |         4.45235 |        10.8827  |

GPU Memory Peak usage [MB] - max_memory_allocated
|                                             |   batch size 1 |   batch size 16 |   batch size 32 |   batch size 64 |
|:--------------------------------------------|---------------:|----------------:|----------------:|----------------:|
| BenchmarkCPU FP32                           |          7.125 |          4      |           6.375 |         143.625 |
| BenchmarkCPU JIT FP32                       |         13.625 |         25.625  |           0     |         181.5   |
| BenchmarkCUDA FP32                          |       4012.88  |       3867      |        3903     |        3900.5   |
| BenchmarkCUDA FP16                          |       3743.5   |       3627      |        3633     |        3585.12  |
| BenchmarkCUDA JIT FP32                      |       3240.38  |       3109      |        3144     |        3050.06  |
| BenchmarkCUDA JIT FP16                      |       3238.62  |       3085.25   |        3136     |        3120.69  |
| BenchmarkTensorRT FP32                      |       4129.5   |       2543      |        4039.38  |        4050.56  |
| BenchmarkTensorRT FP16                      |       4251.19  |       4011      |        2887     |        1969     |
| BenchmarkTensorRT JIT FP32                  |       4190.88  |       4042.19   |        4057     |        4003     |
| BenchmarkTensorRT JIT FP16                  |       4107.06  |       4029      |        4049.94  |        3917.31  |
| BenchmarkTensorPTQ GPU INT8                 |       4070.38  |       2695.69   |        2527     |        3905.19  |
| BenchmarkTensorPTQ JIT FP32                 |       4123.12  |       3816.94   |        4057     |        3996.62  |
| BenchmarkTensorDynamicQuantization CPU INT8 |          0     |          0      |           0     |          32     |
| BenchmarkONNX CPU FP32                      |          0     |          5.0625 |           0     |           0     |
| BenchmarkONNX GPU FP32                      |       1425.88  |       1497.88   |        1320.69  |        1841.88  |

</details>

<details>
<summary>Custom LSTM</summary>

Inference time [ms/batch]
|                                             |   batch size 1 |   batch size 16 |   batch size 32 |   batch size 64 |
|:--------------------------------------------|---------------:|----------------:|----------------:|----------------:|
| BenchmarkCPU FP32                           |        9.73442 |        79.3966  |        87.3652  |        96.4714  |
| BenchmarkCPU JIT FP32                       |        9.24112 |        83.0199  |        81.9827  |        97.5167  |
| BenchmarkCUDA FP32                          |        0.6312  |         0.62433 |         0.85634 |         0.87198 |
| BenchmarkCUDA FP16                          |        2.38455 |         2.44246 |         2.60007 |         2.57795 |
| BenchmarkCUDA JIT FP32                      |        2.19351 |         2.23447 |         2.47491 |         2.40318 |
| BenchmarkCUDA JIT FP16                      |        2.36462 |         2.34376 |         2.48879 |         2.60825 |
| BenchmarkTensorRT FP32                      |        0.76575 |         0.88835 |         1.15607 |         1.42448 |
| BenchmarkTensorRT FP16                      |        0.76033 |         0.91771 |         1.17992 |         1.39111 |
| BenchmarkTensorRT JIT FP32                  |        2.38293 |         2.53469 |         2.70416 |         3.00983 |
| BenchmarkTensorRT JIT FP16                  |        2.41481 |         2.50169 |         2.82023 |         3.01824 |
| BenchmarkTensorDynamicQuantization CPU INT8 |        9.71816 |        79.2601  |        83.2324  |        93.6791  |
| BenchmarkONNX CPU FP32                      |        3.68844 |    RuntimeError |    RuntimeError |    RuntimeError |
| BenchmarkONNX GPU FP32                      |        4.91169 |    RuntimeError |    RuntimeError |    RuntimeError |

GPU Memory Peak usage [MB] - max_memory_allocated
|                                             |   batch size 1 |   batch size 16 |   batch size 32 |   batch size 64 |
|:--------------------------------------------|---------------:|----------------:|----------------:|----------------:|
| BenchmarkCPU FP32                           |       105.375  |         179.25  |         11.875  |          32     |
| BenchmarkCPU JIT FP32                       |         0.0625 |         182.125 |         56.9375 |           7.625 |
| BenchmarkCUDA FP32                          |      4141.81   |        4078.44  |       3993.12   |        3934.56  |
| BenchmarkCUDA FP16                          |      4292.75   |        3878.88  |       3979.56   |        3913.81  |
| BenchmarkCUDA JIT FP32                      |      3688.5    |        3252.06  |       3170.69   |        3098     |
| BenchmarkCUDA JIT FP16                      |      3220.25   |        3055.44  |       3033.88   |        3098.5   |
| BenchmarkTensorRT FP32                      |      4438.12   |        4012.06  |       4173.12   |        3804     |
| BenchmarkTensorRT FP16                      |      4277.88   |        4275.56  |       4177.25   |        4125     |
| BenchmarkTensorRT JIT FP32                  |      4355.19   |        4189.06  |       4257.81   |        4239.75  |
| BenchmarkTensorRT JIT FP16                  |      4456.12   |        4222.75  |       4250.88   |        4213     |
| BenchmarkTensorDynamicQuantization CPU INT8 |       186.562  |          67     |        124.375  |           0     |
| BenchmarkONNX CPU FP32                      |        36      |    RuntimeError |    RuntimeError |    RuntimeError |
| BenchmarkONNX GPU FP32                      |      2477.38   |    RuntimeError |    RuntimeError |    RuntimeError |

</details>

<details>
<summary>BERT</summary>

Inference time [ms/batch]
|                        |   batch size 1 |   batch size 16 |   batch size 32 |   batch size 64 |
|:-----------------------|---------------:|----------------:|----------------:|----------------:|
| BenchmarkCUDA FP32     |        5.06335 |         5.09667 |         5.55995 |         5.58667 |
| BenchmarkCUDA FP16     |        5.64662 |         5.93464 |         6.03207 |         6.26639 |
| BenchmarkCUDA JIT FP32 |        4.55503 |        12.5667  |         3.00945 |         2.9614  |
| BenchmarkCUDA JIT FP16 |        5.20118 |        17.5239  |         3.25109 |         3.39975 |

GPU Memory Peak usage [MB] - max_memory_allocated
|                        |   batch size 1 |   batch size 16 |   batch size 32 |   batch size 64 |
|:-----------------------|---------------:|----------------:|----------------:|----------------:|
| BenchmarkCUDA FP32     |       1542.19  |        1671     |         1765    |        1916.5   |
| BenchmarkCUDA FP16     |       1733.56  |        1229     |         1220.38 |        1285.75  |
| BenchmarkCUDA JIT FP32 |        321.812 |         438.125 |          538    |         747.062 |
| BenchmarkCUDA JIT FP16 |        472.312 |         502     |          618    |         776.562 |

F1 score
|                        |   batch size 1 |   batch size 16 |   batch size 32 |   batch size 64 |
|:-----------------------|---------------:|----------------:|----------------:|----------------:|
| BenchmarkCUDA FP32     |          0.867 |           0.867 |           0.867 |           0.867 |
| BenchmarkCUDA FP16     |          0.867 |           0.867 |           0.867 |           0.867 |
| BenchmarkCUDA JIT FP32 |          0.867 |           0.867 |           0.867 |           0.867 |
| BenchmarkCUDA JIT FP16 |          0.867 |           0.867 |           0.867 |           0.867 |

</details>

<details>
<summary>GPTNeo</summary>

Inference time [ms/batch]
|                    |   batch size 1 |   batch size 16 |   batch size 32 |   batch size 64 |
|:-------------------|---------------:|----------------:|----------------:|----------------:|
| BenchmarkCUDA FP32 |        711.666 |         1483.83 |         2353.65 |     OutOfMemory |
| BenchmarkCUDA FP16 |        845.133 |         1509.55 |         2339.03 |         3397.37 |

GPU Memory Peak usage [MB] - max_memory_allocated
|                    |   batch size 1 |   batch size 16 |   batch size 32 |   batch size 64 |
|:-------------------|---------------:|----------------:|----------------:|----------------:|
| BenchmarkCUDA FP32 |        2262.44 |         8006.19 |        10565    |     OutOfMemory |
| BenchmarkCUDA FP16 |        2256.62 |         4048.56 |         6491.62 |         9836.19 |

</details>
