# Block Profiling Code

### Note:
The current codebase can perform measurements on metrics included in the original Once-for-All repository (https://github.com/mit-han-lab/once-for-all/tree/master/ofa) and NPU latency. We will be updating this repository as we are able to clear data pertaining to other metrics, e.g. in-house latency predictors, for public release. We will also be updating the `search/rm_search` code as it was originally separate from the profiling code.

### Preliminaries:
Data for available in-house predictors (currently only NPU) must be downloaded and used to generate appropriate `.pt` files before profiling can be done.
1. Download the `.csv` files from [Google Drive](https://drive.google.com/drive/folders/1qUqWzc3D3-1LmRxqQiy6u4CRWsmOymuF) and place them in `search/rm_search/data/`
2. For each `.csv` file, run the following command:
```
python search/rm_search/run_ofa_op_graph_lat_predictor.py -subspace <SPACE> -lat_device <DEVICE>
```
Where `sub_space` is the design space, e.g., `mbv3`, `pn` and `lat_device` is the target device, e.g., `npu`. Doing so should place the appropriate `_best.pt` files in `models/Latency/`.

If you have trouble running `run_ofa_op_graph_lat_predictor.py`, see the corresponding README in `search/rm_search`

### Contents: <br/>

This code consists of two parts:

1. Block Profiling, performed using `main.py`
2. Search experiments, performed using scripts in the `/search/rm_search` directory

### Dependencies
```
python 3.6 or 3.7
pytorch >= 1.4.0
torchvision >= 0.4.0
ofa==0.1.0.post202012082159
torchprofile==0.0.2
```

## Commands to perform profiling

```
python3 main.py
    --space {OFAPred, OFASupernet, ProxylessSupernet, ResNet50Supernet}
    --num_archs 10 # Number of architectures to evaluate per unit-layer-block fix
    --blocks # Can be used to specify individual unit-layer-block combinations to profile, see the block_sample method in each search space for details
    --all # Evaluate all possible unit-layer-block combinations
    --device # Which device, e.g., CPU or GPU, to use
    --metrics # Can specify profiling specific metrics (e.g. only certain latencies), see the metrics field in each file in /search_spaces/ for details
    --data # Location of ImageNet data. Ignored when --fast is specified
    --fast # Use fast RAM imagenet validation data loader, see /models/imagenet_RAM_saver.py for details
    --save # Name of experiment, where to save. Information will be saved in /logs/{Space}/{Save}
    --no-log # Binary; do not log experiment information
```
Output data is formatted to be comma separated such that it can easily be transferred to a CSV file for further processing and analysis if desired, e.g., plotting trends.