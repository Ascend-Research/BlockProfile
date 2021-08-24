## Random mutation search related files

Before starting any search you'll need to make sure:

* The OFA supernet checkpoint files are placed under `<your path to the code>/BlockProfile/.torch/ofa_nets/`.
* Latency predictors are pre-trained and the `.pt` checkpoint file placed under `<your path to the code>/BlockProfile/models/Latency/`.
* ImageNet validation data placed in some folder you know, use the flag `-imagenet_data_dir'` to specify where.
* The flag `-fast` can be used to speed up search time using a RAM-based dataloader, however this requires a computer with a lot of memory (>32GB).

#### To train NPU/GPU/CPU latency predictors

Put the `.csv` latency data into `./data/`.

Next, execute `python -u search/rm_search/run_ofa_op_graph_lat_predictor.py -sub_space <SPACE> -lat_device <DEVICE>` where ere the value for the `-sub_space` flag is the design space, e.g., `mbv3` or `pn` and the value for the `-lat_device` is the target device, e.g., `npu` or `gpu`. Doing so should place the appropriate `_best.pt` files in `models/Latency/`.


#### Available Predictors:
* Samsung Note10 (MBv3)
* Huawei NPU (MBv3 and PN)
* Nvidia RTX 2080 Ti (MBv3 and PN)
* AMD Threadripper 2990WX (MBv3 and PN)

#### Note: 
Run all scripts from the toplevel `/BlockProfile/` directory.

#### To run Pareto front search for the ProxylessNAS design space:

With insight, using **NPU** latency predictor:

    CUDA_VISIBLE_DEVICES=0 python -u search/rm_search/run_ofa_pn_rm_pareto_search.py -batch_size 200 -model_name ofa_pn_rm_npu_insight_pareto -mutate_prob_type npu -space_id npu -resolution 224 -lat_predictor_checkpoint models/Latency/ofa_pn_op_graph_npu_lat_predictor_best.pt

No insight, using **NPU** latency predictor:

    CUDA_VISIBLE_DEVICES=0 python -u search/rm_search/run_ofa_pn_rm_pareto_search.py -batch_size 200 -model_name ofa_pn_rm_npu_full_space_pareto -resolution 224 -lat_predictor_checkpoint models/Latency/ofa_pn_op_graph_npu_lat_predictor_best.pt

With insight, using **GPU** latency predictor:

    CUDA_VISIBLE_DEVICES=0 python -u search/rm_search/run_ofa_pn_rm_pareto_search.py -batch_size 200 -model_name ofa_pn_rm_gpu_insight_pareto -mutate_prob_type gpu -space_id gpu -stage_block_count_type gpu -resolution 224 -lat_predictor_checkpoint models/Latency/ofa_pn_op_graph_gpu_lat_predictor_best.pt

No insight, using **GPU** latency predictor:

    CUDA_VISIBLE_DEVICES=0 python -u search/rm_search/run_ofa_pn_rm_pareto_search.py -batch_size 200 -model_name ofa_pn_rm_gpu_full_space_pareto -resolution 224 -lat_predictor_checkpoint models/Latency/ofa_pn_op_graph_gpu_lat_predictor_best.pt

With insight, using **CPU** latency predictor:

    CUDA_VISIBLE_DEVICES=0 python -u search/rm_search/run_ofa_pn_rm_pareto_search.py -batch_size 200 -model_name ofa_pn_rm_cpu_insight_pareto -mutate_prob_type cpu -space_id cpu -stage_block_count_type cpu -resolution 224 -lat_predictor_checkpoint models/Latency/ofa_pn_op_graph_cpu_lat_predictor_best.pt

No insight, using **CPU** latency predictor:

    CUDA_VISIBLE_DEVICES=0 python -u search/rm_search/run_ofa_pn_rm_pareto_search.py -batch_size 200 -model_name ofa_pn_rm_cpu_full_space_pareto -resolution 224 -lat_predictor_checkpoint models/Latency/ofa_pn_op_graph_cpu_lat_predictor_best.pt

#### To run Pareto front search for the OFA (MobileNetV3) design space:

With insight, using **NPU** latency predictor:

    CUDA_VISIBLE_DEVICES=0 python -u search/rm_search/run_ofa_mbv3_rm_pareto_search.py -batch_size 200 -model_name ofa_mbv3_rm_npu_insight_pareto -mutate_prob_type npu -space_id npu -resolution 224 -lat_predictor_checkpoint models/Latency/ofa_mbv3_op_graph_npu_lat_predictor_best.pt

No insight, using **NPU** latency predictor:

    CUDA_VISIBLE_DEVICES=0 python -u search/rm_search/run_ofa_mbv3_rm_pareto_search.py -batch_size 200 -model_name ofa_mbv3_rm_npu_full_space_pareto -resolution 224 -lat_predictor_checkpoint models/Latency/ofa_mbv3_op_graph_npu_lat_predictor_best.pt

With insight, using **GPU** latency predictor:

    CUDA_VISIBLE_DEVICES=0 python -u search/rm_search/run_ofa_mbv3_rm_pareto_search.py -batch_size 200 -model_name ofa_mbv3_rm_gpu_insight_pareto -mutate_prob_type gpu -space_id gpu -stage_block_count_type gpu -resolution 224 -lat_predictor_checkpoint models/Latency/ofa_mbv3_op_graph_gpu_lat_predictor_best.pt

No insight, using **GPU** latency predictor:

    CUDA_VISIBLE_DEVICES=0 python -u search/rm_search/run_ofa_mbv3_rm_pareto_search.py -batch_size 200 -model_name ofa_mbv3_rm_gpu_full_space_pareto -resolution 224 -lat_predictor_checkpoint models/Latency/ofa_mbv3_op_graph_gpu_lat_predictor_best.pt

With insight, using **CPU** latency predictor:

    CUDA_VISIBLE_DEVICES=0 python -u search/rm_search/run_ofa_mbv3_rm_pareto_search.py -batch_size 200 -model_name ofa_mbv3_rm_cpu_insight_pareto -mutate_prob_type cpu -space_id cpu -stage_block_count_type cpu -resolution 224 -lat_predictor_checkpoint models/Latency/ofa_mbv3_op_graph_cpu_lat_predictor_best.pt

No insight, using **CPU** latency predictor:

    CUDA_VISIBLE_DEVICES=0 python -u search/rm_search/run_ofa_mbv3_rm_pareto_search.py -batch_size 200 -model_name ofa_mbv3_rm_cpu_full_space_pareto -resolution 224 -lat_predictor_checkpoint models/Latency/ofa_mbv3_op_graph_cpu_lat_predictor_best.pt

With insight, using **Note10** latency predictor:

    CUDA_VISIBLE_DEVICES=0 python -u search/rm_search/run_ofa_mbv3_rm_pareto_search.py -batch_size 200 -model_name ofa_mbv3_rm_n10_insight_pareto -mutate_prob_type n10 -space_id n10 -stage_block_count_type n10 -resolution 224 -lat_predictor_type n10

No insight, using **Note10** latency predictor:

    CUDA_VISIBLE_DEVICES=3 python -u search/rm_search/run_ofa_mbv3_rm_pareto_search.py -batch_size 200 -model_name ofa_mbv3_rm_n10_full_space_pareto -resolution 224 -lat_predictor_type n10

Once finished, the Pareto front architectures and their values will be printed in the console.

#### To run Max acc search for the ResNet50 design space:

With insight:

    CUDA_VISIBLE_DEVICES=0 python -u search/rm_search/run_ofa_resnet_rm_cons_acc_search.py -batch_size 200 -seed 1 -model_name ofa_resnet_rm_insight_max_acc -space_id max_acc -max_stage_block_count_type max_acc -min_stage_block_count_type max_acc -resolution 224

No insight:

    CUDA_VISIBLE_DEVICES=0 python -u search/rm_search/run_ofa_resnet_rm_cons_acc_search.py -batch_size 200 -seed 1 -model_name ofa_resnet_rm_full_space_max_acc -resolution 224
    
Once finished, the top-10 architectures and their accuracy values will be printed in the console.

#### FAQ
* If you are getting an error like this: `ModuleNotFoundError: No module named 'search'`. 
Add `export PYTHONPATH=/your_path/to/BlockProfile/ && ` before the `CUDA_VISIBLE_DEVICES=...` command.
  
* If you encounter `broken pipe` errors, consider downgrading your version of `PyTorch` and `Torchvision`.