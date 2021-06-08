# Fast ImageNet RAM Validation Loader

Supernet inference relies on the ImageNet validation set (batch norm calibration requires a small number of training images), which while requiring a fraction of the resources of 
the resources used for training, is still expensive in terms of I/O ops.

To speedup this process, we developed a RAM loader, where the entire validation set is forwarded through the usual data
transforms, concatenated together, and saved to disk as a very large tensor. This tensor can then be loaded into RAM at 
the start of an experiment, negating the need for constant I/O ops when performing inference.

Specifying the "--fast" option in main.py will use the fast RAM validation loader, but first the large tensor must be
constructed and saved to disk.

```
python3 imagenet_RAM_saver.py
    --in_dir # Location of ImageNet data
    --out_dir # Where the tensor will be saved, we do no advise changing the default
    --resolution # Resolution that the tensor will be saved as, {192, 208, 224} are good options
```

 Creating the tensor requires a lot of RAM (>32GB), as does using it at runtime, but it can speedup inference greatly.