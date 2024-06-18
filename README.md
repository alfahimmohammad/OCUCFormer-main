# Official Implementation of OCUCFormer: An Over-Complete Under-Complete Transformer Network for Accelerated MRI Reconstruction in PyTorch


### DATASETS:
1. [Automated Cardiac Diagnosis Challenge (ACDC)](https://ieeexplore.ieee.org/document/8360453)
2. [MRBrainS dataset](https://www.hindawi.com/journals/cin/2015/813696/)

#### Directory Structure:
```

├── datasets
    |-- {DATASET_TYPE}
        |-- train
            |-- acc_{ACC_FACTOR}
                |-- 1.h5
                |-- 2.h5
                |..
        |-- validation
           |--acc_{ACC_FACTOR}
                |-- 1.h5
                |-- 2.h5
                |..
├── experiments
    |-- {DATASET_TYPE}
        |-- acc_{ACC_FACTOR}
            |-- {MODEL}
                |-- best_model.pt
                |-- model.pt
                |-- summary
                |-- results
                    |-- 1.h5
                    |-- 2.h5
                    |-- .
                |-- report.txt
```
Example: {DATASET_TYPE} = cardiac, {ACC_FACTOR} = 4x, {MODEL} = OCUC_Former
```

├── datasets
    |-- cardiac
        |-- train
            |--acc_4x
                |-- 1.h5
                |-- 2.h5
                |..
        |-- validation
           |--acc_4x
                |-- 1.h5
                |-- 2.h5
                |..
├── experiments
    |-- cardiac
        |-- acc_4x
            |-- OCUC_Former
                |-- best_model.pt
                |-- model.pt
                |-- summary
                |-- results
                    |-- 1.h5
                    |-- 2.h5
                    |..
```
