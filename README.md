# Official Implementation of OCUCFormer: An Over-Complete Under-Complete Transformer Network for Accelerated MRI Reconstruction in PyTorch

## ABSTRACT:

Many deep learning-based architectures have been proposed for accelerated Magnetic Resonance Imaging (MRI) reconstruction. However, existing encoder-decoder-based popular networks have a few shortcomings: (1) They focus on the anatomy structure at the expense of fine details, hindering their performance in generating faithful reconstructions; (2) Lack of long-range dependencies yields sub-optimal recovery of fine structural details. In this work, we propose an Over-Complete Under-Complete Transformer network (OCUCFormer) which focuses on better capturing fine edges and details in the image and can extract the long-range relations between these features for improved single-coil (SC) and multi-coil (MC) MRI reconstruction. Our model computes long-range relations in the highest resolutions using Restormer modules for improved acquisition and restoration of fine anatomical details. Towards learning in the absence of fully sampled ground truth for supervision, we show that our model trained with under-sampled data in a self-supervised fashion shows a superior recovery of fine structures compared to other works. We have extensively evaluated our network for SC and MC MRI reconstruction on brain, cardiac, and knee anatomies for 4x and 5x acceleration factors. We report significant improvements over popular deep learning-based methods when trained in supervised and self-supervised modes. We have also performed experiments demonstrating the strengths of extracting fine details and the anatomical structure and computing long-range relations within over-complete representations.

### Graphical Abstract of OCUCFormer:


Graphical abstract to illustrate the effect of Under-Complete (UC) and Over-Complete (OC) respectively on the receptive field and the attention mechanism, respectively. (a) Change in the receptive field:  The change in the receptive field observed in under-complete (blue windows) and over-complete networks (green windows) is depicted. This shows that the size of the receptive field in OC is restricted compared to UC, where it enlarges in successive UC layers. (b) Computation of channel-wise self-attention in OC: (1) The features in the input image are converted to OC features in different channels, (2) The arrow marks indicate the long-range dependencies between the OC features within different channels and applying channel-wise self-attention mechanism captures these relations and results in (3), (3) Final OC features in a particular channel and location are computed using OC features from different channels and locations weighted by their relations. The channel-wise self-attention mechanism of Restormer implicitly models and captures the pixel-wise long-range dependencies within the OC features of different channels (colored patches in (b): (1) and (2)). 



![alt_text](https://github.com/alfahimmohammad/OCUCFormer-main/blob/master/Images/graphical_abstract_fig_new.png?raw=true)

The proposed network architecture comprises of Over-complete block (OC-Net), an Under-complete block (UC-Net), and a Hidden State Update block (gray-shaded blocks). The hidden state vector \begin{math}h_{(t)}\end{math} for the OC-Net, \begin{math}h^{*}_{(t)}\end{math} for the UC-Net are initialized with zeros in the shape depicted in the figure at time step $t=0$. In the Hidden State Update block, the 3rd DPConv layer's output feature map is added with the Restormer block's output in the case of OC-Net, resulting in \begin{math}h_{(t+1)}\end{math}. In case of the UC-Net, the output feature map from the 3rd DPConv block is added with the down-sampled feature map $X_{(t)}$, resulting in \begin{math}h^{*}_{(t+1)}\end{math}. The updated hidden state vectors, \begin{math}h_{(t+1)}\end{math} and \begin{math}h^{*}_{(t+1)}\end{math} are first down-sampled/up-sampled (in case of OC-Net/UC-Net respectively) and then concatenated with the output feature maps from the corresponding encoder layers. They are then passed on to the decoder layers of OC-Net/UC-Net, finally resulting in the updated $Y_{(t)}$ and $Y^{*}_{(t)}$. \begin{math}Y_{(t)}\end{math} and \begin{math}Y^{*}_{(t)}\end{math} are used as input images for the next time steps for OC-Net and UC-Net respectively. The OC-Net and UC-Net are recurrently updated for T timesteps in a sequential manner.


## System setup:
#### Dependencies:
[Requirements](https://github.com/alfahimmohammad/OCUCFormer-main/blob/master/requirements.txt)

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
