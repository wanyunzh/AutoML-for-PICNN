# AutoML-for-PICNN
## 1.Datasets

The datasets for the Darcy flow and Poisson equations with different layouts can be downloaded in [the corresponding paper](https://hkustgz-my.sharepoint.com/:f:/g/personal/wzhou266_connect_hkust-gz_edu_cn/EokovhAZ0xRMntQ9JSgQZU4BLSsLPZkWlz18pTs_DdL_Vw?e=zPsB8S.<br />).

The datasets for the Navier-Stokes equations can be downloaded in https://github.com/Jianxun-Wang/phygeonet.<br /> (case3)


## 2. Searching for loss functions

For each dataset, run hpo_algor.py to search for the best loss function.

## 3. Searching for network architectures

### 3.1 Entire-structured search space

1. For each dataset, run search_struct.py to search for the optimal PICNN architecture by using reinforcement learning (multi-trial strategy) and then run retrain.py to get the prediction results. 
2. For each dataset, run retrain_oneshot.py to get the prediction results by using the one-shot strategies

### 3.2 Cell-based search space

For each dataset, run search_cell.py to search for the optimal PICNN architecture for cell-based search space




