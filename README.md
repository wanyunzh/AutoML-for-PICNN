# AutoML-for-PICNN
## 1.Datasets

The datasets for the Darcy flow and heat equations with different layouts can be downloaded in https://hkustgz-my.sharepoint.com/:f:/g/personal/wzhou266_connect_hkust-gz_edu_cn/EokovhAZ0xRMntQ9JSgQZU4BLSsLPZkWlz18pTs_DdL_Vw?e=zPsB8S.<br />


## 2. Searching for loss functions

For each dataset, run hpo_algor.py to search for the best loss function.

## 3. Searching for network architectures

### 3.1 Entire-structured search space

1. For each dataset, run search_struct.py to search for the optimal PICNN architecture by using reinforcement learning (multi-trial strategy) and then run retrain.py to get the prediction results. 
2. For each dataset, run retrain_oneshot.py to get the prediction results  by using one-shot strategies





Python 3.9.7

Python packages: Tensorfollow(vr.2.8.0), pandas and scipy

### 1.4 Starting a prediction

* **prediction of anticancer or antimicrobial peptides**

usage: TriNet.py [-h] [--PSSM_file PSSM_FILE] [--sequence_file  SEQUENCE_FILE] [--output OUTPUT] [--operation_mode OPERATION_MODE]

* **Rquired**

--PSSM_file PSSM_FILE, -p PSSM_FILE

path of PSSM  files

--sequence_file SEQUENCE_FILE, -s SEQUENCE_FILE

path of sequence file

--operation_mode OPERATION_MODE, -mode OPERATION_MODE

c for anticancer prediction and m for antimicrobial prediction

* **Optional**

-h, --help show this help message and exit

--output OUTPUT, -o OUTPUT

path of Trinet result,  defaut with path of current path/output

* **Note**

1.There are four options for '-mode' item, they are sc,sm,fc and fm. s is standard mode(need users to provide pssm) f is fast mode(only need to provide fasta files) c for anticancer peptides prediction and m for antimicrobial peptides prediction.

2.If the '-o' output item is empty in your command line, the corresponding result file will be placed in the current  working path.

* **Typical commands**

The following command is an example for anticancer peptide prediction:

```

$ python ./TriNet/TriNet.py -mode sc -s ./TriNet/ACP_example.fasta -p ./TriNet/pssm_acp_example/ -o ./TriNet/acpout.csv

```

## 2.Repeating experiments in the paper.

All codes and data are placed in the TriNet-Reproducing.zip file. This file needs to be decompressed before reproducing the results and you can read README.md in TriNet-Reproducing.zip for more information.

## Contact

Any questions, problems, bugs are welcome and should be dumped to

wanyunzh@gmail.com

# TriNet
[![DOI](https://zenodo.org/badge/517402277.svg)](https://zenodo.org/badge/latestdoi/517402277)

