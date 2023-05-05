# EvoPlay

This repository contains code and tutorials for protein sequence design task from MGI-X, including ***PAB1_GFP_task***

+ Citation

```
@article{Licko2022evozero,
  title={Self-play reinforcement learning turbocharges protein engineering},
  author={Yi Wang, Hui Tang, Lichao Huang, Lulu Pan, Lixiang Yang, Ming Ni, Huanming Yang, Feng Mu, Meng Yang},
  journal={xxxx},
  doi={xxxx},
  url={x x x},
  year={2022},
  publisher={xxxx}
}

```


# EvoPlay-PAB1_GFP_task

Main code In EvoPlay/code/PAB1_GFP_task/: 

+ Main
  - ./train_m_single_m_p_pab1.py
- Evaluation
  - ./evaluation_by_oracle.py

## Usage


## Getting started

In this task, We train our EvoPlay on two protein dataset(PAB1、GFP) and generate new sequences. to perform this task，we just need two steps:


### Step1: Generate new sequences by using the below command:

### python train_m_single_m_p_pab1.py

After runing this command,the generated sequences will be stored in the evoplay_pab1_generated_sequence_1.csv,we can use the oracle landscape to evaluate the quality of the generated sequences.



### Step2: evaluate generated sequences by using the below command:

### python  evaluate_by_oracle.py

After runing this command,the result will be displayed on the output screen.


