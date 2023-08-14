# EvoPlay-assisted directed evolution 
GB1_PhoQ_task corresponds to "EvoPlay-assisted directed evolution" task in the manuscript.

## Running 
### Generate training data
- Run *active_gp.py* to generate 500 repeats of 384 designed sequences for the following supervised training procedure. The 500 output files of every 384 generated sequences will be stored in */code/GB1_PhoQ_task/output_384_training_seqs/GB1* or */code/GB1_PhoQ_task/output_384_training_seqs/PhoQ* folder.   

### Evaluation
An whole 500 repeats of completely generated training sequences are stored in */data/GB1_PhoQ_data/results/GB1_384trainingSeqs_500repeats_30simulatin* folder,and their supervised training results are in folder*/data/GB1_PhoQ_data/results/GB1_mlde_supervised_output*, which are computed using the code of ftMLDE(https://github.com/fhalab/MLDE). 
- Run *mean_max_3.py* to calculat "Global maximal fitness hit count","Predicted max fitness","Predicted mean fitness" metrics, 
- Run *local_max_hit.py* to see the local peaks count. 



