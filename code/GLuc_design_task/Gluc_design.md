# GLuc design task 
GLuc design task corresponds to Gaussia luciferase engineering in the manuscript 

## Runing 
- Run *active_gp_gluc.py* one time to generate 150 designed Gluc sequences. In the manuscript, the process is run with 10 repeats. The design results will be stored in */code/GLuc_design_task/output_design* folder. 
 
The model input files in */code/GLuc_design_task/input_file* are processed from *5-Gluc突变体位点信息.xlsx* in folder*/data/GLuc_design_preprocess_data/in_house_variant_preprocess*, and all Intermediate files generated during the preprocessing are in this folder. See *gluc_data_process.ipynb* for details of preprocessing.