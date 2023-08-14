import numpy as np
import pandas as pd
import os

# groundtruth = pd.read_excel(r"./input_file/GB1.xlsx")
# dirs = os.listdir(r"../data/GB1_PhoQ_data/results/GB1_mlde_supervised_output")
groundtruth = pd.read_excel(r"./input_file/PhoQ.xlsx")
dirs = os.listdir(r"../../data/GB1_PhoQ_data/results/PhoQ_mlde_supervised_output")

#print(dirs)
max = []
mean = []
count = 0
for d in dirs:
    result = pd.read_csv(r"../../data/GB1_PhoQ_data/results/PhoQ_mlde_supervised_output/" + d + "/PredictedFitness.csv")
    # result = pd.read_csv(
    #     r"../data/GB1_PhoQ_data/results/GB1_mlde_supervised_output/" + d + "/PredictedFitness.csv")
    #print(groundtruth[groundtruth["Variants"]=="TYGM"]["Fitness"].values[0])
    res = result.sort_values(by="PredictedFitness", ascending=False)
    res_384 = res[res["InTrainingData?"]=="YES"]
    res_96 = res[res["InTrainingData?"]=="NO"].head(96)
    res_con = pd.concat([res_384, res_96])
    #head = res.head(96)
    #print(head["AACombo"].values)
    ground_fitness_list = []
    ground_fitness_list_96 = []
    for j in range(len(res_con)):
        combo = res_con.iloc[j]["AACombo"]
    #for combo in res_con["AACombo"].values:
        ground_fitness_list.append(groundtruth[groundtruth["Variants"] == combo]["Fitness"].values[0])
        if res_con.iloc[j]["InTrainingData?"]=="NO":
            ground_fitness_list_96.append(groundtruth[groundtruth["Variants"] == combo]["Fitness"].values[0])
    #print(ground_fitness_list)
    ground_fitness_list = np.array(ground_fitness_list)
    #print(ground_fitness_list)
    print("max:{}".format(np.max(ground_fitness_list)))
    print("mean:{}".format(np.mean(ground_fitness_list_96)))
    max.append(np.max(ground_fitness_list))
    mean.append(np.mean(ground_fitness_list_96))
    if np.max(ground_fitness_list) > 133:
    #if np.max(ground_fitness_list) > 8.7:
        count += 1
print(count)
print("max:{}".format(np.mean(max)))
print("mean:{}".format(np.mean(mean)))