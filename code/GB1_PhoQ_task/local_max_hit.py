import numpy as np
import pandas as pd
import os


groundtruth = pd.read_excel(r"./input_file/PhoQ.xlsx")
dirs = os.listdir(r"../../data/GB1_PhoQ_data/results/PhoQ_mlde_supervised_output")
#dirs = os.listdir(r"E:\project_s\CLADE-main\one hot\output\mlde_output")
#print(dirs)
max = []
mean = []
count = 0
count_TEMH = 0
count_QHDG = 0
count_QMGE = 0
count_TQCE = 0
count_GEMI = 0
count_TKMC = 0
count_QGWY = 0
for d in dirs:
    result = pd.read_csv(r"../../data/GB1_PhoQ_data/results/PhoQ_mlde_supervised_output/" + d + "/PredictedFitness.csv")

    res = result.sort_values(by="PredictedFitness", ascending=False)
    res_384 = res[res["InTrainingData?"] == "YES"]
    res_96 = res[res["InTrainingData?"] == "NO"].head(96)
    res_con = pd.concat([res_384, res_96])
    # head = res.head(96)
    # print(head["AACombo"].values)
    ground_fitness_list = []
    ground_fitness_list_96 = []
    for j in range(len(res_con)):
        combo = res_con.iloc[j]["AACombo"]
    #print(head["AACombo"].values)
    #ground_fitness_list = []
    #for combo in head["AACombo"].values:
        #ground_fitness_list.append(groundtruth[groundtruth["Variants"] == combo]["Fitness"].values[0])
        if combo == "TEMH":
            count_TEMH +=1
        if combo == "QHDG":
            count_QHDG +=1
        if combo == "QMGE":
            count_QMGE +=1
        if combo == "TQCE":
            count_TQCE +=1
        if combo == "GEMI":
            count_GEMI +=1
        if combo == "TKMC":
            count_TKMC +=1
        if combo == "QGWY":
            count_QGWY +=1
    #print(ground_fitness_list)
    #ground_fitness_list = np.array(ground_fitness_list)
    #print(ground_fitness_list)
    print("TEMH:{}".format(count_TEMH)+"QHDG:{}".format(count_QHDG)+"QMGE:{}".format(count_QMGE)+"TQCE:{}".format(count_TQCE)+\
          "GEMI:{}".format(count_GEMI)+"TKMC:{}".format(count_TKMC)+"QGWY:{}".format(count_QGWY))
    #print("mean:{}".format(np.mean(ground_fitness_list)))
    # max.append(np.max(ground_fitness_list))
    # mean.append(np.mean(ground_fitness_list))
    #if np.max(ground_fitness_list) > 133:
    # if np.max(ground_fitness_list) > 8.7:
    #     count += 1
df1 = pd.DataFrame([{"TEMH":count_TEMH,"QHDG":count_QHDG,"QMGE":count_QMGE,"TQCE":count_TQCE,\
          "GEMI":count_GEMI,"TKMC":count_TKMC,"QGWY":count_QGWY}])
#df1 = pd.DataFrame([{"TEMH":1,"QHDG":1,"QMGE":1,"TQCE":1,"GEMI":1,"TKMC":1,"QGWY":1}])
df1.to_csv(r"../../data/GB1_PhoQ_data/results/PhoQ500_local_max_hits_96.csv",index=False) #