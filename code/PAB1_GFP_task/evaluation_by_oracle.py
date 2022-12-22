import tape
import torch
import numpy as np
import pandas as pd
# import datetime
device = "cuda:0"

tokenizer = tape.TAPETokenizer(vocab="iupac")

# model = tape.ProteinBertForValuePrediction.from_pretrained(
#             "/data/oracle_weight/avgfp"
#         ).to(device)
model = tape.ProteinBertForValuePrediction.from_pretrained(
            "/data/oracle_weight/pab1"
        ).to(device)

for j in range(1,2):
    seqs =[]
    all_score =[]

    df_2 = pd.read_csv('/code/PAB1_GFP_task/evozero_pab1_generated_sequence_1.csv')
    sequences = list(df_2['sequence'])
    print('sequence counts',len(sequences)) 
    score_list =[]
    for se in sequences:

        encoded_seqs = torch.tensor(
                        tokenizer.encode(se)
                    ).unsqueeze(0).to(device)
        score = model(encoded_seqs)[0].detach().cpu().numpy().astype(float).reshape(-1)

        score_list.append(score[0])

    print(len(score_list))

    evalute_df = pd.DataFrame({'sequence':sequences,'score':score_list})
    evalute_df.sort_values("score",inplace=True,ascending=False)
    print(evalute_df.head(10))


