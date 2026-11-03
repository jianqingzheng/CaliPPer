import torch
import random
import pandas as pd
import numpy as np
import torch.utils.data as Data
from math import ceil

def split_list_by_ratio(data, test_ratio):
    abs_unique = list({item[0] for item in data})
    ags_unique = list({item[1] for item in data})
    
    test_ab_count = max(1, ceil(len(abs_unique) * test_ratio))
    test_ag_count = max(1, ceil(len(ags_unique) * test_ratio))
    random.seed(42)
    test_abs = set(random.sample(abs_unique, test_ab_count))
    test_ags = set(random.sample(ags_unique, test_ag_count))
    
    test_set = [
        item for item in data 
        if item[0] in test_abs or item[0] in test_ags
    ]
    train_set = [item for item in data if item not in test_set]
    assert len(train_set) + len(test_set) == len(data), "not!"

    return train_set, test_set

    
# def BatchGenerate(pairs, bs, thres_ab, thres_ag):
#     Ab = [(x[0], x[1]) for x in pairs]
#     Ag = [(x[2], x[3]) for x in pairs]
#     label = [x[4] for x in pairs]
#     Ab_tokens, Ab_lens = alphabet_coding(Ab, maxlen = thres_ab)
#     Ag_tokens, Ag_lens = alphabet_coding(Ag, maxlen = thres_ag)
#     Labels = torch.from_numpy(np.array(label, dtype='float32'))
#     Loader_set = Data.DataLoader(dataset=Data.TensorDataset(Ab_tokens, Ag_tokens, Labels),
#                                    batch_size=bs, shuffle=False, drop_last=True)
#     return Loader_set

def sample_load(data_root, ratio):
    sample_file = '%s/ab_ag_pair.csv'%data_root
    ab_info = '%s/antibody.csv'%data_root
    ag_info = '%s/antigen.csv'%data_root

    pairs = pd.read_csv(sample_file, index_col=None, header=0)
    ab_info = pd.read_csv(ab_info, index_col=None, header=0)
    ag_info = pd.read_csv(ag_info, index_col=None, header=0)

    len_ab = [len(l) for l in ab_info['heavy']]
    len_ag = [len(l) for l in ag_info['ag_seq']]
    thres_ab = int(np.percentile(len_ab, ratio))
    thres_ag = int(np.percentile(len_ag, ratio))

    AbAg_pairs = []
    for index, row in pairs.iterrows():
        ag = ag_info.loc[ag_info['ag_name'] == str(row['ag_name'])]
        ab = ab_info.loc[ab_info['ab_name'] == str(row['ab_name'])]
        l_ab = ab['heavy'].str.len()
        l_ag = ag['ag_seq'].str.len()
        if l_ab.iloc[0]<=thres_ab and l_ag.iloc[0]<=thres_ag:
            pair = zip(ab['ab_name'], ag['ag_name'], str(row['label']))
            AbAg_pairs += pair

    return AbAg_pairs, thres_ab, thres_ag