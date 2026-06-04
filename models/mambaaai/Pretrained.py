# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torch
import esm
import pandas as pd
import torch.nn.functional as F
import torch
import os
import numpy as np

#data_root = 'Data/HIV'
data_root = 'Data/CoVAbDab'
ab_info = '%s/antibody.csv'%data_root
ag_info = '%s/antigen.csv'%data_root

ab_info = pd.read_csv(ab_info, index_col=None, header=0)
ag_info = pd.read_csv(ag_info, index_col=None, header=0)

def alphabet_coding(data: list, maxlen: int, save_dir):
    #model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    '''
    # Prepare data (first 3 sequences from ESMStructuralSplitDataset superfamily)
    data = [("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
            ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
            ("protein3 with mask","KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE") ]
    '''
    ESM_encoder, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    # ESM_encoder, alphabet = esm.pretrained.load_model_and_alphabet_local("Data/esm2_t6_8M_UR50D.pt")
    batch_converter = alphabet.get_batch_converter()
    os.makedirs(save_dir, exist_ok=True)
    ESM_encoder.eval()

    with torch.no_grad():
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
        results = ESM_encoder(batch_tokens, repr_layers=[6], return_contacts=False)
        token_representations = results["representations"][6]

        for i, tokens_len in enumerate(batch_lens): 
            seq_feats= F.avg_pool1d(token_representations[i, 1:tokens_len-1], kernel_size=20, stride=20)
            pad_size = maxlen - seq_feats.size(0)
            padded_tensor = F.pad(seq_feats, 
                                (0, 0, 0, pad_size),
                                mode='constant', 
                                value=0)
            np.save(f"{save_dir}/{batch_labels[i]}.npy", padded_tensor)

# pretrained features
len_ab = [len(l) for l in ab_info['heavy']]
len_ag = [len(l) for l in ag_info['ag_seq']]

thres_ab = int(np.percentile(len_ab, 100)); thres_ag = int(np.percentile(len_ag, 100))
# dataset is HIV, thres_ab and thres_ag are set as follows:
# thres_ab = int(np.percentile(len_ab, 90)); thres_ag = int(np.percentile(len_ag, 90))

ab_info = list(zip(ab_info['ab_name'], ab_info['heavy']))
ag_info = list(zip(ag_info['ag_name'], ag_info['ag_seq']))

ab_info = [(x, y) for x, y in ab_info if len(y)<=thres_ab]
ag_info = [(x, y) for x, y in ag_info if len(y)<=thres_ag]

alphabet_coding(ag_info, maxlen = thres_ag, save_dir='Data/Pretrained_Cov/ag')
alphabet_coding(ab_info, maxlen = thres_ab, save_dir='Data/Pretrained_Cov/ab')
# alphabet_coding(ag_info, maxlen = thres_ag, save_dir='Data/Pretrained_HIV/ag')
# alphabet_coding(ab_info, maxlen = thres_ab, save_dir='Data/Pretrained_HIV/ab')
# chunks = np.array_split(ag_info, 4)
# for item in chunks:
#     alphabet_coding(item, maxlen = thres_ag, save_dir='Data/Pretrained_Cov/ag')

