# -*- coding: utf-8 -*-
import esm
import pandas as pd
import torch.nn.functional as F
import torch

def alphabet_coding(data: list, maxlen: int):
    #model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    _, alphabet = esm.pretrained.load_model_and_alphabet_local("Data/esm2_t6_8M_UR50D.pt")
    batch_converter = alphabet.get_batch_converter()
    '''
    # Prepare data (first 3 sequences from ESMStructuralSplitDataset superfamily)
    data = [("protein1", "MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG"),
            ("protein2", "KALTARQQEVFDLIRDHISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE"),
            ("protein3 with mask","KALTARQQEVFDLIRD<mask>ISQTGMPPTRAEIAQRLGFRSPNAAEEHLKALARKGVIEIVSGASRGIRLLQEE") ]
    '''
    _, _, batch_tokens = batch_converter(data) #batch_labels, batch_strs, batch_tokens
    padding_size = (0, maxlen - batch_tokens.shape[-1])
    batch_tokens = F.pad(batch_tokens, pad=padding_size, mode='constant', value=0)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)  
    
    return batch_tokens, batch_lens


def Blosum_feat(data: list, maxlen: int):
    blosum62 = {
            'A': [4, -1, -2, -2, 0,  -1, -1, 0, -2,  -1, -1, -1, -1, -2, -1, 1,  0,  -3, -2, 0],  # A
            'R': [-1, 5,  0, -2, -3, 1,  0,  -2, 0,  -3, -2, 2,  -1, -3, -2, -1, -1, -3, -2, -3], # R
            'N': [-2, 0,  6,  1,  -3, 0,  0,  0,  1,  -3, -3, 0,  -2, -3, -2, 1,  0,  -4, -2, -3], # N
            'D': [-2, -2, 1,  6,  -3, 0,  2,  -1, -1, -3, -4, -1, -3, -3, -1, 0,  -1, -4, -3, -3], # D
            'C': [0,  -3, -3, -3, 9,  -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1], # C
            'Q': [-1, 1,  0,  0,  -3, 5,  2,  -2, 0,  -3, -2, 1,  0,  -3, -1, 0,  -1, -2, -1, -2], # Q
            'E': [-1, 0,  0,  2,  -4, 2,  5,  -2, 0,  -3, -3, 1,  -2, -3, -1, 0,  -1, -3, -2, -2], # E
            'G': [0,  -2, 0,  -1, -3, -2, -2, 6,  -2, -4, -4, -2, -3, -3, -2, 0,  -2, -2, -3, -3], # G
            'H': [-2, 0,  1,  -1, -3, 0,  0,  -2, 8,  -3, -3, -1, -2, -1, -2, -1, -2, -2, 2,  -3], # H
            'I': [-1, -3, -3, -3, -1, -3, -3, -4, -3, 4,  2,  -3, 1,  0,  -3, -2, -1, -3, -1, 3],  # I
            'L': [-1, -2, -3, -4, -1, -2, -3, -4, -3, 2,  4,  -2, 2,  0,  -3, -2, -1, -2, -1, 1],  # L
            'K': [-1, 2,  0,  -1, -3, 1,  1,  -2, -1, -3, -2, 5,  -1, -3, -1, 0,  -1, -3, -2, -2], # K
            'M': [-1, -1, -2, -3, -1, 0,  -2, -3, -2, 1,  2,  -1, 5,  0,  -2, -1, -1, -1, -1, 1],  # M
            'F': [-2, -3, -3, -3, -2, -3, -3, -3, -1, 0,  0,  -3, 0,  6,  -4, -2, -2, 1,  3,  -1], # F
            'P': [-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4, 7,  -1, -1, -4, -3, -2], # P
            'S': [1,  -1, 1,  0,  -1, 0,  0,  0,  -1, -2, -2, 0,  -1, -2, -1, 4,  1,  -3, -2, -2], # S
            'T': [0,  -1, 0,  -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1, 1,  5,  -2, -2, 0],  # T
            'W': [-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1, 1,  -4, -3, -2, 11, 2,  -3], # W
            'Y': [-2, -2, -2, -3, -2, -1, -2, -3, 2,  -1, -1, -2, -1, 3,  -3, -2, -2, 2,  7,  -1], # Y
            'V': [0,  -3, -3, -3, -1, -2, -2, -3, -3, 3,  1,  -2, 1,  -1, -2, -2, 0,  -3, -1, 4],  # V
            '-': [0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0],  # -
        }
    blo_feat = {}
    for row in data:   
        blo = [blosum62[i] for i in list(row[1])]
        blo += [blosum62['-'] for i in range(maxlen-len(blo))]
        blo_feat[row[0]] = blo
    
    return blo_feat
    

def AAIndex_feat(data: list, maxlen: int):
    aaindex = pd.read_csv('Data/aaindex_feature.csv', index_col=0)
    aaindex = dict(zip(aaindex.index,aaindex.values.tolist()))
    aai_feat = {}
    for row in data:   
        aai = [aaindex[i] for i in list(row)]
        aai += [aaindex['-'] for i in range(maxlen-len(aai))]
        aai_feat[row[0]] = aai
    
    return aai_feat

