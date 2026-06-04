# -*- coding: utf-8 -*-
import pandas as pd
import random
import torch
import json
import argparse
import torch.nn as nn
import os
from sklearn.model_selection import KFold
from Models import MambaCross
from Toolkit import Metrics, set_seed_all, make_dir, softmax, AntibodyAntigenDataset, custom_collate_fn
from Loader import *
import math
from torch.utils.data import DataLoader
from tqdm import tqdm

def train_epoch(mymodel, train_loader):
    loss_train = 0
    Y_true, Y_pred = [], []
    #for batch, (ab, ag, labels) in enumerate(tqdm(train_loader)):
    for batch, (ab, ag, labels) in enumerate(train_loader):
        ab = ab.to('cuda:0')
        ag = ag.to('cuda:0')
        labels = labels.to('cuda:0')  
        optimizer.zero_grad()
        preds = mymodel(ab, ag)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
        Y_true += labels.cpu().detach().numpy().tolist()
        Y_pred += preds.cpu().detach().numpy().tolist()
    AUC, AUPR, F1, ACC = Metrics(Y_true, Y_pred)
    print('train-loss=', loss_train/len(train_loader.dataset))
    print('train auc: ' + str(round(AUC, 4)) + '  train aupr: ' + str(round(AUPR, 4)) +
          '  train f1: ' + str(round(F1, 4)) + '  train acc: ' + str(round(ACC, 4)))  

def valid_epoch(mymodel, valid_loader, flag=True):
    loss_valid = 0
    Y_true, Y_pred = [], []
    Return_attentions = []
    mymodel.eval()
    with torch.no_grad():
        #for batch, (ab, ag, labels) in enumerate(tqdm(valid_loader)):
        for batch, (ab, ag, labels) in enumerate(valid_loader):
            ab = ab.to('cuda:0')
            ag = ag.to('cuda:0')
            labels = labels.to('cuda:0')  
            preds = mymodel(ab, ag)
            loss = criterion(preds, labels)
            loss_valid += loss.item()
            Y_true += labels.cpu().detach().numpy().tolist()
            Y_pred += preds.cpu().detach().numpy().tolist()
    AUC, AUPR, F1, ACC = Metrics(Y_true, Y_pred)
    return AUC, AUPR, F1, ACC, loss_valid
    

if __name__ == '__main__':    
    parser = argparse.ArgumentParser('training setting')
    parser.add_argument('--len_ratio', dest='ratio', nargs='?', type=int, default=100, help='Sequence length selection ratio')
    # parser.add_argument('--len_ratio', dest='ratio', nargs='?', type=int, default=90, help='Sequence length selection ratio') If the dataset is HIV!
    parser.add_argument('--seed', dest='seed', nargs='?', type=int, default=42, help='Random seed setting')
    parser.add_argument('--data_root', dest='filepath', nargs='?', type=str, default='Data/CoVAbDab', choices=['Data/HIV', 'Data/CoVAbDab'])
    parser.add_argument('--independent ratio', dest='ir', nargs='?', type=float, default=0.1)
    parser.add_argument('--fold num', dest='n', nargs='?', type=int, default=5)
    parser.add_argument('--epoch num', dest='epoch', nargs='?', type=int, default=50)
    args = parser.parse_args()

    #-----------loading dataset
    set_seed_all(args.seed)
    with open(os.path.join('Param_Model.json'), 'r') as f:
        param = json.load(f)
    AbAg_pairs, thres_ab, thres_ag = sample_load(args.filepath, args.ratio)
    random.shuffle(AbAg_pairs)
    CV, Independent = split_list_by_ratio(AbAg_pairs, test_ratio = args.ir)
    print(len(CV))
    print(len(Independent))
    Independent = AntibodyAntigenDataset(Independent)
    Indep_loader = DataLoader(Independent, batch_size=param['batchsize'], shuffle=True, num_workers=4, 
                              collate_fn=custom_collate_fn)

    kf = KFold(n_splits = args.n, shuffle = True, random_state = (args.seed)) 
    task_save_folder = os.path.join('Results')
    make_dir(task_save_folder)

    #------------model training 
    Best_AUC = 0; Best_AUPR = 0
    All_AUC = []; All_AUPR = []; All_F1 = []; All_ACC = []
    fold = 0
    for train_index, valid_index in kf.split(CV):
        print('-----------Fold = %d-------------'% (fold+1))
        model = MambaCross(hor_dim=thres_ag, ver_dim=thres_ab, 
                        feat_dim=param["latent_dim"], 
                        seq_len=thres_ab+thres_ag,
                        hidden_sizes=param["decoder_hidden_dims"],
                        mamba_layer=param["mamba_layer"],
                        pooling=param["pooling_way"],
                        activation=param["activation"], 
                        drop_ratio=param["dropout"]).to('cuda:0')
        # model = nn.DataParallel(model, device_ids=[0, 1]) 
        # for name, param in model.named_parameters():
        #     print(f"{name}: requires_grad={param.requires_grad}")
        optimizer = torch.optim.Adam(model.parameters(), lr=param['learning_rate'], weight_decay=0)
        criterion = torch.nn.BCELoss()
        #----------dataset spilt
        train_set = list(map(CV.__getitem__, train_index))
        valid_set = list(map(CV.__getitem__, valid_index))
        train_set = AntibodyAntigenDataset(train_set)
        valid_set = AntibodyAntigenDataset(valid_set)
        train_loader = DataLoader(train_set, batch_size=param['batchsize'], num_workers=4, 
                                shuffle=True, collate_fn=custom_collate_fn)
        valid_loader = DataLoader(valid_set, batch_size=param['batchsize'], num_workers=4, 
                                shuffle=True, collate_fn=custom_collate_fn)
        
        Final_AUC = 0; Final_AUPR = 0; Final_F1 = 0; Final_ACC = 0; Final_LOSS = math.inf
        model.train()
        for epoch in range(args.epoch):
            print('epoch = %d'% (epoch+1))
            train_epoch(model, train_loader)
            AUC, AUPR, F1, ACC, LOSS = valid_epoch(model, valid_loader)
            print('valid auc: ' + str(round(AUC, 4)) + '  valid aupr: ' + str(round(AUPR, 4)) +
                '  valid f1: ' + str(round(F1, 4)) + '  valid acc: ' + str(round(ACC, 4)))  
            if(LOSS < Final_LOSS):
                Final_AUC = AUC; Final_AUPR = AUPR; Final_F1 = F1; Final_ACC = ACC
                Final_LOSS = LOSS
        All_AUC.append(Final_AUC)
        All_AUPR.append(Final_AUPR)
        All_F1.append(Final_F1)
        All_ACC.append(Final_ACC) 

        #model saving
        if (np.mean(All_AUC) > Best_AUC) and (np.mean(All_AUPR) > Best_AUPR):
            #torch.save(model.state_dict(), os.path.join(task_save_folder, 'model.pt'))
            Best_AUC = np.mean(All_AUC)
            Best_AUPR = np.mean(All_AUPR)
        fold = fold + 1

    #------------CV-results saving
    results = pd.DataFrame([All_AUC, All_AUPR, All_F1, All_ACC])
    results.columns = ['fold1','fold2','fold3','fold4','fold5']
    results.index = ['auc','aupr','f1','acc']
    file_name = os.path.join(task_save_folder, 'metrics')
    with open(f'{file_name}.csv', 'w') as f:
        results.to_csv(f)     