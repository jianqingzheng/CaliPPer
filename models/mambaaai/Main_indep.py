# -*- coding: utf-8 -*-
import pandas as pd
import random
import torch
import json
import argparse
import os
from Models import MambaCross
from Toolkit import Metrics, set_seed_all, make_dir, AntibodyAntigenDataset, custom_collate_fn
from Loader import *
import math
from torch.utils.data import DataLoader

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_epoch(mymodel, train_loader):
    loss_train = 0
    Y_true, Y_pred = [], []
    for batch, (ab, ag, labels) in enumerate(train_loader):
        ab = ab.to(device)
        ag = ag.to(device)
        labels = labels.to(device)
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

def valid_epoch(mymodel, valid_loader):
    loss_valid = 0
    Y_true, Y_pred = [], []
    mymodel.eval()
    with torch.no_grad():
        for batch, (ab, ag, labels) in enumerate(valid_loader):
            ab = ab.to(device)
            ag = ag.to(device)
            labels = labels.to(device)  
            preds  = mymodel(ab, ag)
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
    parser.add_argument('--epoch num', dest='epoch', nargs='?', type=int, default=100)
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
    task_save_folder = os.path.join('Results')
    make_dir(task_save_folder)

    #------------model training 
    Best_AUC = 0; Best_AUPR = 0
    model = MambaCross(hor_dim=thres_ag, ver_dim=thres_ab, 
                    feat_dim=param["latent_dim"], 
                    seq_len=thres_ab+thres_ag,
                    hidden_sizes=param["decoder_hidden_dims"],
                    mamba_layer=param["mamba_layer"],
                    pooling=param["pooling_way"],
                    activation=param["activation"], 
                    drop_ratio=param["dropout"]).to(device)
    #model = nn.DataParallel(model, device_ids=[0, 1]) 
    
    optimizer = torch.optim.Adam(model.parameters(), lr=param['learning_rate'], weight_decay=0)
    criterion = torch.nn.BCELoss()
    #----------dataset spilt
    train_set = AntibodyAntigenDataset(CV)
    indep_set = AntibodyAntigenDataset(Independent)
    train_loader = DataLoader(train_set, batch_size=param['batchsize'], num_workers=4, 
                            shuffle=True, collate_fn=custom_collate_fn)
    indep_loader = DataLoader(indep_set, batch_size=param['batchsize'], num_workers=4, 
                            shuffle=False, collate_fn=custom_collate_fn)
    
    Final_AUC = 0; Final_AUPR = 0; Final_F1 = 0; Final_ACC = 0; Final_LOSS = math.inf
    model.train()
    for epoch in range(args.epoch):
        print('epoch = %d'% (epoch+1))
        train_epoch(model, train_loader)
        AUC, AUPR, F1, ACC, Loss = valid_epoch(model, indep_loader)
        print('indepd auc: ' + str(round(AUC, 4)) + '  indep aupr: ' + str(round(AUPR, 4)) +
            '  indep f1: ' + str(round(F1, 4)) + '  indep acc: ' + str(round(ACC, 4)))  
        if(AUC > Final_AUC):
            Final_AUC = AUC; Final_AUPR = AUPR; Final_F1 = F1; Final_ACC = ACC
            Final_LOSS = Loss
            torch.save(model.state_dict(), os.path.join(task_save_folder, 'model_indep.pt'))

    #------------cross validation/Seen validating
    results = pd.DataFrame([Final_AUC, Final_AUPR, Final_F1, Final_ACC])
    results.index = ['auc','aupr','f1','acc']
    file_name = os.path.join(task_save_folder, 'metrics_indep')
    with open(f'{file_name}.csv', 'w') as f:
        results.to_csv(f)     