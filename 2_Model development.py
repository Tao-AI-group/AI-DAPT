

### Tools and Packages
##Basics
import pandas as pd
import numpy as np
import sys, random
import math, csv
try:
    import cPickle as pickle
except:
    import pickle 
import string
import os, re
# specify cuda number
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import time
from datetime import datetime
from tqdm import tqdm

## ML and Stats
from sklearn import datasets, linear_model, model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as m
import sklearn.linear_model as lm
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.tree import export_graphviz
from lightgbm import LGBMClassifier
# import lightgbm as lgb
import xgboost as xgb

## DL Framework
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.autograd import Variable
from torch import optim

import models as model 
from EHRDataloader import EHRdataloader
from EHRDataloader import EHRdataFromLoadedPickles as EHRDataset
import utils as ut 
from EHREmb import EHREmbeddings
import def_function as func

#import statsmodels.formula.api as sm
#import patsy
#from scipy import stats

## Visualization
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import optuna
# import optuna.integration.lightgbm as lgb
# import optuna.integration.xgboost as xgb

use_cuda = torch.cuda.is_available()

def time_consumption_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def remove_long_code_pt(data_list):
    '''
    remove the pt with long code 1k+
    '''
    print(f'old list length is {len(data_list)}')
    for pt in data_list:
        if pt[0] in {177021981, 177019335}:
            print(pt[0])
            data_list.remove(pt)
    print(f'new list length is {len(data_list)}', '\n')

    return data_list

def remove_long_code_pt_original(data_list, max_len):
    '''
    remove the pt with long code over max_len
    '''
    print(f'old list length is {len(data_list)}')
    ptSK_long_code_set = set()
    for pt in data_list:
        max_code_len = len(max(pt[-1], key=lambda xmb: len(xmb[1]))[1])
        if max_code_len >= max_len:
            ptSK_long_code_set.add(pt[0])
            data_list.remove(pt)   
    print(f'new list length is {len(data_list)}, ptSK_long_code_set is {ptSK_long_code_set}', '\n')

    return data_list

def run_DL_model(ehr_model, lr, l2, train_sl, valid_sl, test_sl, bmodel_pth, bmodel_st, model_type):
    if model_type == "RNN": 
        ppmode = True
    else: 
        ppmode = False
    ## Data Loading
    # print ('creating the list of training minibatches')
    # for _ in range(5):
    #     new_train = train_sl + valid_sl 
    #     random.shuffle(new_train)
    #     train_sl=news_train[:20000]
    #     valid_sl=new_train[20000:]

    train = EHRDataset(train_sl, sort = True, model = 'RNN')
    train_mbs = list(EHRdataloader(train, batch_size = 128, packPadMode = True))
    valid = EHRDataset(valid_sl, sort = True, model ='RNN')
    valid_mbs = list(EHRdataloader(valid, batch_size = 128, packPadMode = True))
    test = EHRDataset(test_sl, sort = True, model = 'RNN')
    test_mbs = list(EHRdataloader(test, batch_size = 128, packPadMode = True))
 
    ## Hyperparameters -- Fixed for testing purpose
    epochs = 200
    # l2 = 0.0001
    # lr = 0.001
    eps = 1e-4
    w_model = 'RNN'
    # optimizer = optim.Adamax(ehr_model.parameters(), lr = lr, weight_decay = l2, eps = eps)  
    
    optimizer = optim.Adagrad(ehr_model.parameters(), lr = lr, weight_decay = l2, eps = eps) 
    ## could try different optimizer !
    # optimizer: adam; adagrad; adadelta

    ##Training epochs
    bestValidAuc = 0.0
    bestTestAuc = 0.0
    bestValidEpoch = 0
    train_auc_allep = []
    valid_auc_allep = []
    test_auc_allep = []  
    for ep in range(epochs):
        start = time.time()
        current_loss, train_loss = ut.trainbatches(train_mbs, model = ehr_model, optimizer = optimizer)
        avg_loss = np.mean(train_loss)
        train_time = time_consumption_since(start)
        eval_start = time.time()
        Train_auc, y_real, y_hat = ut.calculate_auc(ehr_model, train_mbs, which_model = w_model)
        valid_auc, y_real, y_hat = ut.calculate_auc(ehr_model, valid_mbs, which_model = w_model)
        TestAuc, y_real, y_hat = ut.calculate_auc(ehr_model, test_mbs, which_model = w_model)
        eval_time = time_consumption_since(eval_start)
        # print(f'Epoch: {ep}, Train_auc: {Train_auc}, Valid_auc: {valid_auc}, & Test_auc: {TestAuc}, Avg Loss: {avg_loss}, Train Time: {train_time}, Eval Time: {eval_time}')

        train_auc_allep.append(Train_auc)
        valid_auc_allep.append(valid_auc)
        test_auc_allep.append(TestAuc)

        if valid_auc > bestValidAuc: 
            bestValidAuc = valid_auc
            bestValidEpoch = ep
            bestTestAuc = TestAuc
            ###uncomment the below lines to save the best model parameters
            best_model = ehr_model
            # torch.save(best_model, bmodel_pth)
            # torch.save(best_model.state_dict(), bmodel_st)
        if ep - bestValidEpoch > 10: break
    bestValidAuc = round(bestValidAuc * 100, 2)
    bestTestAuc = round(bestTestAuc * 100, 2)
    print('bestValidAuc %f has a TestAuc of %f at epoch %d' % (bestValidAuc, bestTestAuc, bestValidEpoch), '\n')

    return train_auc_allep, valid_auc_allep, test_auc_allep, bestValidAuc, bestTestAuc, bestValidEpoch

def load_input_pkl(file_type, common_path):
    pkl_result_list = []
    input_name_list = ['.combined.train', '.combined.test', '.combined.valid', '.types']   
    for name in input_name_list:
        input_path = common_path + file_type + name
        pkl_result = pickle.load(open(input_path, 'rb'), encoding='bytes')
        pkl_result_list.append(pkl_result)
    train_sl, test_sl, valid_sl, types_d = pkl_result_list

    return train_sl, test_sl, valid_sl, types_d

def plot_roc_curve(label, score):
    fpr, tpr, ths = m.roc_curve(label, score) ### If I round it gives me an AUC of 64%
    roc_auc = m.auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color = 'darkorange', lw = 3, label = 'ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color = 'navy', lw = 1, linestyle = '--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel('Sensitivity')
    plt.xlabel('1-Specificity')
    plt.title('ROC')
    plt.legend(loc = "lower right")
    plt.show()
    print(f'ths is {ths}')

#### Interactive Plot showing the deep learning model training AUC change over epochs
def plot_DLauc_perf(train_auc_allepv, test_auc_allepv, valid_auc_allepv, title_m):
    epochs = 100
    train_auc_fg = go.Scatter(x= np.arange(epochs), y=train_auc_allepv, name='train')
    test_auc_fg = go.Scatter(x= np.arange(epochs), y=test_auc_allepv, name='test')
    valid_auc_fg = go.Scatter(x= np.arange(epochs), y=valid_auc_allepv, name='valid')
    valid_max = max(valid_auc_allepv)
    test_max = max(test_auc_allepv)
    data = [train_auc_fg, test_auc_fg, valid_auc_fg]
    layout = go.Layout(xaxis=dict(dtick=1), title=title_m)
    layout.update(dict(annotations=[go.layout.Annotation(text="Max Valid", x=valid_auc_allepv.index(valid_max), y=valid_max)]))
    fig = go.Figure(data=data, layout=layout)
    iplot(fig, filename=title_m)

### Model Evaluation
def auc_evaluate(model, test_features, test_labels):
    # predictions = model.predict(test_features)
    pred_prob = model.predict_proba(test_features)
    auc_p = roc_auc_score(test_labels, pred_prob[:,1])
    auc = round(auc_p * 100, 2)
    print('Model Performance')
    print('AUC = {:0.2f}%.'.format(auc))
    return auc

def objective_LSTM(trial):
    train = EHRDataset(train_sl, sort = True, model = 'RNN')
    train_mbs = list(EHRdataloader(train, batch_size = 128, packPadMode = True))
    valid = EHRDataset(valid_sl, sort = True, model = 'RNN')
    valid_mbs = list(EHRdataloader(valid, batch_size = 128, packPadMode = True))
    test = EHRDataset(test_sl, sort = True, model = 'RNN')
    test_mbs = list(EHRdataloader(test, batch_size = 128, packPadMode = True))

    epochs = 100
    w_model = 'RNN'
    # lr = trial.suggest_loguniform('lr', 1e-6, 1e-1)
    # l2 = trial.suggest_loguniform('l2', 1e-6, 1e-2)
    # eps = trial.suggest_loguniform('eps', 1e-8, 1e-3)
    # embed_dim = trial.suggest_int('embed_dim', 2**5, 2**9)
    # hidden_size = trial.suggest_int('hidden_size', 2**5, 2**9)
    lr_expo = trial.suggest_float('lr_expo', -5.0, -1.0)
    l2_expo = trial.suggest_int('l2_expo', -5, -2)
    eps_expo = trial.suggest_int('eps_expo', -5, -3)
    embed_dim_expo = trial.suggest_int('embed_dim_expo', 6, 8)
    hidden_size_expo = trial.suggest_int('hidden_size_expo', 6, 8)
    optimizer_name = trial.suggest_categorical('optimizer_name', ['Adam', 'Adagrad', 'Adamax', 'Adadelta'])

    ehr_model = model.EHR_RNN(input_size_1, embed_dim = 2**embed_dim_expo, hidden_size = 2**hidden_size_expo, n_layers = 1, dropout_r = 0., cell_type = 'LSTM', bii = True, time = True) 
    # print(f'embed_dim {embed_dim_expo}, hidden_size {hidden_size_expo}', '\n')
    # ehr_model = model.EHR_RNN(input_size_1, embed_dim = 256, hidden_size = 256, n_layers = 1, dropout_r = 0., cell_type = 'LSTM', bii = True, time = True) 

    if use_cuda: ehr_model = ehr_model.cuda()  
    torch.cuda.empty_cache() 

    if optimizer_name == 'Adam':
        optimizer = optim.Adam(ehr_model.parameters(), lr=10**lr_expo, weight_decay = 10**l2_expo, eps = 10**eps_expo) 
    elif optimizer_name == 'Adagrad':
        optimizer = optim.Adagrad(ehr_model.parameters(), lr=10**lr_expo, weight_decay = 10**l2_expo, eps = 10**eps_expo) 
    elif optimizer_name == 'Adamax':
        optimizer = optim.Adamax(ehr_model.parameters(), lr=10**lr_expo, weight_decay = 10**l2_expo, eps = 10**eps_expo) 
    elif optimizer_name == 'Adadelta':
        optimizer = optim.Adadelta(ehr_model.parameters(), lr=10**lr_expo, weight_decay = 10**l2_expo, eps = 10**eps_expo) 
 
    bestValidAuc = 0.0
    bestTestAuc = 0.0
    bestValidEpoch = 0
    train_auc_allep = []
    valid_auc_allep = []
    test_auc_allep = []  
    for ep in range(epochs):
        # start = time.time()
        current_loss, train_loss = ut.trainbatches(train_mbs, model = ehr_model, optimizer = optimizer)
        avg_loss = np.mean(train_loss)
        # train_time = time_consumption_since(start)
        # eval_start = time.time()
        Train_auc, y_real, y_hat = ut.calculate_auc(ehr_model, train_mbs, which_model = w_model)
        valid_auc, y_real, y_hat = ut.calculate_auc(ehr_model, valid_mbs, which_model = w_model)
        TestAuc, y_real, y_hat = ut.calculate_auc(ehr_model, test_mbs, which_model = w_model)
        # eval_time = time_consumption_since(eval_start)

        # train_auc_allep.append(Train_auc)
        # valid_auc_allep.append(valid_auc)
        # test_auc_allep.append(TestAuc)

        # print(torch.cuda.memory_allocated())
        if valid_auc > bestValidAuc: 
            bestValidAuc = valid_auc
            bestValidEpoch = ep
            bestTestAuc = TestAuc
            ###uncomment the below lines to save the best model parameters
            best_model = ehr_model
            # torch.save(best_model, 'Readm_test.pth')
            # torch.save(best_model.state_dict(), 'Readm_test.st')
        if ep - bestValidEpoch > 10: break

    bestValidAuc = round(bestValidAuc * 100, 2)

    return bestValidAuc

if __name__ == "__main__":
    start_tm = datetime.now()
    output_folder_name = 'Results_20210610_three_month_98236'
    part_common_path = func.get_common_path('../Results/For_ML/')  + output_folder_name + '/'
    results_output_path = func.get_common_path('../Test_results/')
    DL_results_output_path = part_common_path + '20210611_LSTM_tuning_ischemic_90k_11files.csv'

    selected_list = ['Results_all_input_dischar_tm_7d_365d_window', 'Results_all_input_dischar_tm_366d_730d_window', 'Results_all_input_dischar_tm_181d_365d_window', 'Results_all_input_dischar_tm_366d_546d_window', 'Results_all_input_dischar_tm_547d_730d_window', 'Results_all_input_dischar_tm_181d_270d_window', 'Results_all_input_dischar_tm_271d_365d_window', 'Results_all_input_dischar_tm_366d_455d_window', 'Results_all_input_dischar_tm_456d_545d_window', 'Results_all_input_dischar_tm_546d_635d_window', 'Results_all_input_dischar_tm_636d_730d_window']

    # selected_list = ['Results_all_input_dischar_tm_366d_546d_window','Results_all_input_dischar_tm_547d_730d_window', 'Results_all_input_dischar_tm_366d_730d_window']

    # store the result and output to csv
    tuning_results_list = []
    # specific_folder_name = specific_folder_name_list[-1]

    for idx_folder, specific_folder_name in enumerate(selected_list):
        common_path = part_common_path + specific_folder_name + '/'        
        # file_type_list = ['ischemic', 'bleeding']
        file_type_list = ['ischemic']
        for idx_file_type, file_type in enumerate(file_type_list):
            print(f'{idx_folder+1}-{idx_file_type}. Running DL for {file_type} file from {specific_folder_name}:', '\n')
            
            train_sl, test_sl, valid_sl, types_d = load_input_pkl(file_type, common_path)

            types_d_rev = dict(zip(types_d.values(), types_d.keys()))
            input_size_1 = [len(types_d_rev) + 1]
            print(f'len(train_sl) is {len(train_sl)}, len(valid_sl) is {len(valid_sl)}, len(test_sl) is {len(test_sl)}, input_size_1 is {input_size_1}', '\n')
            train_sl = remove_long_code_pt_original(train_sl, 300)
            test_sl = remove_long_code_pt_original(test_sl, 300)
            valid_sl = remove_long_code_pt_original(valid_sl, 300)
            print(f'new len(train_sl) is {len(train_sl)}, new len(valid_sl) is {len(valid_sl)}, new len(test_sl) is {len(test_sl)}, input_size_1 is {input_size_1}', '\n')
       
            to_convert_data_label = 1
            if to_convert_data_label:
                # train set
                pts_tr = []
                labels_tr = []
                features_tr = []
                for pt in train_sl:
                    pts_tr.append(pt[0])
                    labels_tr.append(pt[1])
                    x = []
                    for v in pt[-1]:
                        x.extend(v[-1])
                    features_tr.append(x)

                # test set
                pts_t = []
                labels_t = []
                features_t = []
                for pt in test_sl:
                    pts_t.append(pt[0])
                    labels_t.append(pt[1])
                    x = []
                    for v in pt[-1]:
                        x.extend(v[-1])
                    features_t.append(x)
                
                mlb = MultiLabelBinarizer(classes = range(input_size_1[0])[1:])
                nfeatures_tr = mlb.fit_transform(features_tr)
                nfeatures_t = mlb.fit_transform(features_t)

            # train = EHRDataset(train_sl, sort = True, model = 'RNN')
            # train_mbs = list(EHRdataloader(train, batch_size = 128, packPadMode = True))
            # valid = EHRDataset(valid_sl, sort = True, model ='RNN')
            # valid_mbs = list(EHRdataloader(valid, batch_size = 128, packPadMode = True))
            # test = EHRDataset(test_sl, sort = True, model = 'RNN')
            # test_mbs = list(EHRdataloader(test, batch_size = 128, packPadMode = True))

            to_tune_LSTM_parameter = 1
            if to_tune_LSTM_parameter:
                tuning_start_time = datetime.now()                        
                print(f'{idx_folder+1}-{idx_file_type+1}. Start LSTM hyper-paramater tunning {file_type}-{specific_folder_name}, time is {tuning_start_time}', '\n')                             
                study = optuna.create_study(direction="maximize")
                study.optimize(objective_LSTM, n_trials=100)
                print("Number of finished trials: ", len(study.trials))
                print("Best trial:")
                trial = study.best_trial
                print("  Value: {}".format(trial.value))
                print("  Params: ")
                for key, value in trial.params.items():
                    print("    {}: {}".format(key, value)) 
                tuning_row = [file_type, specific_folder_name, 'LSTM', trial.value, trial.params]
                tuning_results_list.append(tuning_row)
                # print(RNN_tuning_row)
                tuning_end_time = datetime.now() 
                consumed_time = tuning_end_time - tuning_start_time
                print(f'{idx_folder+1}-{idx_file_type+1}. End LSTM hyper-paramater tunning {file_type}-{specific_folder_name}, tuning_end_time is {tuning_end_time}, consumed_time is {consumed_time}', '\n')        

    to_output_DL_tuning_results_label = 1
    if to_output_DL_tuning_results_label:
        with open(DL_results_output_path, 'w') as DL_output_file:
            DL_writer = csv.writer(DL_output_file)
            # DL_head_row = ['file_type', 'specific_folder_name', 'DL', 'bestTestAuc', 'len(train_sl)', 'len(valid_sl)', 'len(test_sl)', 'lr', 'l2', 'repeat times', 'bestValidAuc', 'average_test_auc_label']
            # RNN_row = [file_type, specific_folder_name, 'RNN', bestTestAuc, len(train_sl), len(valid_sl), len(test_sl), lr, l2, idx_repeat+1, bestValidAuc, bestValidEpoch, 0]
            tuning_head_row = ['file_type', 'specific_folder_name', 'LSTM', 'trial.value', 'trial.params']
            # DL_writer.writerow(DL_head_row) 
            DL_writer.writerow(tuning_head_row)             
            sorted_DL_lists1 = sorted(tuning_results_list, key = lambda x:(x[1]), reverse=False)
            sorted_DL_lists2 = sorted(sorted_DL_lists1, key = lambda x:(x[0]), reverse=False)
            DL_writer.writerows(sorted_DL_lists2)

    print(f'Finishing running LSTM tuning, time now is {datetime.now()}, total consumed time is {datetime.now() - start_tm}')
    print('Congrats!')

