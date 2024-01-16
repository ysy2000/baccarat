# one-hot vector로 embedding layer 통과
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import argparse
import random
import csv
import math
from tqdm import tqdm
import numpy as np
import pandas as pd

import redis
from redisworks import Root

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, ReduceLROnPlateau, MultiStepLR
from torch.autograd import Variable
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from Model.Model_LSTM_em import RNN_model
from Dataset.dataLoader_sales import Load

def get_date_list(types='day', start_datetime='1900-01-01', end_datetime='9999-01-01'):
    res = []
    if types == 'month':
        date_time_start = datetime.strptime(start_datetime, '%Y%m%d')
        date_time_end = datetime.strptime(end_datetime, '%Y%m%d')
        date_diff = date_time_end - date_time_start

        new_time = date_time_start
        while True:

            res.append(new_time.date().strftime("%Y-%m-%d"))
            new_time = new_time +  relativedelta(months=1)
            if (new_time > date_time_end):
                break

        return res

    elif types == 'day' :
        date_time_start = datetime.strptime(start_datetime, '%Y%m%d')
        date_time_end = datetime.strptime(end_datetime, '%Y%m%d')
        date_diff = date_time_end - date_time_start

        new_time = date_time_start
        while True:
            new_time = new_time +  relativedelta(days=1)
            if (new_time > date_time_end):
                break
            new_time = new_time.strftime('%Y-%m-%d')
            res.append(new_time)

def get_features(feat):
    # first: 9가지 item_div01
    item_div01= ['ACC', 'BOTTOM', 'DRESS', 'INNER', 'LOUNGE', 'ONEPC', 'OUTERW', 'SWEATR', 'TOP']
    vectors1 = []
    for f in feat:
        one_hot_vector = [0] * 10
        try: 
            idx = item_div01.index(f)
        except:
            idx = 9
        one_hot_vector[idx] += 1
        vectors1.append(one_hot_vector)
        # vectors1.append(idx)
    
    return torch.tensor(vectors1)

## List to Tensor
def get_padded_set(set, batch_size, seq_length, feat_length):   ##### feat_length = 2

    inputs,labels, r_values, input_lengths, feats = [], [], [], [], []
    pad_num = 0

    for batch in set:

        if len(batch) != batch_size :
            continue

        # seq_lengths = [len(s) - feat_length  for s in batch]   # for total length
        max_seq_size = seq_length - 1

        seqs = torch.zeros(batch_size, max_seq_size)
        seqs = seqs.new_full((batch_size, max_seq_size), pad_num)
        targets = seqs.new_full((batch_size, 1 ), pad_num)
        rep_values = seqs.new_full((batch_size, 1 ), pad_num)
        item_div01s = seqs.new_full((batch_size, 10 ), pad_num)

        for x in range(batch_size):

            sample = batch[x]

            input = torch.LongTensor(sample[:-(feat_length+2)])   # last sample: label
            label = sample[-(feat_length+2)]                      # last sample: label
            r_value = sample[-(feat_length+1)]    # representative_value
            feat = sample[-(feat_length):]

            item_div01 = get_features(feat)
    
            seqs[x].copy_(input) # slice by input size
            targets[x] = label  
            rep_values[x] = r_value
            item_div01s[x].copy_(item_div01.flatten())

        inputs.append(seqs)
        labels.append(targets)
        r_values.append(rep_values)
        feats.append(item_div01s)

        # input_lengths.append(seq_lengths)

    return inputs, labels, r_values, feats

def args_setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type =int , default=16)
    parser.add_argument('--valid_batch_size',type = int, default=4)
    parser.add_argument('--test_batch_size',type = int, default=4)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.00625)   # 0.001
    parser.add_argument('--is_bidirectional', default=True)
    parser.add_argument('--is_dropz',type=bool,default=False)
    parser.add_argument('--seq_length', type=int ,default=30)    # train data + label
    parser.add_argument('--split_rate', type=float, default=0.3)
    parser.add_argument('--model', default= 'Conv_LSTM_v3') # not necessary
    parser.add_argument('--scaler', default='StandardScaler')
    parser.add_argument('--reset', default=False)
    parser.add_argument('--ten_percent',type=bool,default=False)
    parser.add_argument('--feat_len', type=int, default=1) 
    parser.add_argument('--model_path', default='/home/ai/Lion/s_saved_model/hyperparams/2023-04-05 01:49:06_sales_em_30_model.pth')

    return parser.parse_args()


def main():

    ## 01. Setup HyperParameter and Some Variables
    args = args_setup()

    # SetUp CUDA
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    random.seed(2022)
    # Setup Redis
    root = Root(host='localhost', port=16379, db=0)

    ## 02. Data Load Variable 
    df_product = None
    df = None

    ## 03. Redis: Data Save and Load - cache
    if args.reset or root.product.value == None or root.sales.value == None:

        print("Load Data and Will be saved in Redis")
        # Load Data
        dataLoader = Load()

        # product
        df_product = dataLoader.loadProduct()

        # stocks (Sales)
        df = dataLoader.loadSales()

        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='raise')
        df['date'] = df['date'].astype(str)

        # Saved Redis
        root.sales.column = df.columns.values.tolist()
        root.sales.value = df.values.tolist()

        root.product.column = df_product.columns.values.tolist()
        root.product.value = df_product.values.tolist()

    else :
        print("Load Data From Redis")
        df = pd.DataFrame(root.sales.value, columns=root.sales.column)
        df_product = pd.DataFrame(root.product.value, columns=root.product.column)

    ## 04. Dataset Load and Preprocess
    new_data_set = []

    date_list = get_date_list('month', '20190101', '20221130')

    len_date = len(date_list)

    for i, row in df_product.dropna(axis=0).iterrows():
        if args.ten_percent and i>2770:
            print("Use only 10% Data")
            break
        tmp = [0]*len_date
        df_new = df.loc[df['item_code'] == row['item_code']]
        month = df_new.groupby(['date',])['qty'].sum().reset_index()
        for j in range(len(date_list)):
            tmp_data = month.loc[month['date'] == date_list[j], ['qty']]['qty'].values
            if tmp_data.size > 0:    # deprecationwarning
                tmp[j] = tmp_data[0]
        new_data_set.append([row['item_code'], tmp, df['item_div01'][i]])

    seq_length = args.seq_length

    ## 05. Dataset Divide by Sequence Lengths
    new_data_set2 = []
    zero_cnt = 0
    dropz=args.is_dropz

    for idx, data in enumerate(new_data_set):
        ave = np.mean(data[data!=0])
        # 평균량이 작은 데이터 학습에서 빼버리기
        if ave < 0.064:
            continue

        for i in range(len(new_data_set[idx][1])-seq_length - 1):
            tmp = []
            for j in range(0, seq_length):
                tmp.append(new_data_set[idx][1][i+j])

            if dropz:
                if tmp[:-1] == [0] * (seq_length - 1):
                    zero_cnt += 1
                    continue

            tmp.append(ave)
            tmp.append(new_data_set[idx][2])    # item_div01
            new_data_set2.append(tmp)

    random.shuffle(new_data_set2) 
    if dropz:
        print(f'zero percent: {round(zero_cnt/(zero_cnt+len(new_data_set2))*100,2)}%')

    # Train / Valid / Test Data Divide - 8:1:1
    train_size = math.floor(len(new_data_set2) * 0.8) 
    valid_size = math.floor(len(new_data_set2) * 0.1) 

    test = new_data_set2

    # Batch Division
    n3 = args.test_batch_size   # 5

    test_set = [test[i * n3:(i + 1) * n3][:] for i in range((len(test) + n3 - 1) // n3 )]

    test_input, test_label, test_rval, test_feats = get_padded_set(test_set, n3, seq_length, args.feat_len) 
    print(f'feature length: {args.feat_len}')

    ## 06. Model Setup 
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    print(f'this is device: {device}')

    model = torch.load(args.model_path)
    model = model.to(device)
    criterion = nn.MSELoss()

    model_name = args.model_path[-30:-4]
    print(f'model_name: {model_name}')

    model.eval()
    total_loss_test = 0

    with torch.no_grad():
        for i, (data) in  enumerate(zip(test_input, test_label, test_rval, test_feats)):
            input, label, r_value, feats = data

            newInput = torch.stack([input], dim=2)
            newInput = newInput.to(device)
            label = label.to(device)
            r_value = r_value.to(device)
            feats = feats.to(device)

            output = model(newInput, feats)
            output = output.to(device)

            # loss_val = criterion(output, label) # MSE
            for x in torch.div(output,r_value):
                if False in torch.isfinite(x):
                    loss_val = criterion(output,label)    # MSE
                else:
                    loss_val = criterion(torch.div(output,r_value),torch.div(label,r_value))    # MSE
            
            loss_val = torch.sqrt(loss_val) # RMSE
            total_loss_test += loss_val.item()

    test_loss = total_loss_test / len(test_input)

    print("Loss : ", test_loss)
    print("-----------------")
    print("Sequence ", seq_length)
    print(f'model name: {model_name}')


if __name__ == '__main__':
    main()