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
from Model.Model_LSTM import RNN_model
from Dataset.dataLoader import Load

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


## List to Tensor
def get_padded_set(set, batch_size, seq_length):

    inputs,labels, r_values, input_lengths = [], [], [], []

    mini = 99999999999
    maxi = 0
    div_num = seq_length + 1    # 전체길이
    pad_num = 0

    for idx, batch in enumerate(set):


        if len(batch) != batch_size :
            continue

        seq_lengths = [len(s) - 2  for s in batch]   # for total length
        max_seq_size = div_num - 2

        seqs = torch.zeros(batch_size, max_seq_size)
        seqs = seqs.new_full((batch_size, max_seq_size), pad_num)
        targets = seqs.new_full((batch_size, 1 ), pad_num)
        rep_values = seqs.new_full((batch_size, 1 ), pad_num)

        for x in range(batch_size):

            sample = batch[x]

            input = torch.LongTensor(sample[:-2])   # last sample: label
            label = sample[-2]                      # last sample: label
            r_value = sample[-1]                    # representative_value

            mini = min(mini, label)
            maxi = max(maxi, label)

            seqs[x].copy_(input) # slice by input size

            targets[x] = label  
            rep_values[x] = r_value

        inputs.append(seqs)
        labels.append(targets)
        r_values.append(rep_values)

        input_lengths.append(seq_lengths)

    return inputs, labels, r_values, input_lengths, mini, maxi



def args_setup():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type =int , default=16)
    parser.add_argument('--valid_batch_size',type = int, default=4)
    parser.add_argument('--test_batch_size',type = int, default=4)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--lr', type=float, default=0.00625)
    parser.add_argument('--is_bidirectional', default=True)
    parser.add_argument('--is_dropz',type=bool,default=False)
    parser.add_argument('--seq_length', type=int ,default=7)    # 2, 3, 4, 5, 6, 7, 8, 9
    parser.add_argument('--split_rate', type=float, default=0.3)
    parser.add_argument('--model', default= 'Conv_LSTM_v3')
    parser.add_argument('--scaler', default='StandardScaler')
    parser.add_argument('--reset', default=False)
    parser.add_argument('--ten_percent',type=bool,default=False)
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
    if args.reset or root.product.value == None or root.inven.value == None:

        print("Load Data and Will be saved in Redis")
        # Load Data
        dataLoader = Load()

        # product
        df_product = dataLoader.loadProduct()

        # stocks (inventory)
        df = dataLoader.loadInven()

        df['month'] = pd.to_datetime(df['month'], format='%Y-%m-%d', errors='raise')
        df['month'] = df['month'].astype(str)

        # Saved Redis
        root.inven.column = df.columns.values.tolist()
        root.inven.value = df.values.tolist()

        root.product.column = df_product.columns.values.tolist()
        root.product.value = df_product.values.tolist()

    else :
        print("Load Data From Redis")
        df = pd.DataFrame(root.inven.value, columns=root.inven.column)
        df_product = pd.DataFrame(root.product.value, columns=root.product.column)


    ## 04. Dataset Load and Preprocess
    new_data_set = []

    date_list = get_date_list('month', '20190101', '20221130')

    len_date = len(date_list)

    #df_product = df_product[0:len(df_product)*0.1]

    for i, row in df_product.iterrows():
        if args.ten_percent and i>277:
            print("Use only 10% Data")
            break
        tmp = [0]*len_date
        df_new = df.loc[df['barcode'] == row['item_code']]
        month = df_new.groupby(['month',])['qty'].sum().reset_index()
        for j in range(len(date_list)):
            tmp_data = month.loc[month['month'] == date_list[j], ['qty']]['qty'].values
            if tmp_data.size > 0:    # deprecationwarning
                tmp[j] = tmp_data[0]
        new_data_set.append([row['item_code'], tmp])

    '''
    Long-term Goal
    [ [barcorder, [month data : 60, 16, 22, ..., 42], ... ]
    '''

    seq_length = args.seq_length


    ## 05. Dataset Divide by Sequence Lengths
    new_data_set2 = []
    zero_cnt = 0
    dropz=args.is_dropz

    for idx, data in enumerate(new_data_set):
        # 0인 데이터 제외 후 평균
        ave = np.mean(data[data!=0])
        # 평균량이 작다는건 예측에 충분히 의미 없는 것, 학습에서 빼버리기
        if ave < 0.065:
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
            new_data_set2.append(tmp)

    random.shuffle(new_data_set2)    
    if dropz:
        print(f'zero percent: {round(zero_cnt/(zero_cnt+len(new_data_set2))*100,2)}%')


    # Train / Valid / Test Data Divide - 8:1:1
    train_size = math.floor(len(new_data_set2) * 0.8) 
    valid_size = math.floor(len(new_data_set2) * 0.1) 
    test_size = math.floor(len(new_data_set2) * 0.1) 

    train = new_data_set2[:train_size]
    valid = new_data_set2[train_size:train_size+valid_size]
    test = new_data_set2[train_size+valid_size:]


    # Batch Division
    n1 = args.batch_size        # 30
    n2 = args.valid_batch_size  # 5
    n3 = args.test_batch_size   # 5

    train_set = [train[i * n1:(i + 1) * n1][:] for i in range((len(train) + n1 - 1) // n1 )]
    valid_set = [valid[i * n2:(i + 1) * n2][:] for i in range((len(valid) + n2 - 1) // n2 )] 
    test_set = [test[i * n3:(i + 1) * n3][:] for i in range((len(test) + n3 - 1) // n3 )]

    train_input, train_label, train_rval, train_input_lengths, train_min, train_max = get_padded_set(train_set, n1, seq_length) 
    valid_input, valid_label, valid_rval, valid_input_lengths, valid_min, valid_max  = get_padded_set(valid_set, n2, seq_length) 
    test_input, test_label, test_rval, test_input_lengths, test_min, test_max = get_padded_set(test_set, n3, seq_length) 

    train_input_lengths = torch.LongTensor(train_input_lengths)
    valid_input_lengths = torch.LongTensor(valid_input_lengths)
    test_input_lengths = torch.LongTensor(test_input_lengths)

    ## 06. Model Setup 
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    model = RNN_model(rnn_input_dims = 1, device = device, hidden_size= 10, seq_length = seq_length-1, n_layers=4, 
                        dropout_p=0.2, 
                        bidirectional=True,
                        rnn_cell='lstm')

    #------------------------------GPU----------------------------------------------------
    model = model.to(device)
    # train_model = nn.DataParallel(model, device_ids = [0,1])
    model = nn.DataParallel(model)

    #------------------------------Loss--------------------------------------------------
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    #scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=0.0001)
    scheduler = MultiStepLR(optimizer, milestones=[30, 50, 100, 200, 300], gamma=0.5)
    #scheduler =ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    #------------------------------Train-------------------------------------------------
    now = datetime.now()
    model_name = str(datetime.now())[:-7]+"_new loss"

    f = open('./log/'+model_name+'_'+str(args.seq_length)+'.txt', 'w')

    min_train_loss = 99999
    min_valid_loss = 99999
    min_epoch = 0
    min_test_loss = 99999

    epochs = args.epoch # 300
    for epoch in range(epochs):

        model.train()
        total_loss = 0
        for i, (data) in  enumerate(zip(train_input, train_label, train_rval, train_input_lengths)):
            input, label, r_value, train_input_length = data
            optimizer.zero_grad()
            # newInput = torch.stack([input,input2,input3], dim=2)
            newInput = torch.stack([input], dim=2)
            newInput = newInput.to(device)
            label = label.to(device)
            r_value = r_value.to(device)
            # train_input_length = train_input_length.to(device)

            output = model(newInput)
            output = output.to(device)

            loss = criterion(torch.div(output,r_value), torch.div(label,r_value)) # MSE
            loss = torch.sqrt(loss) # RMSE
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            scheduler.step()

        total_loss_val = 0
        model.eval()

        with torch.no_grad():
            for i, (data) in  enumerate(zip(valid_input, valid_label, valid_rval ,valid_input_lengths)):
                # print(len(train_input_lengths))
                input, label, r_value, valid_input_length = data
                newInput = torch.stack([input], dim=2)
                newInput = newInput.to(device)
                label = label.to(device)
                r_value = r_value.to(device)
                # input_length = input_length.to(device)

                output = model(newInput)
                output = output.to(device)

                loss_val = criterion(torch.div(output,r_value), torch.div(label,r_value)) # MSE
                loss_val = torch.sqrt(loss_val) # RMSE
                total_loss_val += loss_val.item()

        total_loss_test = 0

        with torch.no_grad():
            for i, (data) in  enumerate(zip(test_input, test_label, test_rval, test_input_lengths)):
                # print(len(train_input_lengths))
                input, label, r_value, test_input_length = data

                newInput = torch.stack([input], dim=2)
                newInput = newInput.to(device)
                label = label.to(device)
                r_value = r_value.to(device)
                # input_length = input_length.to(device)

                output = model(newInput)
                output = output.to(device)

                # loss_val = criterion(output, label) # MSE

                for x in torch.div(output,r_value):
                    if False in torch.isfinite(x):
                        print("Nan value is here")
                        loss_val = criterion(output,label)    # MSE
                    else:
                        loss_val = criterion(torch.div(output,r_value),torch.div(label,r_value))    # MSE
                
                loss_val = torch.sqrt(loss_val) # RMSE
                total_loss_test += loss_val.item()

        train_loss = total_loss / len(train_input) / seq_length
        val_loss = total_loss_val / len(valid_input)
        test_loss = total_loss_test / len(test_input)


        if test_loss < min_test_loss :
            min_epoch = epoch
            min_train_loss = min(train_loss, min_train_loss)
            min_test_loss = min(test_loss, min_test_loss)
            min_valid_loss = min(val_loss, min_valid_loss)
            torch.save(model, '/home/ai/Lion/saved_model/'+model_name +'_'+str(args.seq_length)+'_model.pth')

        # val_loss = 0
        # if (epoch + 1) %  50 == 0:
        # print("Origin Loss :  (train) ", train_loss, " (validation) ", val_loss, " (test) ", test_loss, " at epoch",  (epoch + 1))
        print("Loss :  (train) ", train_loss, " (validation) ", val_loss, " (test) ", test_loss, "  at epoch",  (epoch + 1))
        f.writelines(str(epoch + 1)+","+str(train_loss)+ ","+ str(val_loss)+","+  str(test_loss)+'\n')
    print("Min Loss :  (train) ", min_train_loss, " (validation) ", min_valid_loss, "   (test) ", min_test_loss,  " at epoch",  (min_epoch + 1))
    f.writelines(str(min_epoch + 1)+","+str(min_train_loss)+ ","+ str(min_valid_loss)+","+ str(min_test_loss) + '\n')

    print("-----------------")
    print("Sequence ", seq_length)
    print(f'model name: {model_name}')

    f.close()


if __name__ == '__main__':
    main()