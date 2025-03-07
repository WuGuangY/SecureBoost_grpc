import os
import os
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import grpc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from core.Calculator import Calculator
from core.ActiveParty_SecureBoost import ActiveParty
from core.PassivePartty_SecureBoost import PassiveParty

from protos import Server_pb2
from protos import Server_pb2_grpc

from vertiFed.verParams import pp, pp_list
from vertiFed.evaluation import evaluate_model_performance, evaluate_redundancy

from utils.params import pp_list, pp, stub_list
from utils.newUtils import load_dataset, divide_data, vertical_federated_feature_selection, filter_data_by_features, \
    evaluate_model_performance, evaluate_redundancy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-n', '--number', dest='passive_num', type=int, default=1)
    args = parser.parse_args()

    #####################################################################################################################################################

    # raw_data, feature_names, sample_indices = load_dataset("D:/Projects/Python_projects/SecureBoostImpl-main(2)/static/data/train_origin.csv")
    raw_data, feature_names, sample_indices = load_dataset(
        "./static/data/train_first_5000.csv")
    # raw_data, feature_names, sample_indices = load_dataset('f./static/data/train_origin.csv')
    # 输出 raw_data、feature_names、sample_indices 的尺寸和类型
    last_column = raw_data[:, -1]  # 提取最后一列
    first_column = raw_data[:, 0]  # 提取第一列，返回一维数组

    # raw_data = raw_data[:, 1:-1]  # 选择从第二列到倒数第二列的所有列
    raw_data = raw_data[:, 1:]  # 选择从第二列到最后的所有列
    selected_features = vertical_federated_feature_selection(raw_data, "variance")  # "variance", "mutual_info"

    raw_data = filter_data_by_features(raw_data, selected_features[0])
    client_data = divide_data(raw_data, feature_names, sample_indices)
    guest_data = client_data['guest']['data']  # ndarray需要被训练函数接收
    host_data = client_data['host']['data']  # ndarray需要被训练函数接收
    #####################################################################################################################################################

    # 转换为DataFrame，适配原load_dataset()函数 ****************************适配尝试**********************************************************
    guest_df = pd.DataFrame(guest_data)
    host_df = pd.DataFrame(host_data)

    # 将第一列 id 值拼接到 DataFrame 的第一列
    guest_df.insert(0, 'id', first_column)
    host_df.insert(0, 'id', first_column)

    # 将最后一列的列名改为 "y"
    guest_df.rename(columns={guest_df.columns[-1]: 'y'}, inplace=True)
    # host_df.rename(columns={host_df.columns[-1]: 'y'}, inplace=True)

    # 将第一列的列名改为 "id"
    guest_df.rename(columns={guest_df.columns[0]: 'id'}, inplace=True)
    host_df.rename(columns={host_df.columns[0]: 'id'}, inplace=True)
    guest_df['id'] = guest_df['id'].astype(str)
    host_df['id'] = host_df['id'].astype(str)
    # 将第一列 "id" 设置为 DataFrame 的索引
    guest_df.set_index('id', inplace=True)
    host_df.set_index('id', inplace=True)

    # 将所有数字列转换为整数
    guest_df = guest_df.apply(pd.to_numeric, errors='ignore').astype(int)
    host_df = host_df.apply(pd.to_numeric, errors='ignore').astype(int)

    test_size = 0.1
    random_state = 42
    guest_train, guest_test = train_test_split(guest_df, test_size=test_size, random_state=random_state)
    host_train, host_test = train_test_split(host_df, test_size=test_size, random_state=random_state)


    # 创建主动方
    id = 0
    path_list = [
        f'temp/file/party-{id}',
        f'temp/model/party-{id}'
    ]

    for path in path_list:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    ap = ActiveParty()
    ap.dataset = guest_train
    ap.testset = guest_test
    ap.feature_split_time = pd.Series(0, index=ap.dataset.index)
    ap.cur_preds = pd.Series(Calculator.init_pred, index=ap.dataset.index)
    ap.feature_split_time = pd.Series(0, index=ap.dataset.index)

    options = [
        ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 接收消息的最大大小
        ('grpc.max_send_message_length', 100 * 1024 * 1024)     # 发送消息的最大大小
    ]
    # 服务器的IP地址
    with grpc.insecure_channel('192.168.59.1:50051',options=options) as channel:
        print("Send Hello")
        stub = Server_pb2_grpc.ServerStub(channel)
        stub_list[0]=stub
        
        # ap.train()
        ap.train_DI()
        file_name = ap.dump_model('./static/model/')
        ap.load_model(file_name)
        evaluate_model_performance(ap.model, ap.testset, selected_features, ap.passive_port)    # 集成了predict
        # evaluate_redundancy(ap.model, client_data['guest']['feature_indices'])