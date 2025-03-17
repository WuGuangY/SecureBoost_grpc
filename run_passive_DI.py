import json
import os
import os
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

from msgs.messages import msg_gradient_file
from protos import Server_pb2
from protos import Server_pb2_grpc

import grpc

from core.Calculator import Calculator
from core.ActiveParty_SecureBoost import ActiveParty
from core.PassivePartty_SecureBoost import PassiveParty

from utils.params import pp_list, pp, stub_list, temp_root
from utils.newUtils import load_dataset, divide_data, vertical_federated_feature_selection, filter_data_by_features, \
    evaluate_model_performance, evaluate_redundancy
from utils.log import logger

#是否继续建立回归树
def continueBuildTree(epoch: int):
    response = stub.PWaitForMessage(Server_pb2.Empty()) #不是文件
    json_data=json.loads(response.json_data)['data']
    if json_data != 3:#非3代表建树完毕
        logger.info(f'{pp.name.upper()}: (epoch : {epoch}) End building tree')
        return False
    logger.info(f'{pp.name.upper()}: (epoch : {epoch}) Continue building tree')
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-n', '--number', dest='passive_num', type=int, default=1)
    args = parser.parse_args()

    
    #读取yaml
    with open('config/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    stub_params = config.get('params', {}).get('stub', {})

    port = stub_params.get('port') #cconfig.get('params')['stub']['port']
    url = stub_params.get('url')
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

    #####################################################################################################################################################

    passive_num = args.passive_num

    ppid = 1

    # 创建被动方
    path_list = [
        f'temp/file/party-{ppid}',
        f'temp/model/party-{ppid}'
    ]

    for path in path_list:
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)

    pp = PassiveParty(ppid)
    pp.dataset = host_train
    pp.testset = host_test
    pp.feature_split_time = pd.Series(0, index=pp.dataset.index)
    pp_list.append(pp)

    options = [
        ('grpc.max_receive_message_length', 100 * 1024 * 1024),  # 接收消息的最大大小
        ('grpc.max_send_message_length', 100 * 1024 * 1024)     # 发送消息的最大大小
    ]

    pathList = ["temp\\file\\party-0","temp\\file\\party-"]
    for path in pathList:
        if not os.path.exists(path):
            os.makedirs(path)

    # 服务器的IP地址
    with grpc.insecure_channel(url+':'+ str(port),options=options) as channel:
        stub = Server_pb2_grpc.ServerStub(channel)
        stub_list[1]=stub

        #被动方对数据的索引进行同态加密
        pp.init_sample_align()

        #被动方将索引信息传递给主动方
        recv_data = pp.get_sample_align()
        # 读取 JSON 文件内容
        with open(recv_data['file_name'], 'r') as f:
            json_data = json.load(f)
        response = stub.PSendMessage(Server_pb2.MessageRequest(json_data=json.dumps(json_data)))
    
        #被动方接受求交后的样本下标
        response = stub.PWaitForMessage(Server_pb2.Empty())
        # 解析返回的 JSON 数据
        try:
            hash_data = json.loads(response.json_data)
            print(f"收到服务器的 JSON 数据: {hash_data}")
            #写入文件sample_align.json
            file_name = os.path.join(temp_root['file'][pp.id], f'sample_align.json')
            with open(file_name, 'w+') as f:
                json.dump(hash_data['data'], f)
        except json.JSONDecodeError:
            print("服务器返回的数据不是有效的 JSON")

        pp.recv_sample_list(file_name)
        epoch = 0
        while continueBuildTree(epoch):
            epoch += 1
            #被动方接受主动方的梯度消息
            response = stub.PGetFile(Server_pb2.Empty())
            files=[]
            for file_info in response.files:
                file_data = file_info.file  # 接收到的文件
                file_name = file_info.name  # 接收到的文件名称
                file_sender = file_info.party_name

                files.append(file_name)

                f = open(file_name, "wb")  # 把传来的文件写入本地磁盘
                f.write(file_data)  # 写文件
                f.close()  # 关闭IO
            logger.info("recvive gradients from A")

            data = msg_gradient_file(pp.name, files[0], files[1], files[2])
            pp.recv_gradients_DI(data)

            #发送给主动方梯度消息
            #获取存储分裂信息的文件
            recv_data = pp.get_splits_sum()
            with open(recv_data["file_name"], 'rb') as f:
                data = f.read()

            files=[]
            files.append(Server_pb2.FileRequest.FileInfo(file=data,name=recv_data["file_name"],party_name=pp.name),)
            #调用rpc
            response = stub.PSendFile(Server_pb2.FileRequest(files=files))
            logger.info(f'{pp.name.upper()}: Sending splits file to A')

            #是否被动方分裂
            response = stub.PWaitForMessage(Server_pb2.Empty())
            json_data=json.loads(response.json_data)['data']
            logger.info(f'{pp.name.upper()}: Waiting for Split or not , result : {json_data}')
            if json_data != 0:#非0代表分裂
                recv_data = pp.confirm_split(json_data)  # 被动方将最佳分裂
                json_data=json.dumps(recv_data)
                stub.PSendMessage(Server_pb2.MessageRequest(json_data=json_data))
     
        #建树完毕，开始评估
        while True:
            # 接受主动方的查询请求 {""party_name": party_name, "look_up_id": look_up_id,"instance_space":instance_space"}
            response = stub.PWaitForMessage(Server_pb2.Empty())
            json_data=json.loads(response.json_data)['data']

            #发送查询结果
            stub.PSendMessage(Server_pb2.MessageRequest(json_data=pp.predict(json_data)))
            
                
