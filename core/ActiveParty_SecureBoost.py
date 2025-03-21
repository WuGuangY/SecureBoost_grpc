import os
import json
import threading
import numpy as np
import requests
import pandas as pd
from collections import deque
from sklearn.model_selection import train_test_split

from protos import Server_pb2
from protos import Server_pb2_grpc
import grpc
from phe import paillier, PaillierPublicKey,PaillierPrivateKey

from core.BQ_Boost import BQ_Boost
from core.DI_Boost import DI_Boost,f2i,i2f
from core.Calculator import Calculator
from utils.log import logger
from utils.params import temp_root, pp_list, pub_key_list,  stub_list
from utils.encryption import serialize_pub_key, serialize_encrypted_number, load_encrypted_number
from utils.decorators import broadcast, use_thread, poll
from msgs.messages import msg_empty, msg_name_file, msg_gradient_file, msg_split_confirm, msg_name_space_lookup

stub=stub_list[0]

class ActiveParty:
    def __init__(self) -> None:
        Calculator.load_config()

        self.id = 0
        self.name = f'party{self.id}'
        self.pub_key, self.pri_key = paillier.generate_paillier_keypair(n_length=Calculator.keypair_gen_length)
        logger.info(f'{self.name.upper()}: Paillier key generated. ')

        self.dataset = None
        self.testset = None

        self.model = Model()

        self.passive_port = {}  # 被动方的名称 - 被动方对象对应
        self.reverse_passive_port = {}  # 被动方对象 - 名称对应

        # 训练中临时变量
        self.cur_split_node = None  # 当前正在分裂的节点
        self.split_nodes = deque()  # 待分裂的节点队列
        self.cur_preds = None
        self.feature_split_time = None  # 各特征的分裂次数

    def load_dataset(self, data_path: str, test_path: str = None):
        """
        加载数据集
        """
        dataset = pd.read_csv(data_path)

        dataset['id'] = dataset['id'].astype(str)

        # 特征编码
        for df in [dataset]:
            df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
            df['Vehicle_Damage'] = df['Vehicle_Damage'].map({'Yes': 1, 'No': 0})
            df['Vehicle_Age'] = df['Vehicle_Age'].map({'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2})

        # for column in dataset.columns:
        #     if column != 'id':
        #         dataset[column] = dataset[column].apply(int)

        self.dataset = dataset.set_index('id')

        # 将数据集划分为训练集和测试集
        self.dataset, self.testset = train_test_split(self.dataset, test_size=0.1, random_state=42)

        self.cur_preds = pd.Series(Calculator.init_pred, index=self.dataset.index)
        self.feature_split_time = pd.Series(0, index=self.dataset.index)

    def broadcast_pub_key(self):
        """
        将公钥广播给所有被动方
        """
        pub_dict = serialize_pub_key(self.pub_key)
        file_name = os.path.join(temp_root['file'][self.id], f'active_pub_key.json')
        with open(file_name, 'w+') as f:
            json.dump(pub_dict, f)

        def send_pub_key(file_name: str):
            data = msg_name_file(self.name, file_name)
            logger.info(f'Sending public key. ')
            for pp in pp_list:
                logger.debug(f'Request data: {data}. ')
                recv_data = pp.recv_active_pub_key(data['file_name'])
                self.passive_port[recv_data['party_name']] = pp
                self.reverse_passive_port[pp] = recv_data['party_name']

        send_pub_key(file_name)
        logger.info(f'{self.name.upper()}: Public key broadcasted to all passive parties. ')

    def sample_align(self):
        """
        样本对齐
        """

        # # 被动方计算样本列表的哈希值
        # logger.debug(f'{self.name.upper()}: Initiating sample alignment. ')
        # for pp in pp_list:
        #     pp.init_sample_align()

        #通知被动方计算样本列表的哈希值
        stub=stub_list[0]

        # stub.ACallP_sample_align(Server_pb2.ServerRequest(message='sample_align'))
        
        def protobuf_to_dict(hash_response):
                idx_map_dict={}
                for pair in hash_response.hash_pairs:
                    idx_map_dict[pair.hash] = pair.original_index
                hash_key_dict=[]
                for key in hash_response.hash_keys:
                    hash_key_dict.append(key)
                return idx_map_dict,hash_key_dict

        # 主动方计算样本列表哈希值
        train_idx = self.dataset.index.tolist()
        request = Server_pb2.IndexRequest(indices=train_idx)
        response = stub.ComputeHash(request)
        train_idx_map,train_hash = protobuf_to_dict(response)
        train_hash=set(train_hash)

        if self.testset is not None:
            valid_idx = self.testset.index.tolist()
            request = Server_pb2.IndexRequest(indices=valid_idx)
            response = stub.ComputeHash(request)
            valid_idx_map,valid_hash = protobuf_to_dict(response)   
            valid_hash=set(valid_hash)    

        # 被动方返回哈希后的样本列表,求交集
        response = stub.AWaitForMessage(Server_pb2.Empty())
        # 解析返回的 JSON 数据
        try:
            hash_data = json.loads(response.json_data)['data']
            # print(f"收到服务器的 JSON 数据: {hash_data}")
        except json.JSONDecodeError:
            print("服务器返回的数据不是有效的 JSON")
        # 求交集
        train_hash = train_hash.intersection(set(hash_data['train_hash']))
        if self.testset is not None:
            valid_hash = valid_hash.intersection(set(hash_data.get('valid_hash', [])))

        logger.info(
            f'{self.name.upper()}: Sample alignment finished. Intersect trainset contains {len(train_hash)} samples. ')

        # 存储样本列表哈希值
        json_data = {'train_hash': list(train_hash)}
        if self.testset is not None:
            json_data['valid_hash'] = list(valid_hash)

        file_name = os.path.join(temp_root['file'][self.id], f'sample_align.json')
        with open(file_name, 'w+') as f:
            json.dump(json_data, f)

        # 向所有被动方发送求交后的样本下标
        response = stub.ASendMessage(Server_pb2.MessageRequest(json_data=json.dumps(json_data)))

        self.dataset = self.dataset.loc[[train_idx_map[th] for th in train_hash], :]
        if self.testset is not None:
            self.testset = self.testset.loc[[valid_idx_map[vh] for vh in valid_hash], :]

    def train_status(self):
        """
        根据训练状态进行相关初始操作
        """
        if len(self.split_nodes) != 0:  # 还有待分裂节点，继续训练
            return True
        # 待分裂节点为空，则根据回归树的数量判断是否结束训练
        if len(self.model) < Calculator.max_trees:  # 树的数量未达到上限，则根据上一棵树更新数据，并将新一棵树加入
            logger.info(f'{self.name.upper()}: No pending node, creating new tree, index {len(self.model)}. ')
            new_root = self.create_new_tree()  # 更新权重并新建一棵树
            self.split_nodes.append(new_root)  # 加入待分裂节点队列
            self.model.append(new_root)  # 加入模型
            self.update_gradients()  # 根据当前节点的预测值更新模型梯度g,h
            return True
        else:
            logger.info(f'{self.name.upper()}: Training completed. ')
            self.create_new_tree()  # 更新最后一棵树的叶子节点权重
            return False
        
    def train_status_DI(self):
        """
        根据训练状态进行相关初始操作
        """
        if len(self.split_nodes) != 0:          # 还有待分裂节点，继续训练
            return True
        # 待分裂节点为空，则根据回归树的数量判断是否结束训练
        if len(self.model) < Calculator.max_trees:          # 树的数量未达到上限，则根据上一棵树更新数据，并将新一棵树加入
            logger.info(f'{self.name.upper()}: No pending node, creating new tree, index {len(self.model)}. ')
            new_root = self.create_new_tree()           # 更新权重并新建一棵树
            self.split_nodes.append(new_root)           # 加入待分裂节点队列
            self.model.append(new_root)                 # 加入模型
            self.update_gradients_DI()         # 根据当前节点的预测值更新模型梯度g,h
            return True
        else:
            logger.info(f'{self.name.upper()}: Training completed. ')
            self.create_new_tree()                  # 更新最后一棵树的叶子节点权重
            return False

    def create_new_tree(self):
        """
        向模型中新建一棵树，并计算上一棵树的叶子节点权重，更新对样本的预测值
        """
        if len(self.model) > 0:
            root = self.model[-1]  # 获取刚建完的树
            for leaf in root.get_leaves():
                instance_space = leaf.instance_space  # 节点包含的样本空间（只在训练和预测过程中有用，不会存储）
                leaf.weight = Calculator.leaf_weight(self.model.grad, self.model.hess, instance_space)  # 计算叶子节点的权重
                self.cur_preds[instance_space] += leaf.weight  # 更新该叶子节点中所有样本的预测值,整棵树结束了才会更新
            logger.info(
                f'{self.name.upper()}: Accuracy after tree {len(self.model) - 1}: {Calculator.accuracy(self.dataset["y"], self.cur_preds)}')  # 计算准确率
        else:
            # 初始化预测值，`Calculator.init_pred` 是一个初始预测值，通常是一个常数值
            self.cur_preds = pd.Series(Calculator.init_pred, index=self.dataset.index)

        new_root = TreeNode(0, self.dataset.index.tolist())
        return new_root

    def update_gradients(self):
        g, h = Calculator.grad(self.cur_preds, self.dataset)
        self.model.update_gradients(g, h, self.pub_key)

    def update_gradients_DI(self):
        g, h = Calculator.grad(self.cur_preds, self.dataset)
        self.model.update_gradients_DI(g, h, self.pub_key)

    def splits_score(self, instance_space: list) -> tuple:
        """
        主动方计算最佳分裂点，返回最佳分裂点信息
        """
        local_best_split = None  # 存储当前最佳的分裂点
        # 遍历数据集中的每一个特征
        for feature in [col for col in self.dataset.columns if col != 'y']:
            feature_values = self.dataset.loc[instance_space, feature].sort_values(ascending=True)
            split_indices = [int(qt * len(feature_values)) for qt in Calculator.quantile]

            for si in split_indices:
                left_space, right_space = feature_values.iloc[si:].index.tolist(), feature_values.iloc[
                                                                                   :si].index.tolist()
                thresh = feature_values.iloc[si]

                # 使用主动方的模型梯度和hess矩阵，计算当前分裂点的分裂得分
                split_score = Calculator.split_score_active(self.model.grad, self.model.hess, left_space, right_space)

                # 如果当前分裂点得分比之前的最佳分裂点得分更优，则更新最佳分裂点
                if not local_best_split or local_best_split[2] < split_score:
                    local_best_split = (feature, thresh, split_score, left_space)

        return local_best_split

    def passive_best_split_score(self, splits_file: str, full_grad_sum: float, full_hess_sum: float) -> tuple:
        """
        一个被动方的最佳分裂点增益
        """
        with open(splits_file, 'r') as f:
            splits_data = json.load(f)

        local_best_split = None
        stub=stub_list[0]

        # 调用 GetPublicKey 方法
        response = stub.GetPublicKey(Server_pb2.Empty())

        # 将字节流反序列化为公钥
        n = int.from_bytes(response.n, byteorder='big')
        self.pub_key = PaillierPublicKey(n=n)

        # 调用 GetPrivateKey 方法
        response = stub.GetPrivateKey(Server_pb2.Empty())

        # 将字节流反序列化为私钥
        p = int.from_bytes(response.p, byteorder='big')
        q = int.from_bytes(response.q, byteorder='big')
        self.pri_key = PaillierPrivateKey(public_key=self.pub_key, p=p, q=q)

        # #从服务器获取解密后的splits_data
        # splits_request = Server_pb2.SplitsRequest(splits_data=[Server_pb2.SplitData(grad_left=split['grad_left'], 
        #                                                                                           hess_left=split['hess_left'])
        #                                                                                           for split in splits_data])
        # response = stub.GetDecryptedSplits(splits_request)

        # for decrypted_split in response.decrypted_splits_data:
        #     split_score = Calculator.split_score_passive(decrypted_split.grad, decrypted_split.hess, full_grad_sum, full_hess_sum)
        #     if not local_best_split or split_score > local_best_split[1]:
        #         local_best_split = (decrypted_split.idx, split_score)

        # aaaa


        for idx, split in enumerate(splits_data):

            left_grad_sum, left_hess_sum = load_encrypted_number(split['grad_left'],self.pub_key), load_encrypted_number(split['hess_left'], self.pub_key)  # 反序列化
            left_grad_sum, left_hess_sum = self.pri_key.decrypt(left_grad_sum), self.pri_key.decrypt(left_hess_sum)  # 解密
            split_score = Calculator.split_score_passive(left_grad_sum, left_hess_sum, full_grad_sum, full_hess_sum)
            if not local_best_split or split_score > local_best_split[1]:
                local_best_split = (idx, split_score)

        return local_best_split
    
    def passive_best_split_score_DI(self, splits_file: str, full_grad_sum: float, full_hess_sum: float) -> tuple:
        """
        一个被动方的最佳分裂点增益
        """
        with open(splits_file, 'r') as f:
            splits_data = json.load(f)
        
        local_best_split = None
        for idx, split in enumerate(splits_data):
            # 解密
            left_grad_sum,left_hess_sum,left_num=split['grad_left'],split['hess_left'],split['nums']
            left_grad_sum=i2f(left_grad_sum)
            left_hess_sum=i2f(left_hess_sum)

            key_list=self.model.keylist
            left_grad_sum=left_grad_sum/key_list[2]*key_list[1]-key_list[0]*left_num
            left_hess_sum=left_hess_sum/key_list[3]

            split_score = Calculator.split_score_passive(left_grad_sum, left_hess_sum, full_grad_sum, full_hess_sum)
            if not local_best_split or split_score > local_best_split[1]:
                local_best_split = (idx, split_score)
        
        return local_best_split

    def train(self):
        # pub_key_list.append(self.pub_key)
        # self.broadcast_pub_key() 不需要广播公钥
        self.sample_align()
        while self.train_status():  # 树没建完
            stub=stub_list[0]

            spliting_node = self.split_nodes.popleft()
            # 检查叶子节点能否继续分裂（到达深度 / 样本过少都会停止分裂）
            if not spliting_node.splitable():
                continue

            #告诉被动方继续建树
            stub.ASendMessage(Server_pb2.MessageRequest(json_data="3")) #3代表继续建树

            logger.info(f'{self.name.upper()}: Splitting node {spliting_node.id}. ')

            # 准备好本节点训练所用的文件
            instance_space_file = os.path.join(temp_root['file'][self.id], f'instance_space.json')
            instance_space = spliting_node.instance_space
            with open(instance_space_file, 'w+') as f:
                json.dump(instance_space, f)        

            # 向被动方发送梯度信息
            data = msg_gradient_file(self.name, instance_space_file, self.model.grad_file, self.model.hess_file)

            #获取存储梯度信息的文件：data['instance_space'],data['grad'],data['hess']
            file_list=[data['instance_space'],data['grad'],data['hess']]
            files=[]
            for file in file_list:
                with open(file, 'rb') as f:
                    data = f.read()       
                files.append(Server_pb2.FileRequest.FileInfo(file=data,name=file,party_name=self.name),)
            #调用rpc
            response = stub.ASendFile(Server_pb2.FileRequest(files=files))

            # json_data=json.dumps(data)
            # stub.ASendMessage(Server_pb2.MessageRequest(json_data=json_data))

            logger.info(f'{self.name.upper()}: Gradients broadcasted to all passive parties. ')

            # 主动方计算最佳分裂点
            local_best_split = self.splits_score(instance_space)
            global_splits = (self.name, 0, local_best_split[2])  # 全局最优分裂点 (训练方名称, 分裂点存储下标, 分裂增益)
            logger.info(f'{self.name.upper()}: Active best split point confirmed. ')

            full_grad_sum, full_hess_sum = self.model.grad[instance_space].sum(), self.model.hess[instance_space].sum()

            # 收集被动方的梯度信息，并确定最佳分裂点
            response = stub.AGetFile(Server_pb2.Empty())

            file_info = response.files[0]

            file_data = file_info.file  # 接收到的文件
            file_name = file_info.name  # 接收到的文件名称
            file_sender = file_info.party_name

            f = open(file_name, "wb")  # 把传来的文件写入本地磁盘
            f.write(file_data)  # 写文件
            f.close()  # 关闭IO

            logger.info(f'{self.name.upper()}: Received split sum from {file_sender}. ')
            passive_best_split = self.passive_best_split_score(file_name, full_grad_sum,
                                                                full_hess_sum)  # (idx, split_score)
            if passive_best_split[1] > global_splits[2]:
                global_splits = ("party-1", passive_best_split[0], passive_best_split[1])


            logger.info(f'{self.name.upper()}: Global best split point confirmed, belongs to {global_splits[0]} with gain {global_splits[2]}. ')
            
            if global_splits[2] < 0:  # 如果分裂增益 < 0 则直接停止分裂
                stub.ASendMessage(Server_pb2.MessageRequest(json_data="0"))
                continue

            # 根据最佳分裂点进行分裂 / 请求被动方确认
            if global_splits[0] == self.name:  # 最佳分裂点属于主动方
                feature_split = {
                    'feature_name': local_best_split[0],
                    'feature_thresh': local_best_split[1]
                }
                left_space = local_best_split[3]
                param = {
                    'party_name': self.name,
                    'left_space': left_space,
                    'feature_split': feature_split
                }
                logger.info(f'{self.name.upper()}: Splitting on {feature_split["feature_name"]}. ')
                #告知被动方无需分类
                stub.ASendMessage(Server_pb2.MessageRequest(json_data="0"))
            else:
                data = msg_split_confirm(global_splits[0], global_splits[1])
                stub.ASendMessage(Server_pb2.MessageRequest(json_data=json.dumps(data)))
                #接受被动方分裂信息
                recv_data = json.loads(stub.AWaitForMessage(Server_pb2.Empty()).json_data)['data']
                look_up_id = recv_data['split_index']
                logger.debug(f'Received confirm data, look up index: {look_up_id}. ')

                # 获得分裂后的左空间
                left_space = recv_data['left_space']
                # with open(recv_data['left_space'], 'r') as f:
                #     left_space = json.load(f)
                param = {
                    'party_name': global_splits[0],
                    'left_space': left_space,
                    'look_up_id': look_up_id
                }
            left_node, right_node = spliting_node.split(**param)
            self.split_nodes.extend([left_node, right_node])
        #告诉被动方建树完毕
        stub.ASendMessage(Server_pb2.MessageRequest(json_data="2")) #2代表建树完毕

    def train_DI(self):
        # pub_key_list.append(self.pub_key)
        # self.broadcast_pub_key() 不需要广播公钥
        self.sample_align()
        while self.train_status_DI(): # 树没建完
            stub=stub_list[0]

            spliting_node = self.split_nodes.popleft()
            # 检查叶子节点能否继续分裂（到达深度 / 样本过少都会停止分裂）
            if not spliting_node.splitable():
                continue
            logger.info(f'{self.name.upper()}: Splitting node {spliting_node.id}. ')

            #告诉被动方继续建树
            stub.ASendMessage(Server_pb2.MessageRequest(json_data="3")) #3代表继续建树

            # 准备好本节点训练所用的文件
            instance_space_file = os.path.join(temp_root['file'][self.id], f'instance_space.json')
            instance_space = spliting_node.instance_space
            with open(instance_space_file, 'w+') as f:
                json.dump(instance_space, f)

            # 向被动方发送梯度信息           
            data = msg_gradient_file(self.name, instance_space_file, self.model.grad_file, self.model.hess_file)

            #获取存储梯度信息的文件：data['instance_space'],data['grad'],data['hess']
            file_list=[data['instance_space'],data['grad'],data['hess']]
            files=[]
            for file in file_list:
                with open(file, 'rb') as f:
                    data = f.read()       
                files.append(Server_pb2.FileRequest.FileInfo(file=data,name=file,party_name=self.name),)
            #调用rpc
            response = stub.ASendFile(Server_pb2.FileRequest(files=files))

            # json_data=json.dumps(data)
            # stub.ASendMessage(Server_pb2.MessageRequest(json_data=json_data))

            logger.info(f'{self.name.upper()}: Gradients broadcasted to all passive parties. ')

            # 主动方计算最佳分裂点
            local_best_split = self.splits_score(instance_space)
            global_splits = (self.name, 0, local_best_split[2])         # 全局最优分裂点 (训练方名称, 分裂点存储下标, 分裂增益)
            logger.info(f'{self.name.upper()}: Active best split point confirmed. ')

            full_grad_sum, full_hess_sum = self.model.grad[instance_space].sum(), self.model.hess[instance_space].sum()

            # 收集被动方的梯度信息，并确定最佳分裂点
            response = stub.AGetFile(Server_pb2.Empty())

            file_info = response.files[0]

            file_data = file_info.file  # 接收到的文件
            file_name = file_info.name  # 接收到的文件名称
            file_sender = file_info.party_name

            f = open(file_name, "wb")  # 把传来的文件写入本地磁盘
            f.write(file_data)  # 写文件
            f.close()  # 关闭IO

            logger.info(f'{self.name.upper()}: Received split sum from {file_sender}. ')
            passive_best_split = self.passive_best_split_score_DI(file_name, full_grad_sum,
                                                                full_hess_sum)  # (idx, split_score)

            if passive_best_split[1] > global_splits[2]:
                global_splits = ('Party-1', passive_best_split[0], passive_best_split[1])


            logger.info(f'{self.name.upper()}: Global best split point confirmed, belongs to {global_splits[0]} with gain {global_splits[2]}. ')

            if global_splits[2] < 0:                    # 如果分裂增益 < 0 则直接停止分裂
                stub.ASendMessage(Server_pb2.MessageRequest(json_data="0"))
                continue
            

            # 根据最佳分裂点进行分裂 / 请求被动方确认
            if global_splits[0] == self.name:           # 最佳分裂点属于主动方
                feature_split = {
                    'feature_name': local_best_split[0], 
                    'feature_thresh': local_best_split[1]
                }
                left_space = local_best_split[3]
                param = {
                    'party_name': self.name, 
                    'left_space': left_space, 
                    'feature_split': feature_split
                }
                logger.info(f'{self.name.upper()}: Splitting on {feature_split["feature_name"]}. ')
                #告知被动方无需分类
                stub.ASendMessage(Server_pb2.MessageRequest(json_data="0"))
            else:
                data = msg_split_confirm(global_splits[0], global_splits[1])
                stub.ASendMessage(Server_pb2.MessageRequest(json_data=json.dumps(data)))
                recv_data = json.loads(stub.AWaitForMessage(Server_pb2.Empty()).json_data)['data']
                look_up_id = recv_data['split_index']
                logger.debug(f'Received confirm data, look up index: {look_up_id}. ')

                # 获得分裂后的左空间
                left_space = recv_data['left_space']
                # with open(recv_data['left_space'], 'r') as f:
                #     left_space = json.load(f)
                param = {
                    'party_name': global_splits[0],
                    'left_space': left_space,
                    'look_up_id': look_up_id
                }
            left_node, right_node = spliting_node.split(**param)
            self.split_nodes.extend([left_node, right_node])

        #告诉被动方建树完毕
        stub.ASendMessage(Server_pb2.MessageRequest(json_data="2")) #2代表建树完毕

    def dump_model(self, file_path: str):
        """
        将模型存储到指定路径
        """
        # 判断给定路径是文件夹还是文件
        dir_path, file_name = os.path.split(file_path)
        if not os.path.exists(dir_path):
            logger.error(f'{self.name.upper()}: Model saving path not exists: \'{dir_path}\'. ')
            return
        if not file_name:
            import time
            file_path = os.path.join(file_path, time.strftime('model%m%d%H%M.json', time.localtime()))

        with open(file_path, 'w+') as f:
            logger.debug(self.model.dump())
            json.dump(self.model.dump(), f)
        logger.info(f'{self.name.upper()}: Model dumped to {file_path}. ')
        return file_path

    def load_model(self, file_path: str):
        """
        从指定路径加载模型
        """
        with open(file_path, 'r') as f:
            model_data = json.load(f)
        for tree_data in model_data:
            self.model.load(tree_data)
        logger.info(f'{self.name.upper()}: Model loaded from {file_path}. ')

    def predict(self):
        """
        对测试集进行预测
        """
        if self.testset is None:
            logger.error(f'{self.name.upper()}: No testset loaded. ')
            return

        logger.info(f'{self.name.upper()}: Start predicting. ')
        self.split_nodes = deque()
        preds = pd.Series(Calculator.init_pred, index=self.testset.index)

        for tree in self.model:
            tree.instance_space = self.testset.index.tolist()
            self.split_nodes.append(tree)  # 树根入队
            while len(self.split_nodes):
                spliting_node = self.split_nodes.popleft()
                instance_space = spliting_node.instance_space
                if not spliting_node.left:  # 为叶子节点
                    preds[instance_space] += spliting_node.weight
                    continue
                elif spliting_node.party_name == self.name:  # 为主动方的分裂节点
                    # 分裂样本空间
                    logger.info(
                        f'{self.name.upper()}: Splitting on node {spliting_node.id} from {spliting_node.party_name}. ')
                    feature_name, feature_thresh = spliting_node.feature_split['feature_name'], \
                    spliting_node.feature_split['feature_thresh']
                    left_space = self.testset.loc[instance_space, feature_name] >= feature_thresh
                    left_space = left_space[left_space].index.tolist()
                    right_space = [idx for idx in instance_space if idx not in left_space]

                    spliting_node.left.instance_space = left_space
                    spliting_node.right.instance_space = right_space

                    self.split_nodes.extend([spliting_node.left, spliting_node.right])
                else:  # 为被动方的分裂节点
                    logger.info(
                        f'{self.name.upper()}: Splitting on node {spliting_node.id} from {spliting_node.party_name}. ')
                    party_name = spliting_node.party_name
                    look_up_id = spliting_node.look_up_id
                    instance_space_file = os.path.join(temp_root['file'][self.id], f'instance_space.json')
                    with open(instance_space_file, 'w+') as f:
                        json.dump(instance_space, f)

                    def get_passive_split(party_name: str, instance_space_file: str, look_up_id: int):
                        pp = self.passive_port[party_name]
                        data = msg_name_space_lookup(party_name, instance_space_file, look_up_id)
                        logger.info(f'{self.name.upper()}: Sending prediction request to {party_name}. ')

                        recv_data = pp.predict(data)

                        with open(recv_data['file_name'], 'r') as f:
                            split_space = json.load(f)
                        left_space, right_space = split_space['left_space'], split_space['right_space']

                        spliting_node.left.instance_space = left_space
                        spliting_node.right.instance_space = right_space

                        self.split_nodes.extend([spliting_node.left, spliting_node.right])

                    get_passive_split(party_name, instance_space_file, look_up_id)

        logger.info(f'{self.name.upper()}: Test accuracy: {Calculator.accuracy(self.testset["y"], preds)}. ')
        logger.info(f'{self.name.upper()}: All finished. ')


class Model:
    def __init__(self, active_idx=0) -> None:
        self.trees = []

        # 原始 & 加密梯度，类型均为 pd.Series
        self.grad = None
        self.hess = None
        self.grad_enc = None
        self.hess_enc = None
        self.grad_file = os.path.join(temp_root['file'][active_idx], f'grad.pkl')
        self.hess_file = os.path.join(temp_root['file'][active_idx], f'hess.pkl')

    def __len__(self):
        return len(self.trees)

    def __getitem__(self, idx):
        return self.trees[idx]

    def append(self, root):
        self.trees.append(root)

    def update_gradients(self, g, h, pub_key):
        """
        更新梯度并加密
        """
        self.grad = g
        self.hess = h

        stub=stub_list[0]

        #同态加密
        # self.encrypt_gradients(pub_key)

        #BQ-Boost
        # 将Pandas Series序列化为JSON字符串
        series1_json = json.dumps(g.to_dict())
        series2_json = json.dumps(h.to_dict())

        request=Server_pb2.SeriesRequest(series_grad=series1_json, series_hess=series2_json)
        
        response = stub.BQBoost(request)
        
        # 反序列化接收到的JSON字符串为Pandas Series
        self.grad_enc = pd.Series(json.loads(response.series_grad))
        self.hess_enc = pd.Series(json.loads(response.series_hess))

        self.grad_enc.to_pickle(self.grad_file)
        self.hess_enc.to_pickle(self.hess_file)

    def update_gradients_DI(self, g, h, pub_key):
        """
        更新梯度并加密
        """
        self.grad = g
        self.hess = h
        stub=stub_list[0]
        
        #DI_Boost
        # 将Pandas Series序列化为JSON字符串
        series1_json = json.dumps(g.to_dict())
        series2_json = json.dumps(h.to_dict())

        request=Server_pb2.SeriesRequest(series_grad=series1_json, series_hess=series2_json)
        
        response = stub.DIBoost(request)
        
        # 反序列化接收到的JSON字符串为Pandas Series
        self.grad_enc = pd.Series(json.loads(response.series_grad))
        self.hess_enc = pd.Series(json.loads(response.series_hess))

        self.keylist = json.loads(response.key_list)
        # key_list[0]=json.loads(response.key_list)

        self.grad_enc.to_pickle(self.grad_file)
        self.hess_enc.to_pickle(self.hess_file)    

    def encrypt_gradients(self, pub_key: PaillierPublicKey):
        """
        将梯度用公钥加密
        """
        from tqdm import tqdm

        with tqdm(total=len(self.grad) * 2) as pbar:
            def encrypt_data(data, pub_key: PaillierPublicKey):
                """
                将 data 加密后转换成字典形式返回
                """
                pbar.update(1)
                enc_data = pub_key.encrypt(data)
                return serialize_encrypted_number(enc_data)

            logger.info(f'Gradients encrypting... ')

            self.grad_enc, self.hess_enc = self.grad.apply(encrypt_data, pub_key=pub_key), self.hess.apply(encrypt_data,
                                                                                                           pub_key=pub_key)
            self.grad_enc.to_pickle(self.grad_file)
            self.hess_enc.to_pickle(self.hess_file)

    def dump(self) -> str:
        data = [tree.dump() for tree in self.trees]
        return data

    def load(self, tree_dict: dict):
        """
        将字典中的数据加载成一棵新树加入模型
        """
        root = TreeNode().load(tree_dict)
        self.trees.append(root)


class TreeNode:
    def __init__(self, id: int = 0, instance_space: list = None) -> None:
        self.id = id  # 节点标号
        self.instance_space = instance_space  # 节点包含的样本空间（只在训练和预测过程中有用，不会存储）

        # 分裂信息
        self.party_name = None  # 该节点所属的训练方
        self.feature_split = None  # 当 self.party_name 为主动方时生效，记录分裂的特征名称和门限值。格式为 {'feature_name': xxx, 'feature_thresh': xxx}
        self.look_up_id = 0  # 当 self.party_name 为被动方时生效，记录该节点的分裂方式存储在被动方的查找表第几行中

        # 左右子树
        self.left = None  # 样本对应特征 >= 门限值时进入左子树
        self.right = None

        # 节点权重
        self.weight = 0.0

    def dump(self):
        """
        将以 self 为树根的树存储在字典中
        """
        data = {'id': self.id}
        if self.left:  # 左右子树一定同时存在或同时不存在，判断左子树即可。不为叶子节点，说明有分裂信息，没有权重
            data['party_name'] = self.party_name

            if self.feature_split:
                data['feature_split'] = self.feature_split
            else:
                data['look_up_id'] = self.look_up_id

            data['left'] = self.left.dump()
            data['right'] = self.right.dump()
        else:
            data['weight'] = self.weight

        # 转换为原生 Python int 类型
        data = self.convert_to_python_int(data)

        return data

    def convert_to_python_int(self, data):
        """
            递归将字典中的所有数值转换为 Python 原生的 int 类型。
        """
        for key, value in data.items():
            if isinstance(value, dict):  # 如果值是字典，递归处理
                data[key] = self.convert_to_python_int(value)
            elif isinstance(value, (int, np.int32, np.int64)):  # 如果值是整数类型
                data[key] = int(value)  # 转换为 Python 原生的 int 类型
        return data

    def load(self, tree_dict: dict):
        """
        根据字典中的数据加载树
        """
        self.id = tree_dict['id']
        if 'party_name' in tree_dict:  # 不为叶子节点
            self.party_name = tree_dict['party_name']
            if 'feature_split' in tree_dict:
                self.feature_split = tree_dict['feature_split']
            else:
                self.look_up_id = tree_dict['look_up_id']
            self.left = TreeNode(0, [])
            self.left.load(tree_dict['left'])
            self.right = TreeNode(0, [])
            self.right.load(tree_dict['right'])
        else:
            self.weight = tree_dict['weight']

        return self

    def splitable(self):
        """
        当节点的标号达到了足够的深度，或者节点的样本数量足够少时，不再尝试分裂
        """
        if self.id >= 2 ** (Calculator.max_depth - 1) - 1:
            return False
        if len(self.instance_space) < Calculator.min_sample:
            return False
        return True

    def split(self, party_name, left_space, feature_split=None, look_up_id=0):
        right_space = list(set(self.instance_space) - set(left_space))
        logger.debug(f'Left space: {len(left_space)}, right space: {len(right_space)}')
        self.party_name = party_name
        self.feature_split = feature_split
        self.look_up_id = look_up_id
        self.left, self.right = TreeNode(self.id * 2 + 1, left_space), TreeNode(self.id * 2 + 2, right_space)
        return self.left, self.right

    def get_leaves(self):
        if not self.left and not self.right:
            yield self
        if self.left:
            yield from self.left.get_leaves()
        if self.right:
            yield from self.right.get_leaves()

