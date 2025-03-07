import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from protos import Server_pb2
from protos import Server_pb2_grpc

from phe import paillier, PaillierPublicKey

from core.Calculator import Calculator
from core.DI_Boost import DI_Boost,f2i,i2f
from utils.log import logger
from utils.params import temp_root, stub_list
from utils.decorators import use_thread
from utils.encryption import load_pub_key, serialize_encrypted_number, load_encrypted_number
from msgs.messages import msg_empty, msg_name_file, msg_split_confirm


def convert_for_json(o):
    """
    递归地将不支持的 NumPy 数据类型转换为 JSON 可序列化的类型。
    """
    if isinstance(o, dict):
        return {k: convert_for_json(v) for k, v in o.items()}
    elif isinstance(o, list):
        return [convert_for_json(element) for element in o]
    elif isinstance(o, (np.int8, np.int16, np.int32, np.int64)):
        return int(o)
    elif isinstance(o, (np.float16, np.float32, np.float64)):
        return float(o)  # 或者使用 o.item()，两者效果相同
    elif isinstance(o, np.ndarray):
        return o.tolist()
    else:
        return o

class PassiveParty:
    def __init__(self, id: int) -> None:
        Calculator.load_config()

        self.id = id
        self.name = f'party{self.id}'          # 被动方名称
        self.active_pub_key = None             # 主动方的公钥，用于数据加密
        self.dataset = None                 # 训练集
        self.testset = None                 # 测试集

        self.look_up_table = pd.DataFrame(columns=['feature_name', 'feature_thresh'])           # 查找表

        # 训练中临时变量
        self.train_status = ''              # 训练状态，取值为 ['Idle', 'Busy', 'Ready']
        self.local_splits = []                # 当前全部可能的分裂点信息
        self.temp_file = ''                 # 临时保存的文件名，当 train_status 为特定值时返回
        self.feature_split_time = None      # 各特征的分裂次数
        

    def load_dataset(self, data_path: str, valid_path: str=None):
        """
        加载数据集
        """
        dataset = pd.read_csv(data_path)

        dataset['id'] = dataset['id'].astype(str)

        # 特征编码
        for df in [dataset]:
            df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

        for column in dataset.columns:
            if column != 'id':
                dataset[column] = dataset[column].apply(int)

        self.dataset = dataset.set_index('id')

        # 将数据集划分为训练集和测试集
        self.dataset, self.testset = train_test_split(self.dataset, test_size=0.1, random_state=42)

        self.feature_split_time = pd.Series(0, index=self.dataset.index)

    def recv_active_pub_key(self, file_name: str):
        """
        保存主动方的公钥，用于梯度计算时的数据处理
        """
        with open(file_name, 'r+') as f:
            pub_dict = json.load(f)
        self.active_pub_key = load_pub_key(pub_dict)
        logger.info(f'{self.name.upper()}: Received public key {str(pub_dict["n"])[:10]}. ')
        return msg_name_file(self.name, '')     # 将本方名称返回给主动方，用于主动方保存名称->端口的映射

    def init_sample_align(self):
        self.sample_align()
        return msg_empty()

    def sample_align(self):
        """
        向主动方返回加密后的样本列表文件
        """
        from utils.sha1 import sha1

        self.train_status = 'Busy'

        stub=stub_list[1]

        # #计算样本的哈希值
        # train_idx = self.dataset.index.tolist()
        # request = Server_pb2.IndexRequest(indices=train_idx)
        # response = stub.ComputeHash(request)
        # def protobuf_to_dict(hash_response):
        #         idx_map_dict={}
        #         for pair in hash_response.hash_pairs:
        #             idx_map_dict[pair.hash] = pair.original_index
        #         hash_key_dict=[]
        #         for key in hash_response.hash_keys:
        #             hash_key_dict.append(key)
        #         return idx_map_dict,hash_key_dict
        
        # train_idx_map,train_hash = protobuf_to_dict(response)

        #计算样本的哈希值
        train_idx = self.dataset.index.tolist()  # 获取训练集样本的索引列表
        train_idx_map = {sha1(idx): idx for idx in train_idx}   # 使用 SHA-1 对每个训练样本的索引进行哈希处理，生成哈希值和原始索引的映射字典
        train_hash = list(train_idx_map.keys())

        # 准备存储映射数据和哈希值的字典
        map_data = {'train_map': train_idx_map}  # 存储训练集索引到哈希值的映射
        json_data = {'train_hash': train_hash}  # 存储训练集样本的哈希值列表

        # 如果验证集存在，则处理验证集的样本哈希值
        if self.testset is not None:

            valid_idx = self.testset.index.tolist()  
            valid_idx_map = {sha1(idx): idx for idx in valid_idx}
            valid_hash = list(valid_idx_map.keys())

            # request = Server_pb2.IndexRequest(indices=valid_idx)
            # response = stub.ComputeHash(request)
            
            # # 调用函数转换protobuf实例为字典
            # valid_idx_map,valid_hash = protobuf_to_dict(response)

            # valid_idx_map,valid_hash=response.hash_pairs,response.hash_keys 
            map_data['valid_map'] = valid_idx_map
            json_data['valid_hash'] = valid_hash

        # 将训练集样本的映射数据保存到 JSON 文件
        map_file = os.path.join(temp_root['file'][self.id], f'idx_map.json')
        with open(map_file, 'w+') as f:
            json.dump(map_data, f) 

        # 将训练集和验证集样本的哈希值保存到另一个 JSON 文件
        sample_file = os.path.join(temp_root['file'][self.id], f'sample_align.json')
        with open(sample_file, 'w+') as f:
            json.dump(json_data, f)

        self.temp_file = sample_file # 将保存的样本哈希值文件路径赋给 temp_file 属性
        self.train_status = 'Ready'
        logger.info(f'{self.name.upper()}: Sample hash finished, ready to return. ')
    
    def get_sample_align(self):
        if self.train_status != 'Ready':
            return msg_empty()
        else:
            self.train_status = 'Idle'
            return msg_name_file(self.name, self.temp_file)

    def recv_sample_list(self, file_name: str) -> str:
        """
        根据主动方对齐后返回的样本列表文件更新本地数据集
        """
        with open(os.path.join(temp_root['file'][self.id], f'idx_map.json'), 'r') as f:
            map_data = json.load(f)

        with open(file_name, 'r') as f:
            hash_data = json.load(f)

        train_idx = [map_data['train_map'][th] for th in hash_data['train_hash']]
        self.dataset = self.dataset.loc[train_idx, :]

        logger.info(f'{self.name.upper()}: Received aligned sample with train length {len(train_idx)}. ')

        if self.testset is not None:
            valid_idx = [map_data['valid_map'][vh] for vh in hash_data['valid_hash']]
            self.testset = self.testset.loc[valid_idx, :]

        return msg_empty()
    
    def recv_gradients(self, recv_dict: dict):
        """
        主动方将数据发送给被动方,被动方根据主动方发来的数据计算可能的最佳分裂点
        """
        self.splits_sum(recv_dict)
        return msg_empty()
    
    def recv_gradients_DI(self, recv_dict: dict):
        """
        主动方将数据发送给被动方,被动方根据主动方发来的数据计算可能的最佳分裂点
        """
        self.splits_sum_DI(recv_dict)
        return msg_empty()
        
    def splits_sum(self, recv_dict: dict):
        """
        根据主动方发来的数据计算可能的最佳分裂点
        参数:
        recv_dict (dict): 包含主动方传递的必要数据，包含：
                      - 'instance_space'：样本空间的文件路径
                      - 'grad'：存储梯度的文件路径
                      - 'hess'：存储海森矩阵的文件路径
        """
        self.train_status = 'Busy'
        assert all(key in recv_dict for key in ['instance_space', 'grad', 'hess']), logger.error('Keys required not found! ')
        logger.debug(f'Begin training. ')

        # 读取样本空间，梯度和海森矩阵的数据
        with open(recv_dict['instance_space'], 'r') as f:
            instance_space = json.load(f)
        grad = pd.read_pickle(recv_dict['grad'])
        hess = pd.read_pickle(recv_dict['hess'])


        # 对梯度和海森矩阵进行加密
        stub=stub_list[1]

        # 将Pandas Series序列化为JSON字符串
        # series1_json = json.dumps(grad.to_dict())
        # series2_json = json.dumps(hess.to_dict())

        # request=Server_pb2.SeriesRequest(series_grad=series1_json, series_hess=series2_json)

        #对梯度和海森矩阵进行解密
        # response = stub.DecryptGradientHessian(request)

        # # 反序列化接收到的JSON字符串为Pandas Series
        # grad = pd.Series(json.loads(response.series_grad))
        # hess = pd.Series(json.loads(response.series_hess))

        #grpc发送和接受文件 https://www.cnblogs.com/ellennet/p/13632848.html


        # 调用 GetPublicKey 方法
        response = stub.GetPublicKey(Server_pb2.Empty())

        # 将字节流反序列化为公钥
        n = int.from_bytes(response.n, byteorder='big')
        pub_key = PaillierPublicKey(n=n)

        # 对梯度和海森矩阵进行解密
        grad = grad.apply(load_encrypted_number, pub_key=pub_key)
        hess = hess.apply(load_encrypted_number, pub_key=pub_key)
        
        # 记录每种分裂方式的梯度和（用于返回给主动方）、分裂方式（用于临时保存），二者下标需要一一对应
        logger.info(f'{self.name.upper()}: Gradients received, start calculating on {len(instance_space)} samples. ')
        self.local_splits = []  # 用于存储本地的分裂点
        cur_splits_sum = []  # 存储每个分裂点的梯度和海森矩阵的和

        # 对数据集中的每个特征（不包括目标变量'y'）进行处理
        for feature in [col for col in self.dataset.columns if col != 'y']:
            feature_values = self.dataset.loc[instance_space, feature].sort_values(ascending=True)      # 取出该特征该样本空间的值,排序
            split_indices = [int(qt * len(feature_values)) for qt in Calculator.quantile]               # 可选的分裂点

            # 对每个候选分裂点，计算左侧样本的梯度和海森矩阵的和
            for si in split_indices:
                left_space = feature_values.iloc[si:].index.tolist()
                thresh = feature_values.iloc[si]  # 分裂阈值
                left_grad_sum, left_hess_sum = grad[left_space].sum(), hess[left_space].sum()

                self.local_splits.append((feature, thresh, left_space))
                
                # 保存分裂点的梯度和海森矩阵的和（加密后的结果）
                cur_splits_sum.append({'grad_left': serialize_encrypted_number(left_grad_sum), 
                                       'hess_left': serialize_encrypted_number(left_hess_sum)})
       
        # 将计算得到的分裂信息保存到文件中
        splits_file = os.path.join(temp_root['file'][self.id], f'splits_file.json')
        with open(splits_file, 'w+') as f:
            json.dump(cur_splits_sum, f)

        self.temp_file = splits_file
        self.train_status = 'Ready'
        logger.info(f'{self.name.upper()}: Split scores finished, ready to return. ')

    def splits_sum_DI(self, recv_dict: dict):
        """
        根据主动方发来的数据计算可能的最佳分裂点
        参数:
        recv_dict (dict): 包含主动方传递的必要数据，包含：
                      - 'instance_space'：样本空间的文件路径
                      - 'grad'：存储梯度的文件路径
                      - 'hess'：存储海森矩阵的文件路径
        """
        self.train_status = 'Busy'
        assert all(key in recv_dict for key in ['instance_space', 'grad', 'hess']), logger.error('Keys required not found! ')
        logger.debug(f'Begin training. ')

        # 读取样本空间，梯度和海森矩阵的数据
        with open(recv_dict['instance_space'], 'r') as f:
            instance_space = json.load(f)
        grad = pd.read_pickle(recv_dict['grad'])
        hess = pd.read_pickle(recv_dict['hess'])  

        # 对梯度和海森矩阵进行解密
        grad = grad.apply(i2f)
        hess = hess.apply(i2f)

        # 记录每种分裂方式的梯度和（用于返回给主动方）、分裂方式（用于临时保存），二者下标需要一一对应
        logger.info(f'{self.name.upper()}: Gradients received, start calculating on {len(instance_space)} samples. ')
        self.local_splits = []  # 用于存储本地的分裂点
        cur_splits_sum = []  # 存储每个分裂点的梯度和海森矩阵的和

        # 对数据集中的每个特征（不包括目标变量'y'）进行处理
        for feature in [col for col in self.dataset.columns if col != 'y']:
            feature_values = self.dataset.loc[instance_space, feature].sort_values(ascending=True)      # 取出该特征该样本空间的值,排序
            split_indices = [int(qt * len(feature_values)) for qt in Calculator.quantile]               # 可选的分裂点

            # 对每个候选分裂点，计算左侧样本的梯度和海森矩阵的和
            for si in split_indices:
                left_space = feature_values.iloc[si:].index.tolist()
                thresh = feature_values.iloc[si]  # 分裂阈值
                left_grad_sum, left_hess_sum = grad[left_space].sum(), hess[left_space].sum()

                self.local_splits.append((feature, thresh, left_space))
                
                # 保存分裂点的梯度和海森矩阵的和（加密后的结果）
                cur_splits_sum.append({'grad_left': convert_for_json(left_grad_sum), 
                                       'hess_left': convert_for_json(left_hess_sum),
                                       'nums':len(left_space)})
       
        # 将计算得到的分裂信息保存到文件中
        splits_file = os.path.join(temp_root['file'][self.id], f'splits_file.json')
        with open(splits_file, 'w+') as f:
            json.dump(cur_splits_sum, f)

        self.temp_file = splits_file
        self.train_status = 'Ready'
        logger.info(f'{self.name.upper()}: Split scores finished, ready to return. ')

    def get_splits_sum(self):
        """
        主动方轮询请求被动方的训练结果
        """
        if self.train_status != 'Ready':
            return msg_empty()
        else:
            self.train_status = 'Idle'
            return msg_name_file(self.name, self.temp_file)
        
    def confirm_split(self, recv_dict: dict):
        """
        对主动方二次请求的确认，将最佳分裂点计入查找表中
        """
        # assert recv_dict['party_name'] == self.name, logger.error(f'Incorrect party name: \'{recv_dict["party_name"]}\'')
        logger.debug(f'Received data: {recv_dict}. ')
        best_split = self.local_splits[int(recv_dict['split_index'])]

        # 新条目添加进查找表
        look_up_index = self.look_up_table.shape[0]         # 查找表的最后一行作为新条目的下标
        self.look_up_table.loc[look_up_index] = {
            'feature_name': best_split[0], 
            'feature_split': best_split[1]
        }

        # 返回给主动方
        left_space_file = os.path.join(temp_root['file'][self.id], f'left_space.json')
        with open(left_space_file, 'w+') as f:
            json.dump(best_split[2], f)

        logger.info(f'{self.name.upper()}: Confirmation received, update look up table on index: {look_up_index}. ')
        
        return msg_split_confirm(self.name, int(look_up_index), left_space_file)

    def predict(self, recv_dict: dict):
        """
        根据主动方传来的样本空间、查找表条目将样本空间分割成左右部分返回
        """
        party_name = recv_dict['party_name']
        instance_space_file = recv_dict['instance_space']
        look_up_id = int(recv_dict['look_up_id'])

        assert party_name == self.name, logger.error(f'Incorrect party name: \'{party_name}\'. ')

        with open(instance_space_file, 'r') as f:
            instance_space = json.load(f)

        logger.info(f'{self.name.upper()}: Predicting on {len(instance_space)} samples. ')
        look_up_entry = self.look_up_table.iloc[look_up_id]
        feature_name, feature_thresh = look_up_entry['feature_name'], look_up_entry['feature_thresh']

        left_space = self.testset.loc[instance_space, feature_name] >= feature_thresh
        left_space = left_space[left_space].index.tolist()
        right_space = [idx for idx in instance_space if idx not in left_space]

        split_space_file = os.path.join(temp_root['file'][self.id], f'split_space.json')
        with open(split_space_file, 'w+') as f:
            json.dump({'left_space': left_space, 'right_space': right_space}, f)
        
        return msg_name_file(self.name, split_space_file)
    