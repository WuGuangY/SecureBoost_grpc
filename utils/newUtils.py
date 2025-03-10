import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from protos import Server_pb2
from utils.log import logger
from core.Calculator import Calculator
from msgs.messages import msg_empty, msg_name_file, msg_gradient_file, msg_split_confirm, msg_name_space_lookup
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier
from collections import deque
import os
from utils.params import temp_root,pp_list,stub_list
import json


# ap.load_dataset('./static/data/train.csv')
def load_dataset(data_path: str, valid_path: str = None):
    """
    加载数据集
    返回原始数据、特征索引和样本索引
    """
    raw_data = pd.read_csv(data_path)
    # raw_data.loc[:, 'Gender'] = raw_data['Gender'].map({'Male': 1, 'Female': 0}).astype(np.int8)
    # raw_data.loc[:, 'Vehicle_Damage'] = raw_data['Vehicle_Damage'].map({'Yes': 1, 'No': 0}).astype(np.int8)
    # raw_data.loc[:, 'Vehicle_Age'] = raw_data['Vehicle_Age'].map({'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2}).astype(np.int8)
    # raw_data.loc[:, 'Region_Code'] = raw_data['Region_Code'].astype(np.int16)
    # raw_data.loc[:, 'Policy_Sales_Channel'] = raw_data['Policy_Sales_Channel'].astype(np.int16)
    raw_data.loc[:, 'Gender'] = raw_data['Gender'].map({'Male': 1, 'Female': 0})
    raw_data.loc[:, 'Vehicle_Damage'] = raw_data['Vehicle_Damage'].map({'Yes': 1, 'No': 0})
    raw_data.loc[:, 'Vehicle_Age'] = raw_data['Vehicle_Age'].map({'< 1 Year': 0, '1-2 Year': 1, '> 2 Years': 2})
    raw_data.loc[:, 'Region_Code'] = raw_data['Region_Code']
    raw_data.loc[:, 'Policy_Sales_Channel'] = raw_data['Policy_Sales_Channel']
    # raw_data.loc[:, 'y'] = raw_data['y'].astype(np.int8)
    # 定义需要排除的列
    exclude_columns = ['id']  # 默认排除 'id' 列
    if 'Response' in raw_data.columns:  # 如果存在 'Response' 列，则也排除
        exclude_columns.append('Response')
    # 提取特征索引（排除 'id' 列和 'Response' 列（如果存在））
    feature_indices = [i for i, col in enumerate(raw_data.columns) if col not in exclude_columns]
    # 提取样本索引
    sample_indices = list(range(raw_data.shape[0]))
    raw_data = raw_data.to_numpy()
    return raw_data, feature_indices, sample_indices


def divide_data(raw_data: np.ndarray, feature_names: list, sample_indices: list):
    """
    将原始数据按照特征和样本索引集划分到host端和guest端。
    """
    feature_split_ratio = 0.2
    num_features = len(feature_names)
    split_point = int(num_features * feature_split_ratio)

    # 划分 host_data 和 guest_data
    host_data = raw_data[:, :split_point]  # host 端特征
    guest_data = raw_data[:, split_point:]  # guest 端特征（不包括 y 列）
    # guest_data = np.hstack((host_data[:, 0].reshape(-1, 1), guest_data))  # 添加第一列
    # host_data = host_data[:, 1:]  # 选择从第二列开始的所有列
    host_sample_indices = sample_indices
    guest_sample_indices = sample_indices

    client_data = {
        'host': {
            'data': host_data,
            'feature_indices': list(range(split_point)),
            'sample_indices': host_sample_indices
        },
        'guest': {
            'data': guest_data,
            'feature_indices': list(range(split_point, num_features)),
            'sample_indices': guest_sample_indices
        }
    }
    return client_data
# def divide_data(raw_data: np.ndarray, feature_names: list, sample_indices: list):
#     """
#     将原始数据按照特征和样本索引集划分到host端和guest端。
#     """
#     feature_split_ratio = 0.5
#     num_features = len(feature_names)
#     split_point = int(num_features * feature_split_ratio)
#
#     # 划分 host_data 和 guest_data
#     host_data = raw_data[:, :split_point]  # host 端特征
#     guest_data = raw_data[:, split_point:]  # guest 端特征（包括 y 列）
#
#     # 确保 guest_data 的最后一列是目标变量 y
#     y_column = guest_data[:, -1].reshape(-1, 1)
#     guest_data = guest_data[:, :-1]  # 去掉最后一列 y
#
#     # 随机挑选 2 列从 host_data 赋给 guest_data
#     host_indices = list(range(split_point))
#     selected_host_indices = np.random.choice(host_indices, 3, replace=False)
#     host_data_selected = host_data[:, selected_host_indices]
#     guest_data = np.hstack((guest_data, host_data_selected))
#
#     # 随机挑选 2 列从 guest_data 赋给 host_data
#     guest_indices = list(range(split_point, num_features - 1))  # 不包括最后一列 y
#     selected_guest_indices = np.random.choice(guest_indices, 3, replace=False)
#     guest_data_selected = guest_data[:, [i - split_point for i in selected_guest_indices]]  # 调整索引
#     host_data = np.hstack((host_data, guest_data_selected))
#
#     # 将目标变量 y 重新添加到 guest_data 的最后一列
#     guest_data = np.hstack((guest_data, y_column))
#
#     # 更新 feature_indices
#     host_feature_indices = list(range(split_point)) + [i for i in selected_guest_indices]
#     guest_feature_indices = list(range(split_point, num_features - 1)) + [i for i in selected_host_indices] + [num_features - 1]
#
#     host_sample_indices = sample_indices
#     guest_sample_indices = sample_indices
#
#     client_data = {
#         'host': {
#             'data': host_data,
#             'feature_indices': host_feature_indices,
#             'sample_indices': host_sample_indices
#         },
#         'guest': {
#             'data': guest_data,
#             'feature_indices': guest_feature_indices,
#             'sample_indices': guest_sample_indices
#         }
#     }
#     return client_data
# def divide_data(raw_data: np.ndarray, feature_ind: list, sample_ind: list):
#     """
#     将原始数据按照特征和样本索引集划分到不同的客户端。
#     参数：
#     raw_data: 原始数据集，ndarray类型。
#     feature_ind: 原始数据特征索引集合，List类型。
#     sample_ind: 原始数据样本索引集合，List类型。
#     返回值：
#     client_data: 划分得到持有不同数据集的客户端索引集，List类型。
#     """
#     client_data = raw_data[np.array(sample_ind), :][:, np.array(feature_ind)]
#
#     return client_data


# todo: 特征选择（按策略）
def vertical_federated_feature_selection(client_data: np.ndarray, feature_selection_criterion: str):
    """
    根据特征选择准则对客户端数据进行特征选择。
    参数：
    client_data: 客户端上的数据，ndarray类型。
    feature_selection_criterion: 特征选择准则，String类型。
    返回值：
    selected_features: 选择后的数据特征集合，List类型。
    """
    if not isinstance(client_data, np.ndarray):
        raise ValueError("client_data must be a numpy array.")
    if not isinstance(feature_selection_criterion, str):
        raise ValueError("feature_selection_criterion must be a string.")
    if feature_selection_criterion not in ["variance", "correlation", "importance", "mutual_info"]:
        raise ValueError("Unsupported feature selection criterion.")

    selected_features = []

    # 判断是否包含目标变量 y
    has_target = client_data.shape[1] > 1 and np.any(client_data[:, -1] != client_data[:, -1][0])

    if has_target:
        print("this is y!")
        X = client_data[:, :-1]  # 特征
        y = client_data[:, -1]  # 目标
        y = y.astype(int)  # 转换为整数类型
        if feature_selection_criterion == "variance":
            selector = VarianceThreshold(threshold=0.1)
            selector.fit(client_data)
            selected = np.where(selector.variances_ > 0.1)[0]
        elif feature_selection_criterion == "correlation":
            # target = client_data[:, -1]
            # correlations = np.corrcoef(client_data[:, :-1].T, target)
            # selected = np.where(np.abs(correlations[-1, :-1]) > 0.5)[0]
            correlations = np.corrcoef(X.T, y)  # 计算相关性
            selected = np.where(np.abs(correlations[-1, :-1]) > 0.5)[0]
        elif feature_selection_criterion == "importance":
            # X = client_data[:, :-1]  # 特征
            # y = client_data[:, -1]  # 目标
            # y = y.astype(int)  # 转换为整数类型

            model = RandomForestClassifier()
            model.fit(X, y)
            importances = model.feature_importances_
            selected = np.where(importances > 0.1)[0]
        elif feature_selection_criterion == "mutual_info":
            # X = client_data[:, :-1]  # 特征
            # y = client_data[:, -1]  # 目标
            # y = y.astype(int)  # 转换为整数类型

            mi = mutual_info_classif(X, y)
            selected = np.where(mi > 0.05)[0]  # 选择互信息大于 0.5 的特征
    else:
        if feature_selection_criterion == "variance":
            selector = VarianceThreshold(threshold=0.1)
            selector.fit(client_data)
            selected = np.where(selector.variances_ > 0.1)[0]
        else:
            raise ValueError("For clients without target variable, only 'variance' criterion is supported.")

    selected_features.append(selected.tolist())
    return selected_features


def filter_data_by_features(data: np.ndarray, feature_indices: list) -> np.ndarray:
    """
    根据特征索引筛选数据集。

    参数：
    data: 原始数据集，ndarray类型。
    feature_indices: 要保留的特征索引列表，List类型。

    返回值：
    filtered_data: 筛选后的数据集，ndarray类型。
    """
    return data[:, feature_indices]  # 根据特征索引筛选数据
# def filter_data_by_features(data: np.ndarray, feature_indices: list) -> np.ndarray:
#     """
#     根据特征索引筛选数据集，并去除 id 列和 y 列（如果存在）。
#
#     参数：
#     data: 原始数据集，ndarray类型。
#     feature_indices: 要保留的特征索引列表，List类型。
#
#     返回值：
#     filtered_data: 筛选后的数据集，ndarray类型。
#     """
#     # 假设 id 列是第 0 列，y 列是最后一列
#     # 去除 id 列和 y 列的索引
#     filtered_indices = [i for i in feature_indices if i != 0 and i != data.shape[1] - 1]
#
#     return data[:, filtered_indices]  # 根据特征索引筛选数据

def evaluate_model_performance(model, client_data, selected_feature_inds, passive_port):
    """
    评估模型基于所选特征子集训练的模型性能。

    参数：
    model (object): 联邦学习模型。
    client_data (ndarray): 测试数据集。
    selected_feature_inds (List): 存储每个客户端所选特征索引集。

    返回值：
    accuracy (float): 模型预测准确率。
    """
    if client_data is None:
        return

    logger.info(f'Start predicting. ')

    #获取stub
    stub=stub_list[0]

    split_nodes = deque()
    # preds = pd.Series(Calculator.init_pred, index=client_data.index)

    # selected_data = client_data[:, selected_feature_inds]
    selected_data = client_data
    # preds = pd.Series(Calculator.init_pred, index=selected_data[:, -1].index)  # 最后一列为真实标签
    preds = pd.Series(Calculator.init_pred, index=selected_data.index)
    name = 'party0'
    id = 0
    for tree in model:
        tree.instance_space = selected_data.index.tolist()
        split_nodes.append(tree)  # 树根入队
        while len(split_nodes):
            spliting_node = split_nodes.popleft()
            instance_space = spliting_node.instance_space
            if not spliting_node.left:  # 为叶子节点
                preds[instance_space] += spliting_node.weight
                continue
            elif spliting_node.party_name == name:  # 为主动方的分裂节点
                # 分裂样本空间
                logger.info(f'{name.upper()}: Splitting on node {spliting_node.id} from {spliting_node.party_name}. ')
                feature_name, feature_thresh = spliting_node.feature_split['feature_name'], spliting_node.feature_split[
                    'feature_thresh']
                left_space = selected_data.loc[instance_space, feature_name] >= feature_thresh
                left_space = left_space[left_space].index.tolist()
                right_space = [idx for idx in instance_space if idx not in left_space]

                spliting_node.left.instance_space = left_space
                spliting_node.right.instance_space = right_space

                split_nodes.extend([spliting_node.left, spliting_node.right])
            else:  # 为被动方的分裂节点
                logger.info(f'{name.upper()}: Splitting on node {spliting_node.id} from {spliting_node.party_name}. ')
                party_name = spliting_node.party_name
                look_up_id = spliting_node.look_up_id

                recv_data={
                    "party_name": party_name,
                    "look_up_id": look_up_id,
                    "instance_space":instance_space
                }
                #向被动方发送查询请求
                stub.ASendMessage(Server_pb2.MessageRequest(json_data=json.dumps(recv_data)))

                #接受被动方的分裂结果
                response = stub.AWaitForMessage(Server_pb2.Empty())
                json_data=json.loads(response.json_data)['data']

                #更新结果
                spliting_node.left.instance_space = json_data['left_space']
                spliting_node.right.instance_space = json_data['right_space']
                split_nodes.extend([spliting_node.left, spliting_node.right])

                
    logger.info(f'{name.upper()}: Test accuracy: {Calculator.accuracy(selected_data["y"], preds)}. ')
    logger.info(f'{name.upper()}: All finished. ')
    accuracy = Calculator.accuracy(selected_data["y"], preds)
    return accuracy


# todo: 计算冗余度
def evaluate_redundancy(model, feature_ind):
    """
    评估所选特征集合的冗余度。
    参数：
    model (Model): 联邦学习模型。
    feature_ind (list): 所选特征的索引集合。
    返回值：
    redundancy (float): 冗余度值。
    """
    logger.info(f'Start evaluating redundancy of selected features. ')

    # 初始化特征贡献次数字典
    feature_contributions = {ind: 0 for ind in feature_ind}

    # 遍历模型中的所有树，统计每个特征的贡献次数
    for tree in model.trees:
        stack = [tree]
        while stack:
            node = stack.pop()
            if node.feature_split:
                # 检查分裂特征是否在所选特征索引中
                if node.feature_split['feature_name'] in feature_ind:
                    feature_contributions[node.feature_split['feature_name']] += 1
            if node.left:
                stack.append(node.left)
            if node.right:
                stack.append(node.right)

    # 将特征贡献次数转换为列表
    contributions_list = [feature_contributions[ind] for ind in feature_ind]

    # 如果只有一个特征，无法计算相关性矩阵，直接返回0
    if len(contributions_list) == 1:
        redundancy = 0.0
    else:
        # 计算特征贡献次数之间的相关性矩阵
        correlation_matrix = np.corrcoef(contributions_list)

        # 计算冗余度
        redundancy = 1 - np.mean(np.abs(correlation_matrix - np.eye(len(contributions_list))))  # 排除对角线上的1

    logger.info(f'Redundancy of selected features: {redundancy}. ')
    return redundancy

if __name__ == '__main__':
    ####################################################################################示例1：一整个数据集进行划分
    raw_data, feature_names, sample_indices = load_dataset("D:/Projects/Python_projects/FinancialRiskControl/SellPredictionData/train.csv")
    # 输出 raw_data、feature_names、sample_indices 的尺寸和类型
    print(f"raw_data: 类型={type(raw_data)}, 尺寸={raw_data.shape}")
    print("第一行数据:", raw_data[0])
    print("第二行数据:", raw_data[1])
    print("第二行数据:", raw_data[2])
    print("第二行数据:", raw_data[3])
    print("第二行数据:", raw_data[4])
    print(f"feature_names: 类型={type(feature_names)}, 尺寸={len(feature_names)}")
    print(feature_names)
    print(f"sample_indices: 类型={type(sample_indices)}, 尺寸={len(sample_indices)}")

    # 存储最后一列
    last_column = raw_data[:, -1]  # 提取最后一列
    raw_data = raw_data[:, 1:-1]  # 选择从第二列到倒数第二列的所有列
    
    selected_features = vertical_federated_feature_selection(raw_data, "variance")
    print(f"***********************selected_raw_data_features: 类型={type(selected_features)}")
    print(selected_features)
    raw_data = filter_data_by_features(raw_data, selected_features[0])
    print(f"raw_data_new: 类型={type(raw_data)}, 尺寸={raw_data.shape}")
    print("第一行数据:", raw_data[0])
    print("第二行数据:", raw_data[1])
    print("第二行数据:", raw_data[2])
    print("第二行数据:", raw_data[3])
    print("第二行数据:", raw_data[4])

    client_data = divide_data(raw_data, feature_names, sample_indices)
    guest_data = client_data['guest']['data']
    host_data = client_data['host']['data']
    # 将最后一列拼接到guest_data
    guest_data = np.hstack((raw_data, last_column.reshape(-1, 1)))  # 将最后一列添加到guest_data
    # ####################################################################################示例2：按原代码，host与guest各自进行筛选
    # raw_data_Active, feature_names, sample_indices = load_dataset("D:/Projects/Python_projects/SecureBoostImpl-main/static/data/train.csv")
    # selected_features_Active = vertical_federated_feature_selection(raw_data_Active, "variance")

    # raw_data_Passive, feature_names, sample_indices = load_dataset("D:/Projects/Python_projects/SecureBoostImpl-main/static/data/train.csv")
    # selected_features_Active = vertical_federated_feature_selection(raw_data_Passive, "variance")

    # client_data = divide_data(raw_data_Active, feature_names, sample_indices)  # 
    # guest_data = client_data
    # guest_data_filtered = filter_data_by_features(guest_data, selected_features[0])

    # client_data = divide_data(raw_data_Passive, feature_names, sample_indices)  # 
    # host_data = client_data
    # host_data_filtered = filter_data_by_features(host_data, selected_features[0])
    # print(f"selected_features: 类型={type(host_data_filtered)}")
    # print(f"host_data_filtered: 类型={type(host_data_filtered)}, 尺寸={host_data_filtered.shape}")

    # print(f"guest_data: 类型={type(guest_data)}, 尺寸={guest_data.shape}")
    # print("第一行数据:", guest_data[0])
    # print("第二行数据:", guest_data[1])
    # print("第二行数据:", guest_data[2])
    # print("第二行数据:", guest_data[3])
    # print("第二行数据:", guest_data[4])
    # print(f"host_data: 类型={type(host_data)}, 尺寸={host_data.shape}")
    # print("第一行数据:", host_data[0])
    # print("第二行数据:", host_data[1])
    # print("第二行数据:", host_data[2])
    # print("第二行数据:", host_data[3])
    # print("第二行数据:", host_data[4])

    # print(f"host_data: 类型={type(host_data)}, 尺寸={host_data.shape}")