from concurrent import futures
import json
import os
import time
import grpc
import pandas as pd
from protos import Server_pb2
from protos import Server_pb2_grpc

from phe import paillier, PaillierPublicKey, EncryptedNumber
from utils.sha1 import sha1
from utils.encryption import load_pub_key, serialize_encrypted_number, load_encrypted_number
from utils.log import logger

from core.BQ_Boost import BQ_Boost
from core.DI_Boost import DI_Boost,f2i,i2f
from core.Calculator import Calculator


# #序列化EncryptedNumber对象
# def custom_encoder(obj):
#     if isinstance(obj, EncryptedNumber):
#         # 手动提取 PaillierPublicKey 的 n 属性
#         public_key_info = {'n': str(obj.public_key.n)}  # 如果 n 是一个大整数，需要转换为字符串以确保可序列化
        
#         return {
#             '__type__': 'EncryptedNumber',
#             'public_key': public_key_info,
#             'ciphertext': str(obj._EncryptedNumber__ciphertext),  # 注意这里使用了私有变量的访问方式
#             'exponent': obj.exponent,
#             'is_obfuscated': obj._EncryptedNumber__is_obfuscated
#         }
#     raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')


class ServerServicer(Server_pb2_grpc.ServerServicer):
    def __init__(self):
        # 生成Paillier密钥对
        self.pub_key, self.pri_key = paillier.generate_paillier_keypair(n_length=2048)  # 设置适当的密钥长度
        self.json_data_toA = None
        self.json_data_toP = None

        #file_name,file_sender
        self.file_data_toA = []
        self.file_data_toP = []
        
    # #A请求P进行样本对齐
    # def ACallP_sample_align(self, request,context):
    #     logger.info("ACallP_sample_align")
    #     # print(f'Passive : {request.message}!')
    #     return Server_pb2.ServerResponse(message=f'Passive : {request.message}!')
    
    def ComputeHash(self, request, context):
        # 计算SHA-1哈希值
        train_idx = request.indices
        train_idx_map = {sha1(idx): idx for idx in train_idx}
        
        response = Server_pb2.HashResponse()
        for hash_key, original_index in train_idx_map.items():
            hash_pair = response.hash_pairs.add()
            hash_pair.hash = hash_key
            hash_pair.original_index = original_index
        
        response.hash_keys.extend(train_idx_map.keys())
        
        return response
    
    def GetPublicKey(self, request, context):
        """
        实现 GetPublicKey 方法，返回公钥。
        """
        logger.info("get publicKey")

        # 将公钥的 n 值序列化为字节流
        n_bytes = self.pub_key.n.to_bytes((self.pub_key.n.bit_length() + 7) // 8, byteorder='big')
        return Server_pb2.PublicKey(n=n_bytes) 
    
    def GetPrivateKey(self, request, context):
        """
        实现 GetPrivateKey 方法，返回私钥。
        """
        logger.info("GetPrivateKey")
        # 将私钥的 p 和 q 值序列化为字节流
        p_bytes = self.pri_key.p.to_bytes((self.pri_key.p.bit_length() + 7) // 8, byteorder='big')
        q_bytes = self.pri_key.q.to_bytes((self.pri_key.q.bit_length() + 7) // 8, byteorder='big')
        return Server_pb2.PrivateKey(p=p_bytes, q=q_bytes)
    
    def LoadEncryptedNumber(self, request, context):
        series_grad = json.loads(request.series_grad)
        series_hess = json.loads(request.series_hess)
        grad = pd.Series(series_grad)
        hess = pd.Series(series_hess)

        grad = grad.apply(load_encrypted_number, pub_key=self.pub_key)
        hess = hess.apply(load_encrypted_number, pub_key=self.pub_key)

        def encrypted_number_to_dict(encrypted_number):
            """
            将 EncryptedNumber 对象转换为字典。
            """
            return {
                'ciphertext': encrypted_number.ciphertext(),
                'exponent': encrypted_number.exponent
            }
        
        # 将加密后的 Series 转换为字典，并将每个 EncryptedNumber 转换为字典
        grad_dict = grad.apply(encrypted_number_to_dict).to_dict()
        hess_dict = hess.apply(encrypted_number_to_dict).to_dict()

        # 将字典序列化为 JSON 字符串
        q_grad_json = json.dumps(grad_dict)
        q_hess_json = json.dumps(hess_dict)

        return Server_pb2.SeriesResponse(series_grad=q_grad_json, series_hess=q_hess_json)
    
    def ASendMessage(self, request, context):
        """客户端 A 发送 JSON 数据"""
        while self.json_data_toP is not None:  # 如果有数据，持续等待
            time.sleep(0.5)  # 每0.5秒检查一次
        # 将 JSON 数据转化为字典
        logger.info(f"ASendMessage")
        try:
            json_obj = json.loads(request.json_data)
        except json.JSONDecodeError:
            return Server_pb2.MessageResponse(json_data="无效的 JSON 数据")
        
        # 存储解析后的 JSON 数据
        self.json_data_toP = json_obj

        return Server_pb2.MessageResponse()
    
    def PSendMessage(self, request, context):
        """客户端 P 发送 JSON 数据"""
        while self.json_data_toA is not None:  # 如果有数据，持续等待
            time.sleep(0.5)  # 每0.5秒检查一次
        # 将 JSON 数据转化为字典
        logger.info(f"PSendMessage")

        try:
            json_obj = json.loads(request.json_data)
        except json.JSONDecodeError:
            return Server_pb2.MessageResponse(json_data="无效的 JSON 数据")
        
        # 存储解析后的 JSON 数据
        self.json_data_toA = json_obj

        return Server_pb2.MessageResponse()
    
    def PSendFile(self, request, context):
        while len(self.file_data_toA)!=0:  # 如果有数据，持续等待
            time.sleep(0.5)  # 每0.5秒检查一次
        logger.info(f"P send file : {[x.name for x in request.files]}")
        for file_info in request.files:
            file_data = file_info.file  # 接收到的文件
            file_name = file_info.name  # 接收到的文件名称
            file_sender = file_info.party_name

            self.file_data_toA.append([file_name,file_sender])
            f = open(file_name, "wb")  # 把传来的文件写入本地磁盘
            f.write(file_data)  # 写文件
            f.close()  # 关闭IO

        return Server_pb2.FileReply(msg='ok')
    
    def ASendFile(self, request, context):
        while len(self.file_data_toP) != 0:  # 如果有数据，持续等待
            time.sleep(0.5)  # 每0.5秒检查一次
        logger.info(f"A send file : {[x.name for x in request.files]}")
        for file_info in request.files:
            file_data = file_info.file  # 接收到的文件
            file_name = file_info.name  # 接收到的文件名称
            file_sender = file_info.party_name

            self.file_data_toP.append([file_name,file_sender])
            f = open(file_name, "wb")  # 把传来的文件写入本地磁盘
            f.write(file_data)  # 写文件
            f.close()  # 关闭IO

        return Server_pb2.FileReply(msg='ok')
    
    def AGetFile(self, request, context):
        """客户端 A 等待 文件 数据"""
        while len(self.file_data_toA) == 0:  # 如果没有数据，持续等待
            time.sleep(1)  # 每0.5秒检查一次

        logger.info("AGetFile")
        #要传输的文件
        files=[]
        
        # 服务器响应客户端 的 JSON 数据
        while(len(self.file_data_toA)):
            file_name,file_sender= self.file_data_toA.pop(0)
            with open(file_name, 'rb') as f:
                data = f.read()
            files.append(Server_pb2.FileRequest.FileInfo(file=data,name=file_name,party_name=file_sender),)
        return Server_pb2.FileRequest(files=files)
    
    def PGetFile(self, request, context):
        """客户端 P 等待 文件 数据"""
        while len(self.file_data_toP) < 3:  # 如果没有数据，持续等待
            time.sleep(1)  # 每0.5秒检查一次
  
        #要传输的文件
        files=[]
        files_name=[]
        
        # 服务器响应客户端的 JSON 数据
        while(len(self.file_data_toP)):
            file_name,file_sender= self.file_data_toP.pop(0)
            files_name.append(file_name)
            with open(file_name, 'rb') as f:
                data = f.read()
            files.append(Server_pb2.FileRequest.FileInfo(file=data,name=file_name,party_name=file_sender),)
        logger.info(f"PGetFile : {files_name}")
        return Server_pb2.FileRequest(files=files)

    def AWaitForMessage(self, request, context):
        """客户端  等待 JSON 数据"""
        while not self.json_data_toA:  # 如果没有数据，持续等待
            time.sleep(1)  # 每0.5秒检查一次
        # 服务器响应客户端的 JSON 数据
        logger.info("A get message")

        response_json = json.dumps({"status": "success", "message": "数据已准备好", "data": self.json_data_toA})
        self.json_data_toA = None
        return Server_pb2.MessageResponse(json_data=response_json)
    
    def PWaitForMessage(self, request, context):
        """客户端  等待 JSON 数据"""
        while self.json_data_toP is None:  # 如果没有数据，持续等待
            time.sleep(1)  # 每0.5秒检查一次

        logger.info("P GET MESSAGE")
        # 服务器响应客户端 B 的 JSON 数据
        response_json = json.dumps({"status": "success", "message": "数据已准备好", "data": self.json_data_toP})
        self.json_data_toP = None
        return Server_pb2.MessageResponse(json_data=response_json)
    
    # #客户端获取解密后的数据
    # def GetDecryptedSplits(self, request, context):
    #     decrypted_splits_data = []
        
    #     for idx, split in enumerate(request.splits_data):
    #         left_grad_sum = load_encrypted_number(split.grad_left, self.pub_key)
    #         left_hess_sum = load_encrypted_number(split.hess_left, self.pub_key)
            
    #         left_grad_sum = self.pri_key.decrypt(left_grad_sum)
    #         left_hess_sum = self.pri_key.decrypt(left_hess_sum)
            
    #         decrypted_splits_data.append(
    #             Server_pb2.DecryptedSplitInfo(
    #                 idx=idx,
    #                 grad=left_grad_sum,
    #                 hess=left_hess_sum
    #             )
    #         )

    #     return Server_pb2.SplitsResponse(decrypted_splits_data=decrypted_splits_data)
    
    # #被动端获得解密后的数据（公钥）
    # def DecryptGradientHessian(self, request, context):

    #     logger.info("DecryptGradientHessian")

    #     # 将接收到的JSON字符串反序列化为Pandas Series
    #     series_grad = json.loads(request.series_grad)
    #     series_hess = json.loads(request.series_hess)
    #     grad = pd.Series(series_grad)
    #     hess = pd.Series(series_hess)

    #     # 对梯度和海森矩阵进行解密
    #     q_grad = grad.apply(load_encrypted_number, pub_key=self.pub_key)
    #     q_hess = hess.apply(load_encrypted_number, pub_key=self.pub_key)    

    #     # 将新的Pandas Series序列化为JSON字符串
    #     q_grad_json = json.dumps(q_grad.to_dict(), default=custom_encoder)
    #     q_hess_json = json.dumps(q_hess.to_dict(), default=custom_encoder)

    #     return Server_pb2.SeriesResponse(series_grad=q_grad_json, series_hess=q_hess_json)
        
    
    def BQBoost(self, request, context):
        try:
            logger.info("BQBoost")
            # 将接收到的JSON字符串反序列化为Pandas Series
            series_grad = json.loads(request.series_grad)
            series_hess = json.loads(request.series_hess)
            grad = pd.Series(series_grad)
            hess = pd.Series(series_hess)

            q_grad, q_hess = BQ_Boost((grad, hess),32, self.pub_key)
            
            # 将新的Pandas Series序列化为JSON字符串
            q_grad_json = json.dumps(q_grad.to_dict())
            q_hess_json = json.dumps(q_hess.to_dict())

            return Server_pb2.SeriesResponse(series_grad=q_grad_json, series_hess=q_hess_json)
        
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Error processing series: {str(e)}')
            return Server_pb2.SeriesResponse()
        
    def DIBoost(self, request, context):
        try:
            logger.info("DIBoost")
            # 将接收到的JSON字符串反序列化为Pandas Series
            series_grad = json.loads(request.series_grad)
            series_hess = json.loads(request.series_hess)
            grad = pd.Series(series_grad)
            hess = pd.Series(series_hess)

            q_grad, q_hess, key_list = DI_Boost((grad, hess), 1.5)
            

            # 将新的Pandas Series序列化为JSON字符串
            q_grad_json = json.dumps(q_grad.to_dict())
            q_hess_json = json.dumps(q_hess.to_dict())

            key_list_json = json.dumps(key_list)

            return Server_pb2.DISeriesResponse(series_grad=q_grad_json, series_hess=q_hess_json,key_list=key_list_json)
        
        except Exception as e:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f'Error processing series: {str(e)}')
            return Server_pb2.SeriesResponse()

def serve():
    # 设置最大消息大小为 100MB（100 * 1024 * 1024）
    options = [
        ('grpc.max_receive_message_length', 1000 * 1024 * 1024),  # 接收消息的最大大小
        ('grpc.max_send_message_length', 1000 * 1024 * 1024)     # 发送消息的最大大小
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=options)
    Server_pb2_grpc.add_ServerServicer_to_server(ServerServicer(), server)
    # 监听所有网络接口
    server.add_insecure_port('[::]:50052')
    server.start()
    print("Server started, listening on 50052")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()