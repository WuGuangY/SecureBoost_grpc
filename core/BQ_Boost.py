import numpy as np
from utils.params import pub_key_list
from utils.log import logger
from phe import PaillierPublicKey
from utils.encryption import serialize_encrypted_number
import pandas as pd
def encrypt_gradients(grad, pub_key: PaillierPublicKey):
        """
        将梯度用公钥加密
        """
        from tqdm import tqdm

        with tqdm(total=len(grad)*2) as pbar:

            def encrypt_data(data, pub_key: PaillierPublicKey):
                """
                将 data 加密后转换成字典形式返回
                """
                pbar.update(1)
                enc_data = pub_key.encrypt(data)
                return serialize_encrypted_number(enc_data)
            
            logger.info(f'Gradients encrypting... ')

            grad_enc = grad.apply(encrypt_data, pub_key=pub_key)
            return grad_enc
        
def bucket_quantization(g: pd.Series, s: int,pub_key) -> pd.Series:
    """
    对梯度或二阶导数值进行分桶量化。
    
    参数:
        g (pd.Series): 梯度或二阶导数数据。
        s (int): 分桶数。
        
    返回:
        pd.Series: 量化加密后的数据。
    """
    # 确定每个桶的边界
    min_g = g.min()
    max_g = g.max()
    bucket_boundaries = np.linspace(min_g, max_g, s + 1)

    bucket_boundaries_pd = pd.Series(bucket_boundaries)
    
    # 加密
    enc_bucket_boundaries = encrypt_gradients(bucket_boundaries_pd, pub_key)


    bucket_enc_dict_grad = dict(zip(bucket_boundaries, enc_bucket_boundaries))
    
    def quantize_value(value):
        for j in range(s):
            if bucket_boundaries[j] <= value < bucket_boundaries[j + 1]:
                return bucket_enc_dict_grad[bucket_boundaries[j]]
        return bucket_enc_dict_grad[bucket_boundaries[-1]]  # 如果值正好等于最大值，则返回最后一个桶的下界
    
    # 使用 apply 方法进行量化
    quantized_g = g.apply(quantize_value)
    
    return quantized_g


def BQ_Boost(data, bin_num, pub_key):
    """
    对模型梯度和二阶导数数据进行BQ-Boost方案优化。
    
    参数:
        data (tuple): 包含两个元素的元组，分别是梯度数据(grad)和二阶导数数据(hess)。
        bin_num (int): 梯度和二阶导数分桶数。
        pub_key (PaillierPublicKey): 公钥对象。
        
    返回:
        tuple: 量化的梯度和二阶导数数据。
    """
    grad, hess = data

    # 确保输入是 Pandas Series 类型
    if not isinstance(grad, pd.Series):
        grad = pd.Series(grad)
    if not isinstance(hess, pd.Series):
        hess = pd.Series(hess)
    
    # 对梯度数据进行分桶量化
    quantized_grad = bucket_quantization(grad, bin_num,pub_key=pub_key)
    
    # 对二阶导数数据进行分桶量化
    quantized_hess = bucket_quantization(hess, bin_num,pub_key=pub_key)

    return (quantized_grad, quantized_hess)


