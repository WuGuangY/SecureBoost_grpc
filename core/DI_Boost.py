import numpy as np
import pandas as pd
from utils.params import key_list

def generate_noise(sigma):
    """
    生成指定形状的高斯噪声。
    
    参数:
        sigma (float): 噪声的标准差。
        shape (tuple): 噪声数组的形状。
        
    返回:
        np.ndarray: 生成的噪声。
    """
    return np.random.normal(0, sigma)

def add_noise(data, noise):
    return data + noise

def multi_noise(data, noise):
    return data * noise

def f2i(x, k=5):
    return np.int8(np.clip(np.round(x * (2 ** k)), -128, 127))

def i2f(x, k=5):
    return np.float32(np.float32(x) / (2 ** k))

def DI_Boost(data, theta):
    """
    对模型梯度和二阶导数数据进行DI-Boost方案优化。
    
    参数:
        data (tuple): 包含两个元素的元组，分别是梯度数据(grad)和二阶导数数据(hess)。
        theta (float): 控制噪声大小的参数。
        
    返回:
        tuple: 处理后的梯度和二阶导数数据。
    """
    gi, hi = data
    
    # 确保输入是 Pandas Series 类型
    if not isinstance(gi, pd.Series):
        gi = pd.Series(gi)
    if not isinstance(hi, pd.Series):
        hi = pd.Series(hi)
    
    bias_k = max(np.abs(gi))
    gi_bias = gi + bias_k    
    l2_norm = np.linalg.norm(gi)
    gi_bias_divide = gi_bias / l2_norm
    
    noise_gi = generate_noise(theta)
    noise_hi = generate_noise(theta)
    
    noise_gi_bias_divide = gi_bias_divide * noise_gi  
    point_noise_hi = hi * noise_hi
    
    # 使用 apply 方法对每个元素应用 f2i 转换
    fixed_point_noise_hi = point_noise_hi.apply(f2i)
    noise_gi_bias_divide = noise_gi_bias_divide.apply(f2i)
    
    key_list[0] = [bias_k, l2_norm, noise_gi, noise_hi]
    
    return noise_gi_bias_divide, fixed_point_noise_hi, [bias_k, l2_norm, noise_gi, noise_hi]