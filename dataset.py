import numpy as np
import pandas as pd
import torch
from tqdm import tqdm # 用于构建索引时的进度条
import os
from torch.utils.data import  Dataset


# 定义字符集 (保持不变)
CHARPROTSET = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
    "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
    "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
    "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24,
    "Z": 25
}

CHARISOSMISET = {
    "#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
    "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
    "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
    "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
    "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
    "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
    "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
    "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64, ":": 65
}

def label_chars(chars, char_set, length):
    X = np.zeros(length, dtype=np.int32)
    for i, ch in enumerate(chars):
        X[i] = char_set.get(ch, 0)
    return X

class DTADataLazy(Dataset):
    def __init__(self, aff, dataset_name='davis'):
        self.aff = np.array(aff)
        self.dataset_name = dataset_name
        self.len = len(self.aff)
        
        # 1. 加载序列字典 (序列文本占用的内存相对较小，可以保留在内存中加速访问)
        # 如果序列也非常多导致内存不足，也可以改为惰性加载，但通常 CSV 解析后的字典几百MB以内
        seq_path = f"./data/{dataset_name}/sequence.csv"
        smi_path = f"./data/{dataset_name}/smiles.csv"
        
        print(f"Loading sequence maps for {dataset_name}...")
        self.seq_dict = pd.read_csv(seq_path).set_index('TARGETID')['SEQUENCE'].to_dict()
        self.smi_dict = pd.read_csv(smi_path).set_index('DRUGID')['SMILES'].to_dict()
        
        # 2. 构建文件路径索引 (而不是加载数据本身)
        # 格式: { 'drug_id': 'path/to/file.npz', ... }
        self.fp_path_map = {}
        self.desc_path_map = {}
        
        desc_path = f"./data/{dataset_name}/protein_descriptor/"
        fp_path = f"./data/{dataset_name}/drug_fingerprint/"
        
        print(f"Indexing fingerprint files in {fp_path}...")
        if os.path.exists(fp_path):
            for file in tqdm(os.listdir(fp_path)):
                if file.endswith(".npz"):
                    # 解析 ID，逻辑需与之前保持一致
                    if len(file.split('.')) > 2:
                        item_id = file.split('.')[0]+'.'+file.split(".")[1]
                    else:
                        item_id = file.split(".")[0]
                    # 存储完整路径
                    self.fp_path_map[item_id] = os.path.join(fp_path, file)
                    
        print(f"Indexing descriptor files in {desc_path}...")
        if os.path.exists(desc_path):
            for file in tqdm(os.listdir(desc_path)):
                if file.endswith(".npz"):
                    if len(file.split('.')) > 2:
                        item_id = file.split('.')[0]+'.'+file.split(".")[1]
                    else:
                        item_id = file.split(".")[0]
                    self.desc_path_map[item_id] = os.path.join(desc_path, file)

        # 预定义期望的键，用于加载时提取
        self.desc_keys = ['aac', 'dac', 'tpc', 'ctd']
        # self.fp_keys = ['ecfp',  'maccs', 'rdkit', 'vsa']
        self.fp_keys = ['ecfp',  'maccs', 'erg', 'vsa', 'pubchem'] # davis 多了 pubchem
        # 缓存最近访问的几个文件？(可选优化，如果磁盘IO成为瓶颈)
        # 对于大多数现代 SSD，直接读取即可。如果需要，可以使用 functools.lru_cache 装饰器包装加载函数

    def _load_fp(self, drug_id):
        """惰性加载药物指纹"""
        str_id = str(drug_id)
        path = self.fp_path_map.get(str_id)
        data = np.load(path, allow_pickle=False)
        arrays = [data[k] for k in self.fp_keys if k in data.files]
        if arrays:
                return np.concatenate(arrays)

    def _load_desc(self, target_id):
        """惰性加载蛋白描述符"""
        path = self.desc_path_map.get(target_id)
        data = np.load(path, allow_pickle=False)
        arrays = [data[k] for k in self.desc_keys if k in data.files]
        if arrays:
            return np.concatenate(arrays)


    def __getitem__(self, index):
        drug_id = self.aff[index][0]
        target_id = self.aff[index][1]
        
        # 1. 处理 SMILES 和 Sequence (从内存字典获取，速度快)
        smiles_str = self.smi_dict.get(drug_id, "")[:1024]
        sequence_str = self.seq_dict.get(target_id, "")[:1024]
        
        smiles = torch.LongTensor(label_chars(smiles_str, CHARISOSMISET, 1024))
        sequence = torch.LongTensor(label_chars(sequence_str, CHARPROTSET, 1024))
        
        # 2. 惰性加载描述符和指纹 (从磁盘读取)
        desc_arr = self._load_desc(target_id)
        fp_arr = self._load_fp(drug_id)
        
        desc_tensor = torch.from_numpy(desc_arr).float()
        fp_tensor = torch.from_numpy(fp_arr).float()
        
        # 3. Label
        value = torch.FloatTensor([float(self.aff[index][2])])
        
        return smiles, sequence, desc_tensor, fp_tensor, value
    
    def __len__(self):
        return self.len