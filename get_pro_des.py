from propy import PyPro
import numpy as np
import pandas as pd
import os

# 读取数据
dataset = 'Kd'
current_path = os.getcwd() + f'/data/{dataset}/sequence.csv'
save_path = os.getcwd() + f'/data/{dataset}/protein_descriptor/'
os.makedirs(save_path, exist_ok=True)

df = pd.read_csv(current_path)

# 遍历每一行，计算蛋白质描述符并保存为单独的 .npz 文件
for index, row in df.iterrows():
    target_id = row['TARGETID']
    sequence = row['SEQUENCE']
    if os.path.exists(os.path.join(save_path, f"{target_id}.npz")):
        continue
    # 计算描述符
    protein = PyPro.GetProDes(sequence)
    aac = np.array([v for _, v in protein.GetAAComp().items()])
    ctd = np.array([v for _, v in protein.GetCTD().items()])
    dac = np.array([v for _, v in protein.GetDPComp().items()])
    tpc = np.array([v for _, v in protein.GetTPComp().items()])
    
    # 保存为 .npz 文件
    np.savez_compressed(
        os.path.join(save_path, f"{target_id}.npz"),
        aac=aac,
        ctd=ctd,
        dac=dac,
        tpc=tpc
    )

print(f"Protein descriptors saved as .npz files in {save_path}")