from rdkit import Chem
from skfp.fingerprints import ECFPFingerprint, VSAFingerprint, MACCSFingerprint,PubChemFingerprint,ERGFingerprint
import numpy as np
import pandas as pd
import os

def extract_and_check_fingerprints(dataset='IC50'):
    current_path = os.path.join(os.getcwd(), f'data/{dataset}/smiles.csv')
    save_path = os.path.join(os.getcwd(), f'data/{dataset}/drug_fingerprint/')
    os.makedirs(save_path, exist_ok=True)

    df = pd.read_csv(current_path)
    
    # 统计信息
    total_count = len(df)
    valid_count = 0
    invalid_smiles_count = 0
    nan_inf_count = 0
    invalid = list()
    print(f"Start processing {total_count} molecules for {dataset}...")

    for index, row in df.iterrows():
        drug_id = row['DRUGID']
        smiles = row['SMILES']
        output_file = os.path.join(save_path, f"{drug_id}.npz")
        
        # 1. 生成分子对象
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            print(f"[SKIP] Invalid SMILES for DRUGID {drug_id}: {smiles}")
            invalid.append(drug_id)
            invalid_smiles_count += 1
            # 可选：如果之前存在该文件，建议删除，防止使用旧数据
            if os.path.exists(output_file):
                os.remove(output_file)
            continue
        else:
                # 2. 计算指纹特征
                ecfp_fp = ECFPFingerprint().transform([molecule]).squeeze()
                pubchem_fp = PubChemFingerprint().transform([molecule]).squeeze()
                maccs_fp = MACCSFingerprint().transform([molecule]).squeeze()
                erg_fp = ERGFingerprint().transform([molecule]).squeeze()
                vsa_fp =  VSAFingerprint().transform([molecule]).squeeze()
                # 3. 异常值检查 (NaN / Inf)
                all_fps = [ecfp_fp, pubchem_fp, maccs_fp, erg_fp, vsa_fp]
                for name, fp in zip(['ecfp', 'pubchem', 'maccs', 'erg','vas'], all_fps):
                    if np.isnan(fp).any() or np.isinf(fp).any():
                            print(name)
                            print(f"[WARNING] NaN/Inf detected in {name} for DRUGID {drug_id}")
                    

                # 5. 保存为 .npz 文件
                np.savez_compressed(
                        output_file,
                        ecfp=ecfp_fp,
                        pubchem=pubchem_fp,
                        maccs=maccs_fp,
                        erg=erg_fp,
                        vsa=vsa_fp,
                )
                valid_count += 1
        


    print(f"\n--- Summary for {dataset} ---")
    print(f"Total: {total_count}")
    print(f"Valid & Saved: {valid_count}")
    print(f"Invalid SMILES / Errors: {invalid_smiles_count}")
    print(f"Skipped due to NaN/Inf: {nan_inf_count}")
    print(f"--------------------------\n")
    for drug_id in invalid:
        print(f"[INVALID] DRUGID {drug_id}")
# # 执行检查 'KI',  'IC50', 'EC50' ,'KI','KD','IC50','EC50' , 'IC50', 'EC50', 'KI', 'KD'
# extract_and_check_fingerprints()
for ds in ['kiba','metz','davis']:
    if os.path.exists(f'data/{ds}/smiles.csv'):
        extract_and_check_fingerprints(ds)