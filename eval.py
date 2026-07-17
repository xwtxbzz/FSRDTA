from metrics import *
from dataset import *
from model import *
from sklearn.model_selection import train_test_split
from torch.utils.data import  DataLoader
def evaluate_model(model_path, dataset='metz', start='', batch_size=256, num_workers=8):
    """
    评估模型性能
    
    Args:
        model_path: 模型权重路径
        dataset: 数据集名称
        start: 数据分割方式 ('cold', 'cold_drug', 'cold_target', '')
        batch_size: 批次大小
        num_workers: 数据加载线程数
    """
    
    print("=" * 60)
    print(f"Evaluating Model: {model_path}")
    print(f"Dataset: {dataset}, Split: {start if start else 'random'}")
    print("=" * 60)
    
    # 1. 准备设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. 加载数据
    data_path = f"./data/{dataset}/affinity.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Cannot find {data_path}")
    
    data = pd.read_csv(data_path)
    

    print("Using random split")
    data = data.to_numpy()[:]
    train_data, test_data = train_test_split(data, test_size=0.16667, random_state=42)
    
    print(f"Train size: {train_data.shape}, Test size: {test_data.shape}")
    
    # 3. 创建数据集和数据加载器
    test_dataset = DTADataLazy(test_data, dataset_name=dataset)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True
    )
    
    # 4. 创建模型
    # 从数据中确定n_drug和n_target
    all_drugs = data['DRUGID'].unique() if isinstance(data, pd.DataFrame) else np.unique(data[:, 0])
    all_targets = data['TARGETID'].unique() if isinstance(data, pd.DataFrame) else np.unique(data[:, 1])
    n_drug = len(all_drugs)
    n_target = len(all_targets)
    print(f"Number of drugs: {n_drug}, Number of targets: {n_target}")
    
    model = FSR_DTA(
        n_drug=66,
        n_target=26,
        hidden_dim=128,
        dropout=0.
    )
    
    # 5. 加载模型权重
    if os.path.isfile(model_path):
        print(f"Loading checkpoint from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
            epoch = checkpoint.get("epoch", 0)
            best_mse = checkpoint.get("best_mse", float("inf"))
            print(f"Loaded from epoch {epoch+1}, Best MSE: {best_mse:.6f}")
        else:
            # 兼容旧格式
            model.load_state_dict(checkpoint)
            print("Loaded old-format state_dict.")
    else:
        print(f"Warning: Model file {model_path} not found!")
        return None
    
    model = model.to(device)
    model.eval()
    
    # 6. 评估
    print("\nStarting evaluation...")
    y_true = []
    y_pred = []
    val_loss = 0.
    
    def data_process(data):
        d_list = [d.cuda() for d in data[:-1]]
        value = data[-1].cuda()
        return tuple(d_list), value
    
    with torch.no_grad():
        test_bar = tqdm(test_loader, desc="Evaluating")
        
        for te_data in test_bar:
            data, value = data_process(te_data)
            
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = nn.functional.mse_loss(output, value)
            
            val_loss += loss.item()
            
            y_true.extend(value.cpu().numpy().flatten())
            y_pred.extend(output.cpu().numpy().flatten())
    
    val_loss /= len(test_loader)
    
    # 7. 计算详细指标
    print("\n" + "=" * 60)
    print("Evaluation Results:")
    print("=" * 60)
    
    # 转换为numpy数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # 计算所有指标
    mse = get_mse(y_true, y_pred)
    rmse = get_rmse(y_true, y_pred)
    pearson = get_pearson(y_true, y_pred)
    spearman = get_spearman(y_true, y_pred)
    ci = get_ci(y_true, y_pred)
    rm2 = get_rm2(y_true, y_pred)
    
    print(f"\nMSE:      {mse:.6f}")
    print(f"RMSE:     {rmse:.6f}")
    print(f"Pearson:  {pearson:.6f}")
    print(f"Spearman: {spearman:.6f}")
    print(f"CI:       {ci:.6f}")
    print(f"RM2:      {rm2:.6f}")
    print(f"Avg Loss: {val_loss:.6f}")
    
    # 8. 保存结果到文件
    result_file = f'./results/eval_{dataset}_{start}_{model_name}.txt'
    os.makedirs('./results', exist_ok=True)
    
    with open(result_file, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Dataset: {dataset}, Split: {start if start else 'random'}\n")
        f.write(f"Test samples: {len(y_true)}\n")
        f.write("=" * 60 + "\n")
        f.write(f"MSE:      {mse:.6f}\n")
        f.write(f"RMSE:     {rmse:.6f}\n")
        f.write(f"Pearson:  {pearson:.6f}\n")
        f.write(f"Spearman: {spearman:.6f}\n")
        f.write(f"CI:       {ci:.6f}\n")
        f.write(f"RM2:      {rm2:.6f}\n")
        f.write(f"Avg Loss: {val_loss:.6f}\n")
        f.write("=" * 60 + "\n")
    
    print(f"\nResults saved to: {result_file}")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'pearson': pearson,
        'spearman': spearman,
        'ci': ci,
        'rm2': rm2,
        'loss': val_loss
    }


# =========================
# 主程序入口
# =========================

if __name__ == "__main__":
    
    # 配置参数
    configs = [
        # {
        #     'dataset': 'davis',
        #     'start': '',
        #     'model_name': 'FSR_DTA'
        # },
        {
            'dataset': 'metz',
            'start': '',
            'model_name': 'FSR_DTA'
        },
        # {
        #     'dataset': 'davis',
        #     'start': 'cold',
        #     'model_name': 'FSR_DTA'
        # },
    ]
    
    for config in configs:
        dataset = config['dataset']
        start = config['start']
        model_name = config['model_name']
        
        # 构建模型路径
        if start:
            model_path = f"./model/{model_name}_{dataset}_{start}.pth"
        else:
            model_path = f"./model/{model_name}_{dataset}_.pth"
        
        print(f"\n{'=' * 80}")
        print(f"Evaluating: {dataset} (start={start if start else 'random'})")
        print(f"{'=' * 80}\n")
        
        try:
            results = evaluate_model(
                model_path=model_path,
                dataset=dataset,
                start=start,
                batch_size=256,
                num_workers=8
            )
            print("\n✓ Evaluation completed successfully!")
            
        except Exception as e:
            print(f"\n✗ Error during evaluation: {e}")
            import traceback
            traceback.print_exc()