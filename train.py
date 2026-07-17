from sklearn.model_selection import train_test_split
from torch.utils.data import  DataLoader
from tqdm.auto import tqdm
import numpy as np
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")
from metrics import *
from dataset import *
from model import *
# ================= 使用示例 =================

dataset = 'metz' # 或者 'davis'
start = ''
print(f"Preparing dataset: {dataset}")

# 1. 读取 affinity 数据
data_path = f"./data/{dataset}/affinity.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Cannot find {data_path}")

data = pd.read_csv(data_path)


print("full")
data = data.to_numpy()[:]
train_data, test_data = train_test_split(data, test_size=0.16667, random_state=42) # 42

print(f"Train size: {train_data.shape}, Test size: {test_data.shape}")

# 3. 创建惰性加载的数据集实例
# 注意：这里不会立即加载所有 npz 文件到内存，只会扫描目录建立索引
train_dataset = DTADataLazy(train_data, dataset_name=dataset)
test_dataset = DTADataLazy(test_data, dataset_name=dataset)

# 4. 创建 DataLoader
# num_workers > 0 非常重要！它允许并行地从磁盘加载数据，掩盖 IO 延迟
# 如果 num_workers=0，主进程串行读取磁盘，训练会非常慢

train_loader = DataLoader(
    train_dataset,
    batch_size=256,
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    drop_last=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=256,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True
)

print("DataLoaders created successfully. Memory usage should be low now.")
def data_process(data):
    d_list = [d.cuda() for d in data[:-1]]
    value = data[-1].cuda()
    return tuple(d_list), value
model_name = "FSR_DTA"

# 实例化模型
epoch = 0
model = FSR_DTA(
        n_drug=66,
        n_target=26,
        hidden_dim=128,
        dropout=0.,
    )
# ===============================
# Model Information
# ===============================
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total Params      : {total_params:,}")
print(f"Trainable Params  : {trainable_params:,}")

# ===============================
# Device
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ===============================
# Load Checkpoint
# ===============================
model_path = f"./model/{model_name}_{dataset}_{start}.pth"

start_epoch = 0
best_mse = float("inf")

if os.path.isfile(model_path):

    print("=" * 60)
    print(f"Loading checkpoint:")
    print(model_path)

    checkpoint = torch.load(
        model_path,
        map_location=device
    )

    # ----------兼容旧版本----------
    if isinstance(checkpoint, dict) and "model" in checkpoint:

        model.load_state_dict(checkpoint["model"])

        start_epoch = checkpoint.get("epoch", 0) + 1
        best_mse = checkpoint.get("best_mse", float("inf"))

        print(f"Resume Epoch : {start_epoch}")
        print(f"Best MSE     : {best_mse:.6f}")

    else:
        # 兼容以前只保存state_dict的模型
        model.load_state_dict(checkpoint)
        print("Loaded old-format state_dict.")

    print("=" * 60)

else:
    print("=" * 60)
    print("No checkpoint found.")
    print("Training from scratch.")
    print("=" * 60)

# 将模型移动到 GPU (如果可用)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model loaded on {device}")
##############################################################
# Loss / Optimizer
##############################################################

loss_fn = nn.MSELoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-4,
    betas=(0.9, 0.999)
)

##############################################################
# AMP
##############################################################

scaler = torch.cuda.amp.GradScaler()

##############################################################
# Scheduler
##############################################################

warmup_epochs = 10
max_epoch = 500

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=20,
    T_mult=2,
    eta_min=1e-6
)

##############################################################
# Early Stop
##############################################################

best_mse = 1e9
patience = 70
counter = 0

##############################################################
# Train
##############################################################

for epoch in range(max_epoch):

    ###########################################
    # Warmup
    ###########################################

    if epoch < warmup_epochs:

        lr = 3e-4 * (epoch + 1) / warmup_epochs

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    ###########################################
    # Train
    ###########################################

    model.train()

    running_loss = 0.

    train_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}/{max_epoch}"
    )

    for tr_data in train_bar:

        data, value = data_process(tr_data)

        optimizer.zero_grad(set_to_none=True)

        ###########################################
        # AMP Forward
        ###########################################

        with torch.cuda.amp.autocast():

            pred = model(data)

            mse_loss = nn.functional.mse_loss(
                pred,
                value
            )

            huber_loss = nn.functional.smooth_l1_loss(
                pred,
                value
            )

            loss = 0.8 * mse_loss + 0.2 * huber_loss

        ###########################################
        # Backward
        ###########################################

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5.0
        )

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item()

        train_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.2e}"
        )

    ###########################################
    # Scheduler
    ###########################################

    if epoch >= warmup_epochs:

        scheduler.step(epoch - warmup_epochs)

    train_loss = running_loss / len(train_loader)

    print(f"\nTrain Loss : {train_loss:.5f}")

    ###########################################
    # Validation
    ###########################################

    model.eval()

    y = []
    p = []

    val_loss = 0.

    with torch.no_grad():

        test_bar = tqdm(
            test_loader,
            desc="Validation"
        )

        for te_data in test_bar:

            data, value = data_process(te_data)

            with torch.cuda.amp.autocast():

                output = model(data)

                loss = nn.functional.mse_loss(
                    output,
                    value
                )

            val_loss += loss.item()

            y.extend(
                value.cpu().numpy().flatten()
            )

            p.extend(
                output.cpu().numpy().flatten()
            )
    val_loss /= len(test_loader)

    ###########################################
    # Metrics
    ###########################################

    mse = calculate_metrics(
        np.array(y),
        np.array(p),
        dataset=dataset + "_" + model_name + "_" + start,
        type="test"
    )

    ###########################################
    # Save
    ###########################################

    if mse < best_mse:

        best_mse = mse

        counter = 0

        torch.save(
            {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "best_mse": best_mse,
            },
            model_path,
        )

        print("=" * 60)
        print(f"Best Model Saved")
        print(f"Epoch : {epoch+1}")
        print(f"MSE   : {best_mse:.6f}")
        print("=" * 60)

    else:

        counter += 1

        print(
            f"No Improvement ({counter}/{patience})"
        )

    ###########################################
    # Epoch Summary
    ###########################################

    print(
        f"Epoch [{epoch+1}/{max_epoch}] "
        f"Train Loss={train_loss:.5f} "
        f"Val Loss={val_loss:.5f} "
        f"Best MSE={best_mse:.5f}"
    )

    ###########################################
    # Early Stop
    ###########################################

    if counter >= patience:

        print("\nEarly stopping!")

        break

print("\nTraining Finished.")