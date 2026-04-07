# config
@dataclass
class TrainingConfig:
    # データセットパス
    dataset_root: str = "data"

    # データ関連
    batch_size: int = 32
    num_workers: int = 4

    # モデル関連
    in_channels: int = 3
    num_classes: int = 13  # NYUv2データセットの場合

    # 学習関連
    epochs: int = 100
    learning_rate: float = 0.001

    #convnext
    #weight_decay: float = 1e-4

    #swin transformer
    weight_decay: float = 1e-1

    # データ分割関連
    train_val_split: float = 0.9  # 訓練データの割合

    # デバイス設定
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # チェックポイント関連
    checkpoint_dir: str = "checkpoints"
    save_interval: int = 5  # エポックごとのモデル保存間隔

    # データ拡張・前処理関連
    image_size: tuple = (256, 256)
    normalize_mean: tuple = (0.485, 0.456, 0.406)  # ImageNetの標準化パラメータ
    normalize_std: tuple = (0.229, 0.224, 0.225)

    def __post_init__(self):
        import os
        os.makedirs(self.checkpoint_dir, exist_ok=True)

# depthの統計量の出力
def calculate_depth_stats(dataloader, device):
    print("Calculating depth statistics...")
    sum_depth = 0.0
    sum_sq_depth = 0.0
    total_pixels = 0

    for _, depth, _ in dataloader:
        depth = depth.to(device)
        # バッチ内の全ピクセル数を加算
        total_pixels += depth.numel()
        # 合計値と二乗和を蓄積
        sum_depth += torch.sum(depth).item()
        sum_sq_depth += torch.sum(depth ** 2).item()

    # 平均と標準偏差の算出
    mean = sum_depth / total_pixels
    std = np.sqrt((sum_sq_depth / total_pixels) - (mean ** 2))

    return mean, std

# 統計量の取得
device = config.device
# print(f"Using device: {device}")
depth_mean, depth_std = calculate_depth_stats(train_data, device)
print(f"Dataset Depth Mean: {depth_mean:.4f}")
print(f"Dataset Depth Std:  {depth_std:.4f}")