# データ前処理の定義(Albumentationによる実装)
# 訓練用：データ拡張を含む
train_transform = A.Compose([
    A.Resize(config.image_size[1], config.image_size[0]),
    # --- 幾何学的変換 (RGB & Depth 両方に適用) ---
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.2, rotate_limit=15, p=0.7, border_mode=0),
    A.RandomResizedCrop(height=config.image_size[1], width=config.image_size[0], scale=(0.5, 1.0), p=0.7),

    # 室内パースを変化させる歪み (追加推奨)
    A.OneOf([
        A.GridDistortion(p=1.0),
        A.OpticalDistortion(distort_limit=0.2, shift_limit=0.2, p=1.0),
    ], p=0.3),

    # --- 色・質感の変換 (RGBのみに適用するのが理想) ---
    # ※Albumentationsでは targets={'depth':'mask'} としている場合、
    #  一部の関数はRGBにしか効かないのでそのまま入れてOKです
    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=1.0),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
        A.CLAHE(p=1.0), # コントラスト強調
    ], p=0.5),

    A.OneOf([
        A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        A.Sharpen(p=1.0),
    ], p=0.3),

    # --- 強力な正則化 (追加推奨) ---
    # モデルに「特定のピクセルに依存させない」ための欠落
    A.CoarseDropout(max_holes=8, max_height=config.image_size[1]//10,
                    max_width=config.image_size[0]//10, min_holes=4, p=0.4),
    #正規化
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
],
    additional_targets={'depth': 'mask'}) # depthをmaskと同じように扱う

# テスト用：リサイズと正規化のみ
test_transform = A.Compose([
    A.Resize(config.image_size[1], config.image_size[0]),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
],
    additional_targets={'depth': 'mask'})
