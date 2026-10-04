# My_Portfolio

自身の開発・研究経験をまとめた物です

## コンペ

松尾研究室のコンペで上位を獲得した際の記録です。

### DL基礎講座 - Semantic Segmentation 
- **タスク**: 室内シーンのセマンティックセグメンテーション
- **結果**: mIOU 0.658
- **順位**: 上位6.1%
- 詳細は[DL/README.md](DL/README.md)を参照

### GCI - スポーツ分析における2値分類
- **タスク**: 選手データからドラフト指名の有無を予測
- **結果**: AUC 0.84705
- **順位**: 上位3.1%
- 詳細は[GCI/README.md](GCI/README.md)を参照

### 主な使用技術
- **Deep Learning**: PyTorch / timm / transformers / albumentations
- **Data Analysis**: XGBoost /  scikit-learn / Optuna / matplotlib /pandas 
- **言語**: Python

## 研究

UAV空撮画像から復元した3次元点群による稲穂圃場の把握と収量予測に取り組んでいます。
詳細は[Research/README.md](Research/README.md)を参照

### 学部：UAV画像を用いた実験水田の3次元復元
- NeRFと3DGSをPoC比較し、3DGSによるデータ拡張で復元精度を向上

### 修士：3次元点群を直接入力とした収量予測（進行中）
- PointNet/PointNet++を回帰用に改良し、区画ごとの点群から収量(g/m²)を予測
- 収量予測についてはこちら：[Point_net_for_my_research](https://github.com/you-tkhs/Point_net_for_my_research)
