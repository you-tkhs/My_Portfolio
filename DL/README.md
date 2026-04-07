# Semantic Segmentation (NYUv2)

## 概要
RGB画像と深度マップを入力とした室内シーンの
セマンティックセグメンテーション。

- **タスク**: 13クラスのピクセル単位分類
- **データ**: NYUv2データセット（訓練795枚）
- **結果**: mIOU 0.382（ベースライン）→ 0.658
- **順位**: 23位 / 373人中

## 主な工夫
- ConvNeXtをbackboneとしたUperUNet
- Panoptic Segmentationで事前学習済みSwin Transformerとのアンサンブル
- RGBと深度マップを4chで入力するよう改造
- 少データに対応したデータ拡張（室内環境を考慮）
- Dice Loss + Cross Entropyの組み合わせ
- TTA（Test Time Augmentation）

## 使用技術
- PyTorch / timm / transformers / albumentations

## 詳細
取り組みの詳細は[レポート](report.pdf)を参照してください。

## Note
本コードはGoogle Colabのノートブックから
ポートフォリオ用に抜粋・整理したものです。
データセットおよび課題の詳細は非公開のため、
推論の実行コードは含まれていません。