class MultiClassDiceLoss(nn.Module):
    def __init__(self, num_classes=13, ignore_index=None,smooth=1e-6):
        super(MultiClassDiceLoss, self).__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: [Batch, 13, H, W] - モデルの出力（Softmax適用前）
        targets: [Batch, H, W] - 正解ラベル（0~12のインデックス）
        """
        # 1. Softmaxを適用して確率にする
        probs = F.softmax(logits, dim=1)

        # 2. targetsをOne-hot形式に変換 [Batch, H, W] -> [Batch, 13, H, W]
        # その後、計算のために次元を入れ替える

        # ignore_index が範囲外(例: 255)でもエラーにならないよう一旦クランプ
        targets_clamped = targets.clone()
        if self.ignore_index is not None:
            mask = (targets != self.ignore_index)
            targets_clamped[~mask] = 0  # 一時的に0を代入

        targets_one_hot = F.one_hot(targets_clamped, num_classes=self.num_classes)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()

        # 3. ignore_index のピクセルを強制的に 0 にする
        if self.ignore_index is not None:
            # mask: [B, H, W] -> [B, 1, H, W] に拡張して全チャンネルに適用
            mask = mask.unsqueeze(1).float()
            probs = probs * mask
            targets_one_hot = targets_one_hot * mask

        # 3. クラスごとにDiceを計算するため、バッチと空間次元をフラット化
        # [Batch, 13, H, W] -> [13, Batch * H * W]
        probs = probs.transpose(0, 1).reshape(self.num_classes, -1)
        targets_one_hot = targets_one_hot.transpose(0, 1).reshape(self.num_classes, -1)

        # 4. クラスごとの交差部分と和を計算
        intersection = (probs * targets_one_hot).sum(dim=1)
        cardinality = probs.sum(dim=1) + targets_one_hot.sum(dim=1)

        # 5. 各クラスのDice係数を計算
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)

#         # 6. ignore_index に対応するクラスを平均から除外する場合
#         if self.ignore_index is not None and 0 <= self.ignore_index < self.num_classes:
#             # 指定されたインデックス以外のDiceスコアのみ抽出
#             relevant_indices = [i for i in range(self.num_classes) if i != self.ignore_index]
#             dice_score = dice_score[relevant_indices]

#         # 6. 平均Dice Lossを返す (1 - Mean Dice Score)
#         return 1. - dice_score.mean()

        # --- 修正: そのバッチに存在するクラスのみで平均を取る ---
        # ターゲットに1ピクセルも存在しないクラスは、予測を頑張る必要がないため除外する
        # (ただし smooth があるので 0 除算は起きませんが、精度に悪影響を与えます)

        # 各クラスのターゲットの合計が 0 より大きいものだけを抽出
        exist_mask = targets_one_hot.sum(dim=1) > 0

        # 存在したクラスの Dice Score だけを平均する
        if exist_mask.any():
            return 1. - dice_score[exist_mask].mean()
        else:
            # 万が一、全てのピクセルが ignore_index だった場合
            return 1. - dice_score.mean()