# 2つの畳み込み層とバッチ正規化、ReLUを含むブロック
# UNetの各層で使用される基本的な畳み込みブロック
# 入力サイズが(4,320,256),out_channelが64と想定と想定
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels,dropout_prob=0.0):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),#4x320x256→64x((320-3+2)+1)x((256-3+2)+1)=64x320x256
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            # ここに挿入（BNの後、次のConvの前）
            nn.Dropout2d(p=dropout_prob) if dropout_prob > 0 else nn.Identity(),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),#64x320x256→((320-3+2)+1)x((256-3+2)+1)=64x320x256
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)#4x320x256→64x320x256


class PPM(nn.Module):
    """Pyramid Pooling Module: 異なるスケールで大域的情報を抽出"""
    def __init__(self, in_channels, out_channels, bins=(1, 2, 3, 6)):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(bin_size),
                nn.Conv2d(in_channels, out_channels // len(bins), kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels // len(bins)),
                nn.ReLU(inplace=True)
            ) for bin_size in bins
        ])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        h, w = x.shape[2:]
        out = [x]
        for stage in self.stages:
            # 各ビンの特徴量を元のサイズにリサイズして結合
            out.append(F.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=False))
        return self.bottleneck(torch.cat(out, dim=1))

# 下記リンク先のmIoU実装を利用
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
class mIoUScore(object):
    def __init__(self, n_classes,ignore_index=255):
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)& (label_true != self.ignore_index)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask].astype(int), minlength=n_class ** 2
        ).reshape(n_class, n_class)    # ij 成分は，target がクラス i ， 予測がクラス j だったピクセルの数
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        hist = self.confusion_matrix
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iou = np.nanmean(iou)
        return mean_iou

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))