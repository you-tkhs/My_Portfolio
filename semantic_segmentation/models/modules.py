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
