# U-net(convnext)
class UNet(nn.Module):
    def __init__(self, convnext_backbone,in_channels, num_classes):
        super().__init__()

        #事前学習済みモデルを用いる
        #エンコーダ
        self.encoder = ConvNeXtEncoder(convnext_backbone,in_channels)

        # --- Global Average Pooling 関連の追加 ---
        # e4のチャンネル数は768。GAPで(Bx768x1x1)にした後、1x1 Convで調整
        self.gap = nn.AdaptiveAvgPool2d(1)

        # 改善案：ボトルネック構造の導入
        self.gap_conv = nn.Sequential(
            nn.Conv2d(768, 48, kernel_size=1), # 圧縮
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 768, kernel_size=1), # 復元
            nn.Sigmoid()
            )

        #デコーダ
        self.dec3 = DoubleConv(768 + 384, 384,dropout_prob=0.3)

        self.dec2 = DoubleConv(384 + 192, 192,dropout_prob=0.2)

        self.dec1 = DoubleConv(192 + 96, 96)

#         # 追加: 1/4から元サイズ(1/1)へ戻すためのデコーダ
#         # Swinの最小解像度は 1/4 なので、ここからさらに2倍×2倍のアップサンプルが必要
#         self.up0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
#         self.dec0 = DoubleConv(96, 64) # スキップ接続なし、または元の入力を使う

#         self.final = nn.Conv2d(64, num_classes, kernel_size=1)

        # 1/4 -> 1/2 -> 1/1 のステップをより丁寧に
        # 入力画像(1/1)から低次の特徴を抽出する層 (Skip用)
        self.initial_feat = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.up0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec0 = DoubleConv(96, 64)

        self.up_final = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec_final = DoubleConv(64+32, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        #事前学習済みモデルを使用
        x_in = self.initial_feat(x)

        features=self.encoder(x)

        e1,e2,e3,e4=features#[1/4,1/8,1/16,1/32]の解像度

        #エンコーダ
        #e1#Bxin_channelsx320x256→Bx96x80x64
        #e2#Bx96x80x64→Bx192x40x32
        #e3#Bx192x40x32→Bx384x20x16
        #e4#Bx384x20x16→Bx768x10x8

        # --- GAPの実装 ---
        # ボトルネック(e4)に対してグローバルな情報を抽出
        gap_feat = self.gap(e4)           # Bx768x10x8 -> Bx768x1x1
        gap_feat = self.gap_conv(gap_feat) # チャンネルごとの重要度を計算

        # e4にグローバルな情報を掛け合わせる（SE-Blockのような仕組み）
        e4_weighted = e4 * gap_feat
        # -----------------

        #デコーダ
        up3 = F.interpolate(e4_weighted, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([up3, e3], dim=1))#Bx(768+384)x20x16→Bx384x20x16

        up2 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([up2, e2], dim=1))#Bx(384+192)x40x32→Bx128x40x32

        up1 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([up1, e1], dim=1))#Bx(192+96)x80x64→Bx96x80x64

        d0 = self.up0(d1)#Bx96x80x64→Bx96x160x128
        d0 = self.dec0(d0)#Bx96x160x128→Bx64x160x128

        #out = F.interpolate(d0, size=x.shape[2:], mode='bilinear', align_corners=False)#Bx64x160x128→Bx64x320x256
        # 1/2 -> 1/1 (入力時の特徴量 x_in と結合)
        out = self.up_final(d0)#Bx64x160x128→Bx64x320x256
        out = self.dec_final(torch.cat([out, x_in], dim=1))#Bx(64+32)x320x256→Bx64x320x256

        #return self.final(out)#Bx64x320x256→Bx13x320x256
        return self.final_conv(out)#Bx64x320x256→Bx13x320x256

#uper-Unet(convnext)
class UperUNet(nn.Module):
    def __init__(self, convnext_backbone, in_channels, num_classes):
        super().__init__()

        self.encoder = ConvNeXtEncoder(convnext_backbone, in_channels)

        # 1. PPM (ボトルネック部分を強化)
        # ConvNeXt stage3の出力 768ch を受けて 256ch などに集約
        self.ppm = PPM(in_channels=768, out_channels=256)

        # 2. FPN Lateral Layers (各ステージのチャンネル数を統一)
        # デコーダでの結合をスムーズにするため、各層を同じチャンネル数(例: 512)に変換
        self.lateral3 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.lateral2 = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.lateral1 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # 3. Refinement / Decoding Layers
        # U-Net的な DoubleConv を使用して融合
        self.dec3 = DoubleConv(256 + 256, 256, dropout_prob=0.4)
        self.dec2 = DoubleConv(256 + 256, 128, dropout_prob=0.3)
        self.dec1 = DoubleConv(128 + 256, 128, dropout_prob=0.2)

        # 4. Final Upsampling & Head (1/4 -> 1/1)
        self.initial_feat = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
#         self.up_final = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)
#         self.dec_final = DoubleConv(128 + 64, 64)

        self.up0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec0 = DoubleConv(128, 64)

        self.up_final = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec_final = DoubleConv(64+32, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Initial Feature for Skip Connection (1/1 res)
        x_in = self.initial_feat(x)

        # Encoder (ConvNeXt)
        e1, e2, e3, e4 = self.encoder(x) # [1/4, 1/8, 1/16, 1/32]

        #エンコーダ
        #e1#Bxin_channelsx320x256→Bx96x80x64
        #e2#Bx96x80x64→Bx192x40x32
        #e3#Bx192x40x32→Bx384x20x16
        #e4#Bx384x20x16→Bx768x10x8

        # PPM
        p4 = self.ppm(e4) #Bx768x10x8→Bx256x10x8
        # FPN-style Top-down path + U-Net Skip Connections
        # Stage 3 (1/16)
        p3 = F.interpolate(p4, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([p3, self.lateral3(e3)], dim=1)) # 256 ch #Bx(256+256)x20x16→Bx256x20x16

        # Stage 2 (1/8)
        p2 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([p2, self.lateral2(e2)], dim=1)) # 128 ch #Bx(256+256)x40x32→Bx128x40x32

        # Stage 1 (1/4)
        p1 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([p1, self.lateral1(e1)], dim=1)) # 128 ch #Bx(128+256)x80x64→Bx128x80x64

#         # Final Output (1/4 -> 1/1)
#         out = self.up_final(d1)
#         out = self.dec_final(torch.cat([out, x_in], dim=1))

#         return self.final_conv(out)
        d0 = self.up0(d1)#Bx128x80x64→Bx128x160x128
        d0 = self.dec0(d0)#Bx128x160x128→Bx64x160x128

        #out = F.interpolate(d0, size=x.shape[2:], mode='bilinear', align_corners=False)#Bx64x160x128→Bx64x320x256
        # 1/2 -> 1/1 (入力時の特徴量 x_in と結合)
        out = self.up_final(d0)#Bx64x160x128→Bx64x320x256
        out = self.dec_final(torch.cat([out, x_in], dim=1))#Bx(64+32)x320x256→Bx64x320x256

        #return self.final(out)#Bx64x320x256→Bx13x320x256
        return self.final_conv(out)#Bx64x320x256→Bx13x320x256

#U-net(swin_transformer_panopatic)
class UNet(nn.Module):
    def __init__(self, swin_backbone,in_channels, num_classes):
        super().__init__()

        #事前学習済みモデルを用いる
        #エンコーダ
        self.backbone=swin_backbone
        self.encoder = self.backbone

        old_conv=self.backbone.embeddings.patch_embeddings.projection#(3,96)

        new_conv = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size, stride=old_conv.stride, padding=old_conv.padding
        )


        with torch.no_grad():
        # RGB は pretrained をコピー
            new_conv.weight.data[:, :3] = old_conv.weight.data

        # Depth は RGB の平均で初期化
            new_conv.weight.data[:, 3:4] = old_conv.weight.data.mean(dim=1, keepdim=True)


        self.backbone.embeddings.patch_embeddings.projection=new_conv

        # --- 追加: これでエラーを回避します ---
        self.backbone.config.num_channels = in_channels
        # もし SwinPatchEmbeddings 自身にプロパティがある場合も更新
        if hasattr(self.backbone.embeddings.patch_embeddings, "num_channels"):
            self.backbone.embeddings.patch_embeddings.num_channels = in_channels

        #デコーダ
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec3 = DoubleConv(768 + 384, 384,dropout_prob=0.3)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = DoubleConv(384 + 192, 192,dropout_prob=0.2)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec1 = DoubleConv(192 + 96, 96)

        # 追加: 1/4から元サイズ(1/1)へ戻すためのデコーダ
        # Swinの最小解像度は 1/4 なので、ここからさらに2倍×2倍のアップサンプルが必要
        self.up0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec0 = DoubleConv(96, 64) # スキップ接続なし、または元の入力を使う

        self.up_final = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        #事前学習済みモデルを使用
        # 1. バックボーンから特徴抽出
        outputs = self.backbone(x)

        # SwinBackboneの出力オブジェクトからテンソルのリストを取り出す
        # これにより 'str' (名前) ではなく torch.Tensor が取得できます
        features = outputs.feature_maps

        # features[0]: 1/4  (B, 96, 80, 60) ※320x240入力の場合
        # features[1]: 1/8  (B, 192, 40, 30)
        # features[2]: 1/16 (B, 384, 20, 15)
        # features[3]: 1/32 (B, 768, 10, 7)

        e1, e2, e3, e4 = features

        #エンコーダ
        #e1#Bxin_channelsx224x224→Bx56x56x56
        #e2#Bx96x56x56→Bx192x28x28
        #e3#Bx192x28x28→Bx384x14x14
        #e4#Bx384x14x14→Bx768x7x7


        #デコーダ
        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))#Bx(768+384)x14x14→Bx384x14x14
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))#Bx(384+192)x28x28→Bx128x28x28
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))#Bx(192+96)x56x56→Bx96x56x56
        d0 = self.up0(d1)#Bx96x56x56→Bx96x112x112
        d0 = self.dec0(d0)#Bx96x112x112→Bx64x112x112

        out = self.up_final(d0)#Bx64x112x112→Bx64x224x224

        return self.final(out)#Bx64x224x224→Bx13x224x224
   
#uper-Unet(swin_transformer_panopatic)
class SwinUperNet(nn.Module):
    def __init__(self, swin_backbone, in_channels, num_classes):
        super().__init__()

        # --- エンコーダの設定 (Swin Transformer) ---
        self.backbone = swin_backbone

        # 入力チャンネル数の変更 (RGB 3ch -> in_channels)
        old_conv = self.backbone.embeddings.patch_embeddings.projection
        new_conv = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size, stride=old_conv.stride, padding=old_conv.padding
        )
        with torch.no_grad():
            new_conv.weight.data[:, :3] = old_conv.weight.data
            if in_channels > 3:
                new_conv.weight.data[:, 3:in_channels] = old_conv.weight.data.mean(dim=1, keepdim=True)

        self.backbone.embeddings.patch_embeddings.projection = new_conv
        self.backbone.config.num_channels = in_channels
        if hasattr(self.backbone.embeddings.patch_embeddings, "num_channels"):
            self.backbone.embeddings.patch_embeddings.num_channels = in_channels

        # --- UperNet 構成要素 ---

        # 1. PPM: Swinの最終層 (1/32) 768ch -> 256ch
        self.ppm = PPM(in_channels=768, out_channels=256)

        # 2. FPN Lateral Layers: Swinの各出力 [1/16, 1/8, 1/4] を 256ch に統一
        self.lateral3 = nn.Sequential(
            nn.Conv2d(384, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.lateral2 = nn.Sequential(
            nn.Conv2d(192, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.lateral1 = nn.Sequential(
            nn.Conv2d(96, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        # 3. Decoding Layers (FPN Top-down + Skip Connection)
        self.dec3 = DoubleConv(256 + 256, 256, dropout_prob=0.3) # 1/16
        self.dec2 = DoubleConv(256 + 256, 128, dropout_prob=0.2) # 1/8
        self.dec1 = DoubleConv(128 + 256, 128, dropout_prob=0.1) # 1/4

        # 4. Final Reconstruction (1/4 -> 1/1)
        # 入力の高解像度情報を保持するための初期特徴抽出
        self.initial_feat = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.up0 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) # 1/4 -> 1/2
        self.dec0 = DoubleConv(128, 64)

        self.up_final = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) # 1/2 -> 1/1
        self.dec_final = DoubleConv(64 + 32, 64)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # 0. 入力サイズの保存と初期特徴抽出
        x_in = self.initial_feat(x)

        # 1. Encoder (Swin Backbone)
        outputs = self.backbone(x)
        # features[0]:1/4, [1]:1/8, [2]:1/16, [3]:1/32
        e1, e2, e3, e4 = outputs.feature_maps

        # 2. PPM (1/32 特徴)
        p4 = self.ppm(e4) # 256ch

        # 3. Top-down path (UperNet/UNet style)
        # Stage 3 (1/16)
        p3 = F.interpolate(p4, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([p3, self.lateral3(e3)], dim=1))

        # Stage 2 (1/8)
        p2 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([p2, self.lateral2(e2)], dim=1))

        # Stage 1 (1/4)
        p1 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([p1, self.lateral1(e1)], dim=1))

        # 4. Upsampling to Original Size
        # 1/4 -> 1/2
        d0 = self.up0(d1)
        d0 = self.dec0(d0)

        # 1/2 -> 1/1
        out = self.up_final(d0)
        out = self.dec_final(torch.cat([out, x_in], dim=1))

        return self.final_conv(out)