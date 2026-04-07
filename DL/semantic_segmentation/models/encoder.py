class ConvNeXtEncoder(nn.Module):
    def __init__(self, base_model,in_channels):
        super().__init__()

        old_conv=base_model.stem_0#(3,96)

        new_conv = nn.Conv2d(
            in_channels, old_conv.out_channels,
            kernel_size=old_conv.kernel_size, stride=old_conv.stride, padding=old_conv.padding
        )

        with torch.no_grad():
        # RGB は pretrained をコピー
            new_conv.weight.data[:, :3] = old_conv.weight.data
        # Depth は RGB の平均で初期化
            new_conv.weight.data[:, 3:4] = old_conv.weight.data.mean(dim=1, keepdim=True)

        self.stem_0 = new_conv
        self.stem_1 = base_model.stem_1
        self.stage0 = base_model.stages_0
        self.stage1 = base_model.stages_1
        self.stage2 = base_model.stages_2
        self.stage3 = base_model.stages_3

    def forward(self, x):
        features = []
        x = self.stem_0(x) #1/4 res
        x = self.stem_1(x)
        x = self.stage0(x)
        features.append(x) # 1/4 res, 96 ch

        x = self.stage1(x)
        features.append(x) # 1/8 res, 192 ch

        x = self.stage2(x)
        features.append(x) # 1/16 res, 384 ch

        x = self.stage3(x)
        features.append(x) # 1/32 res, 768 ch

        return features
