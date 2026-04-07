# モデルとトレーニングの設定
# device = config.device
# print(f"Using device: {device}")

# ------------------
#    Model
# ------------------
#convnextの時

#unet
#model = UNet(convnext_backbone=convnext,in_channels=config.in_channels, num_classes=config.num_classes).to(device)
#uperunet
#model = UperUNet(convnext_backbone=convnext,in_channels=config.in_channels, num_classes=config.num_classes).to(device)

#swin_transformerの時

#uent
model = UNet(swin_backbone=swin_transformer,in_channels=config.in_channels, num_classes=config.num_classes).to(device)
#uperunet
#model = SwinUperNet(swin_backbone=swin_transformer,in_channels=config.in_channels, num_classes=config.num_classes).to(device)

# ------------------
#    optimizer
# ------------------

# --------------------------------------------------------
#    Optimizer (差分学習率設定)(convnext)
# ---------------------------------------------------------

# 1. パラメータをグループ化する
# エンコーダ（事前学習済みモデル）のパラメータ
# encoder_params = list(model.encoder.parameters())

# デコーダおよびその他の新規層のパラメータ
# encoder以外の全てのパラメータを抽出
# other_params = [
#     p for n, p in model.named_parameters()
#     if not n.startswith('encoder.')
# ]

# # 2. 学習率の設定
# # 例: デコーダ（新規層）は通常の学習率、エンコーダ（既習層）は 1/10 に設定
# base_lr = config.learning_rate
# params_to_optimize = [
#     {'params': encoder_params, 'lr': base_lr * 0.80}, # 低い学習率
#     {'params': other_params,   'lr': base_lr}        # 通常の学習率
# ]

# # AdamW を使用
# optimizer = optim.AdamW(params_to_optimize, weight_decay=config.weight_decay)


# --------------------------------------------------------
#    Optimizer (差分学習率設定)(swin_transormer)
# ---------------------------------------------------------

#↓凍結するとき
# --- バックボーンの重みを固定 ---
# for param in model.backbone.parameters():
#     param.requires_grad = False

# # --- 入力層（patch_embed.proj）はチャンネル数を変えたので学習対象にする ---
# for param in model.backbone.embeddings.patch_embeddings.projection.parameters():
#     param.requires_grad = True

# params_to_update = [p for p in model.parameters() if p.requires_grad]
# optimizer = optim.AdamW(params_to_update, lr=config.learning_rate,weight_decay=config.weight_decay)


#↓差分学習率の時
# # 1. パラメータをグループ化する
# # エンコーダ（事前学習済みモデル）のパラメータ
# # --- バックボーンの重みを固定 ---
def get_optimizer_params(model, weight_decay, base_lr):
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    optimizer_grouped_parameters = []

    for nm, p in param_dict.items():
        # ウェイト減衰から除外するパラメータ
        wd = 0.0 if any(nd in nm.lower() for nd in ["bias", "norm", "layernorm"]) else weight_decay

        # 判定を「backbone.」で始まるかどうかに絞る
        # self.backbone = swin_backbone と定義しているため
        if nm.startswith("backbone."):
            # 4ch化した初層以外は 1/10 の学習率にする
            if "patch_embeddings.projection" in nm or "patch_embed.proj" in nm:
                lr = base_lr
            else:
                lr = base_lr * 0.1  # ここが 0.00002 になるべき
        else:
            lr = base_lr  # デコーダー部分

        optimizer_grouped_parameters.append({"params": [p], "weight_decay": wd, "lr": lr})
    return optimizer_grouped_parameters
# 実行
grouped_params = get_optimizer_params(model, weight_decay=config.weight_decay, base_lr=config.learning_rate)
optimizer = optim.AdamW(grouped_params) # lrはグループごとに設定されているためここでは不要
# 設定されている学習率のユニークな値を表示
print("設定されている学習率一覧:", set([group['lr'] for group in optimizer.param_groups]))
#-------------------------------------------------------------



# ------------------
#  CutMix 関数 (セグメンテーション用)
# ------------------
def apply_cutmix(inputs, targets, alpha=1.0, p_cutmix=0.5):
    r = np.random.rand()
    if r >  p_cutmix:
        return inputs, targets, 1.0, False # 通常時

    batch_size = inputs.size(0)
    index = torch.randperm(batch_size).to(inputs.device)

    lam = np.random.beta(alpha, alpha)

    # 破壊的変更を避けるため clone() を使用
    inputs = inputs.clone()
    targets = targets.clone()

    # CutMix
    W, H = inputs.size(2), inputs.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)



    inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[index, :, bbx1:bbx2, bby1:bby2]
    # ラベルも同様の矩形領域を入れ替え
    targets[:, bbx1:bbx2, bby1:bby2] = targets[index, bbx1:bbx2, bby1:bby2]

    # 実際の面積比でlamを再計算
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    return inputs, targets, lam, True

#-------------------
#     Loss
#-------------------
#クロスエントロピー
criterion = nn.CrossEntropyLoss(ignore_index=255)

#重み付きクロスエントロピー
#criterion = nn.CrossEntropyLoss(weight=weights.to(device), ignore_index=255)

#Focal Loss
#criterion = FocalLoss(ignore_index=255)

#Dice Loss
criterion_dice=MultiClassDiceLoss(ignore_index=255)

#-------------------
#     mIOU
#-------------------
metrics = mIoUScore(n_classes=config.num_classes,ignore_index=255)

#------------------
#    Early stopping
#------------------
# 冒頭で時間を固定しておくとファイル管理が楽です
start_time = time.strftime("%Y%m%d%H%M%S")

#convnextの時
# early_stopping = EarlyStopping(patience=80, path=f"best_model_{start_time}.pt")
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=450)

#swin transformerの時
early_stopping = EarlyStopping(patience=30, path=f"best_model_{start_time}.pt")
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=250)

#-------------------
#     Lossの可視化
#------------------
# --- 1. 履歴保存用のリストを初期化 ---
history = {
    'train_loss': [],
    'val_loss': [],
    'val_miou': []
}

# ------------------
#    Training
# ------------------
num_epochs = config.epochs
scaler = GradScaler()

#model.train()
for epoch in range(num_epochs):
    train_total_loss = 0
    print(f"on epoch: {epoch+1}")
#     with tqdm(train_data) as pbar:

    model.train()

    #convnextのとき
    #cutmix_warmup_end=300

    #swin_transformetの時
    cutmix_warmup_end=200

    for batch_idx, (image, depth, label) in enumerate(train_data):
            image, depth, label = image.to(device), depth.to(device), label.to(device)
            optimizer.zero_grad()

            with autocast():
              #depth画像の正規化
              depth_normalized = (depth - depth_mean) / depth_std

              x = torch.cat((image, depth_normalized), dim=1) # (B,3,320,240)+(B,1,320,240)→(B,4,320,240)

               #cutmixの適応
               #p_cutmix=0.5 など、適用率を調整してください
              if epoch < cutmix_warmup_end:
                    x, mixed_label, lam, is_cutmix = apply_cutmix(
                        x, label, alpha=1.5, p_cutmix=0.5
                        )
                    pred = model(x)#(B,4,320,240)→(B,13,320,240)

#               　  3. Lossの計算 (Mixupの場合はラベルを混ぜる)
#               　  CutMix または 通常 (CutMixは矩形置換済みなので単一LossでOK)
#             　　  loss = criterion(pred, mixed_label)
#                   loss = criterion(pred, label)#pred(B,13,320,240),label[各要素は0~12](B,320,240)
                    loss=0.8*criterion(pred,mixed_label)+0.2*criterion_dice(pred,mixed_label)
              else:
                    pred = model(x)#(B,4,320,240)→(B,13,320,240)
                    loss=0.8*criterion(pred,label)+0.2*criterion_dice(pred,label)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_total_loss += loss.item()
            #del image, depth, label, pred, loss

    #print(f'Epoch {epoch+1}, Train_Loss: {total_loss / len(train_data)}')

    avg_train_loss = train_total_loss / len(train_data)

# ---------------
#      Validation
#----------------
    model.eval()
    val_total_loss = 0
    metrics.reset() # mIoU計算用の混同行列をリセット

    with torch.no_grad():
        for image_v, depth_v, label_v in val_data:
            image_v, depth_v, label_v = image_v.to(device), depth_v.to(device), label_v.to(device)

            with autocast():
                depthv_normalized = (depth_v - depth_mean) / depth_std

                x_v = torch.cat((image_v, depthv_normalized), dim=1)
                pred_v = model(x_v)
                #v_loss = criterion(pred_v,label_v)
                v_loss = 0.8*criterion(pred_v, label_v) + 0.2*criterion_dice(pred_v, label_v)

            val_total_loss += v_loss.item()

            # mIoU計算のために予測値をインデックスに変換
            preds_idx = torch.argmax(pred_v, dim=1)
            # numpyに変換して metrics に追加
            metrics.update(label_v.cpu().numpy(), preds_idx.cpu().numpy())

    avg_val_loss = val_total_loss / len(val_data)
    current_miou = metrics.get_scores()

    # --- 2. 履歴をリストに追加 ---
    history['train_loss'].append(avg_train_loss)
    history['val_loss'].append(avg_val_loss)
    history['val_miou'].append(current_miou)

    # 学習率スケジューラーの更新
    scheduler.step()
    e_lr = optimizer.param_groups[0]['lr']#エンコーダのLR

    #↓凍結学習の時は消す
    d_lr = optimizer.param_groups[1]['lr'] # デコーダのLR

    # ログ出力
    print(f'  [Train] Loss: {avg_train_loss:.4f}')

    #↓凍結学習の時
    #print(f'  [Val]   Loss: {avg_val_loss:.4f}, mIoU: {current_miou:.4f}, E_LR: {e_lr:.8f}')

    #↓差分学習の時
    print(f'  [Val]   Loss: {avg_val_loss:.4f}, mIoU: {current_miou:.4f}, E_LR: {e_lr:.8f},D_LR: {d_lr:.8f}')

    # --- Early Stopping 判定 ---
    # mIoUをスコアとして渡し、改善がなければカウンタが増える
    early_stopping(current_miou, model)

    if early_stopping.early_stop:
        print("Early stopping triggered. Training stopped.")
        break

# ------------------
#    3. グラフの描画
# ------------------
epochs_range = range(1, len(history['train_loss']) + 1)

plt.figure(figsize=(12, 5))

# Lossのグラフ
plt.subplot(1, 2, 1)
plt.plot(epochs_range, history['train_loss'], label='Train Loss')
plt.plot(epochs_range, history['val_loss'], label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# mIoUのグラフ
plt.subplot(1, 2, 2)
plt.plot(epochs_range, history['val_miou'], label='Val mIoU', color='green')
plt.title('Validation mIoU')
plt.xlabel('Epochs')
plt.ylabel('mIoU')
plt.legend()

plt.tight_layout()
plt.savefig(f"learning_curve_{start_time}.png") # グラフを画像として保存
plt.show()


# ------------------
#    Finalize
# ------------------
# 学習終了後、Early Stoppingによって保存されたベストモデルをロードする
model.load_state_dict(torch.load(early_stopping.path))
print(f"Loaded best model from {early_stopping.path}")

# モデルの保存
current_time = time.strftime("%Y%m%d%H%M%S")
model_path = f"model_{current_time}.pt"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")