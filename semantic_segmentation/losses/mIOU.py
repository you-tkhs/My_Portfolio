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