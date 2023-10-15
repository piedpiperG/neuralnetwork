import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc


def plot_pr_curve_sorted(y_true, y_prob):
    # 根据分类器的预测结果从大到小对样例进行排序
    sorted_indices = sorted(range(len(y_prob)), key=lambda i: y_prob[i], reverse=True)
    y_true_sorted = [y_true[i] for i in sorted_indices]
    y_prob_sorted = [y_prob[i] for i in sorted_indices]

    # 初始化变量来计算PR曲线的值
    precision, recall, thresholds = [], [], []
    tp, fp, fn = 0, 0, sum(y_true_sorted)

    # 计算PR值
    for i in range(len(y_prob_sorted)):
        threshold = y_prob_sorted[i]
        thresholds.append(threshold)
        if y_true_sorted[i] == 1:
            tp += 1
            fn -= 1
        else:
            fp += 1
        precision.append(tp / (tp + fp))
        recall.append(tp / (tp + fn))

    # 计算PR曲线下面积AUC
    pr_auc = auc(recall, precision)

    # 绘制PR曲线
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label='PR curve (AUC = {:.2f})'.format(pr_auc))
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.show()


def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.show()


# 示例数据
sample = [(1, 0.9), (1, 0.8), (0, 0.7), (1, 0.6), (1, 0.55),
          (1, 0.54), (0, 0.53), (0, 0.52), (1, 0.51), (0, 0.505),
          (1, 0.4), (0, 0.39), (1, 0.38), (0, 0.37), (0, 0.36),
          (0, 0.35), (1, 0.34), (0, 0.33), (1, 0.30), (0, 0.1)]

y_true = [item[0] for item in sample]
y_prob = [item[1] for item in sample]

plot_pr_curve_sorted(y_true, y_prob)
plot_roc_curve(y_true, y_prob)
