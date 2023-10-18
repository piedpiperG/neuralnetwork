import numpy as np
from Method import predict


def accuracy(initial_nn_params, input_layer_size, hidden_layer_size, num_labels, X_train, y_train):
    # 重新分割，获得三个层次之间两两的权重
    Theta1 = np.reshape(initial_nn_params[:hidden_layer_size * (input_layer_size + 1)], (
        hidden_layer_size, input_layer_size + 1))  # shape = (100, 785)
    Theta2 = np.reshape(initial_nn_params[hidden_layer_size * (input_layer_size + 1):],
                        (num_labels, hidden_layer_size + 1))  # shape = (10, 101)
    # 训练集的准确度
    pred = predict(Theta1, Theta2, X_train)
    # print('Training Set Accuracy: {:f}'.format((np.mean(pred == y_train) * 100)))
    return np.mean(pred == y_train) * 100
