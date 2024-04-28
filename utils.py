import matplotlib.pyplot as plt
import numpy as np
import os

def plot_history(history, results_dir):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    plt.title('Loss during training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['val_accuracy'], label='Validation Accuracy')
    plt.title('Validation Accuracy during training')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.savefig(os.path.join(results_dir, 'history.png'))


def save_model_params(params, filepath):
    np.savez(filepath, **params)

def load_model_params(filepath):
    data = np.load(filepath)
    return {key: data[key] for key in data.files}


def cross_entropy_loss(y_pred, y_true):
    m = y_true.shape[1]  # 样本数量
    # 计算交叉熵损失
    log_likelihood = -np.sum(y_true * np.log(y_pred + 1e-9))  # 添加一个小常数以防止对数为负无穷
    loss = log_likelihood / m
    return loss


def l2_loss(params, lambda_reg):
    """Compute L2 regularization loss."""
    l2 = 0.0
    for k in params:
        if 'W' in k:
            l2 += (lambda_reg / 2) * np.sum(np.square(params[k]))
    return l2

def apply_l2_regularization(grads, params, lambda_reg, batch_size):
    """Apply L2 regularization to the gradients."""
    for k in params:
        if 'W' in k:
            grads['dW' + k[1:]] += (lambda_reg / batch_size) * params[k]
    return grads

def compute_accuracy(predictions, labels):
    """计算分类准确率"""
    preds = np.argmax(predictions, axis=0)
    truth = np.argmax(labels, axis=0)
    return np.mean(preds == truth)