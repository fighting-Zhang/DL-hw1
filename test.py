import numpy as np
import json
from model import NeuralNetwork
from dataset import load_fashion_mnist, preprocess_data
from utils import compute_accuracy
import os

def load_model_params(filepath):
    """加载模型参数"""
    data = np.load(filepath, allow_pickle=True)
    return {key: data[key] for key in data.files}

def load_hyperparameters(filepath):
    """从JSON文件加载超参数"""
    with open(filepath, 'r') as f:
        hyperparams = json.load(f)
    return hyperparams


def evaluate(model, X_test, Y_test):
    """测试模型并计算准确率"""
    Y_pred, _ = model.forward(X_test)
    accuracy = compute_accuracy(Y_pred, Y_test)
    return accuracy

def main():
    result_dir = 'results_lr_0.1_drop_0.7_hidden_512_l2_0.0'
    hyperparams = load_hyperparameters(os.path.join(result_dir, 'hyperparams.json'))
    print("Loaded Hyperparameters:", hyperparams)

    model = NeuralNetwork([784, int(hyperparams['hidden_size']), 10], activation=hyperparams["activation"])
    
    model_params = load_model_params(os.path.join(result_dir, 'best_model_params.npz'))
    model.params = model_params

    X_test, Y_test = load_fashion_mnist('./fashion-mnist', training=False)  
    X_test, Y_test = preprocess_data(X_test, Y_test)

    accuracy = evaluate(model, X_test, Y_test)
    print(f"Model accuracy on test set: {accuracy:.2%}")

if __name__ == "__main__":
    main()
