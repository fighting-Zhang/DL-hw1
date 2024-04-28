import numpy as np
import matplotlib.pyplot as plt
from model import NeuralNetwork
import os
import json


def visualize_all_layers_weights(model, results_dir=None):
    layer_sizes = model.layer_sizes
    # 遍历所有层，注意我们只关心有权重的层，即除了输入层以外的所有层
    for i in range(1, len(layer_sizes)):
        W = model.params['W' + str(i)]
        num_neurons = W.shape[0]
        input_features = W.shape[1]
        
        # 确定子图的尺寸
        num_cols = min(10, num_neurons)  # 最多每行显示10个神经元的权重
        num_rows = (num_neurons + num_cols - 1) // num_cols
        
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 1.5, num_rows * 1.5))
        fig.suptitle(f'Weights of Layer {i}', fontsize=16)
        
        for j, ax in enumerate(axes.flat):
            if j < num_neurons:
                # 对于第一层，我们可以将权重视作28x28的图像
                if i == 1:
                    image = W[j].reshape(28, 28)
                    ax.imshow(image, cmap='viridis', interpolation='nearest')
                else:
                    # 对于其他层，可能只能显示权重向量
                    image = W[j].reshape(-1, 1)
                    ax.imshow(image, cmap='viridis', aspect='auto', interpolation='nearest')
                
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # plt.show()
        plt.savefig(os.path.join(results_dir, f'layer_{i}_weights.png'))


if __name__ == '__main__':
    # Load the model
    results_dir = 't_results'

    with open(os.path.join(results_dir, 'hyperparams.json'), 'r') as f:
        hyperparams = json.load(f)


    # 定义网络结构和训练参数
    model = NeuralNetwork([784, int(hyperparams['hidden_size']), 10], 
                          activation=hyperparams['activation'])
    
    model.load_params(os.path.join(results_dir, 'best_model_params.npz'))
    
    # Visualize the weights of all layers
    visualize_all_layers_weights(model, results_dir)
