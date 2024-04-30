import os
import numpy as np
import json
from model import NeuralNetwork
from utils import cross_entropy_loss, l2_loss, apply_l2_regularization, compute_accuracy, save_model_params, plot_history
from dataset import load_fashion_mnist, preprocess_data
from lr_scheduler import LRScheduler


def train(model, X_train, Y_train, X_val, Y_val, epochs, initial_lr, batch_size, lr_strategy, l2_reg=0.0, results_dir='results', decay_rate=0.95, epochs_drop=10, drop=0.5, milestones=[30, 60]):
    lr_scheduler = LRScheduler(initial_lr, strategy=lr_strategy, decay_rate=decay_rate, epochs_drop=epochs_drop, drop=drop, milestones=milestones)

    best_val_loss = float('inf')
    best_params = {}

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }

    for epoch in range(epochs):
        train_loss = 0
        batches = 0

        learning_rate = lr_scheduler.get_lr(epoch, epochs)

        permutation = np.random.permutation(X_train.shape[1])
        X_train_shuffled = X_train[:, permutation]  #(num_features, num_samples)
        Y_train_shuffled = Y_train[:, permutation]


        for i in range(0, X_train.shape[1], batch_size):
            X_batch = X_train_shuffled[:, i:i+batch_size]
            Y_batch = Y_train_shuffled[:, i:i+batch_size]

            Y_pred, caches = model.forward(X_batch)
            loss = cross_entropy_loss(Y_pred, Y_batch) + l2_loss(model.params, l2_reg)
            grads = model.backward(X_batch, Y_batch, caches)
            grads = apply_l2_regularization(grads, model.params, l2_reg, batch_size)
            model.update_parameters(grads, learning_rate)

            train_loss += loss
            batches += 1

        train_loss /= batches
        # Validate the model
        Y_val_pred, _ = model.forward(X_val)
        val_loss = cross_entropy_loss(Y_val_pred, Y_val) + l2_loss(model.params, l2_reg)
        val_accuracy = compute_accuracy(Y_val_pred, Y_val)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = {k: v.copy() for k, v in model.params.items()}  # Deep copy best params
            save_model_params(best_params, os.path.join(results_dir, 'best_model_params.npz'))

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_accuracy'].append(val_accuracy)

        print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2%}, lr: {learning_rate:.4f}")

    model.params = best_params  # Load the best model
    return model, history





def main():
    X_train, Y_train, X_val, Y_val = load_fashion_mnist('./fashion-mnist', training=True, val_split=0.1)  
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_val, Y_val = preprocess_data(X_val, Y_val) # (num_features, num_samples)
    
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    
    hyperparams = {
        'hidden_size': 512,  # 输入层784个神经元，隐藏层512个神经元，输出层10个神经元
        'activation': 'relu',
        'initial_lr': 0.1, #0.01,
        'batch_size': 64,
        'lr_strategy': 'step',
        'l2_reg': 0.001,
        'epochs': 100,
        'decay_rate': 0.95,
        'epochs_drop': 10,
        'drop': 0.7,
        'milestones': [30, 60]
    }

    with open(os.path.join(results_dir, 'hyperparams.json'), 'w') as f:
        json.dump(hyperparams, f, indent=4)

    # 定义网络结构和训练参数
    model = NeuralNetwork([784, int(hyperparams['hidden_size']), 10], 
                          activation=hyperparams['activation']) 
    
    trained_model, history = train(model, X_train, Y_train, X_val, Y_val, 
                          epochs = hyperparams['epochs'], 
                          initial_lr = hyperparams['initial_lr'], 
                          batch_size = hyperparams['batch_size'], 
                          lr_strategy = hyperparams['lr_strategy'],
                          l2_reg = hyperparams['l2_reg'],
                          results_dir=results_dir,
                          decay_rate=hyperparams['decay_rate'],
                          epochs_drop=hyperparams['epochs_drop'],
                          drop=hyperparams['drop'],
                          milestones=hyperparams['milestones'])
    
    plot_history(history, results_dir)

    
if __name__ == "__main__":
    main()
