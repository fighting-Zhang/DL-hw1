import numpy as np
from model import NeuralNetwork
from train import train
from dataset import preprocess_data, load_fashion_mnist
from test import evaluate
import os
from utils import plot_history
import json

def grid_search(train_data, val_data, test_data):
    initial_lrs = [0.001, 0.01, 0.1]
    
    lr_strategy = 'step' # 'step', 'exponential', 'multistep', 'cosine'
    epochs_drop = 10
    drops = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    # lr_strategy = 'multistep'
    milestones = [30, 60]
    # drop = 0.8

    # lr_strategy = 'exponential'
    decay_rate = 0.95

    # lr_strategy = 'cosine'
    
    hidden_sizes = [64, 128, 256, 512]
    l2_regs = [0., 0.0001, 0.001, 0.01, 0.1]

    activation = 'relu'

    epochs = 100
    batch_size = 64

    best_val_acc = 0
    best_params = None

    results = []

    for lr in initial_lrs:
        for drop in drops:
            for hidden_size in hidden_sizes:
                for l2 in l2_regs:


                    hyperparams = {
                        'hidden_size': hidden_size,
                        'activation': activation,
                        'initial_lr': lr,
                        'batch_size': batch_size,
                        'lr_strategy': lr_strategy,
                        'l2_reg': l2,
                        'epochs': epochs,
                        'decay_rate': decay_rate,
                        'epochs_drop': epochs_drop,
                        'drop': drop,
                        'milestones': milestones
                    }

                    results_dir = 'results' + f'_lr_{lr}_drop_{drop}_hidden_{hidden_size}_l2_{l2}'
                    os.makedirs(results_dir, exist_ok=True)
                    with open(os.path.join(results_dir, 'hyperparams.json'), 'w') as f:
                        json.dump(hyperparams, f, indent=4)

                    model = NeuralNetwork([784, hidden_size, 10], activation=activation)
                    
                    trained_model, history = train(model, 
                                            train_data[0], train_data[1], 
                                            val_data[0], val_data[1], 
                                            epochs = hyperparams['epochs'], 
                                            initial_lr = hyperparams['initial_lr'], 
                                            batch_size = hyperparams['batch_size'], 
                                            lr_strategy = hyperparams['lr_strategy'],
                                            l2_reg = hyperparams['l2_reg'],
                                            results_dir = results_dir,
                                            decay_rate = hyperparams['decay_rate'],
                                            epochs_drop = hyperparams['epochs_drop'],
                                            drop = hyperparams['drop'],
                                            milestones = hyperparams['milestones']
                                            )
                    
                    plot_history(history, results_dir)
                    
                    val_accuracy = evaluate(trained_model, test_data[0], test_data[1])
                    results.append((lr, drop, hidden_size, l2, val_accuracy))

                    if val_accuracy > best_val_acc:
                        best_val_acc = val_accuracy
                        best_params = (lr, drop, hidden_size, l2)

                    print(f"lr: {lr}, drop: {drop}, hidden_size: {hidden_size}, l2: {l2}, val_acc: {val_accuracy}")

    print(f"Best validation accuracy: {best_val_acc}")
    print(f"Best parameters: Learning rate: {best_params[0]}, Drop: {best_params[1]}, Hidden size: {best_params[2]}, L2 reg: {best_params[3]}")

    return results


def main():
    X_train, Y_train, X_val, Y_val = load_fashion_mnist('./fashion-mnist', training=True, val_split=0.1)  
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_val, Y_val = preprocess_data(X_val, Y_val)

    X_test, Y_test = load_fashion_mnist('./fashion-mnist', training=False)
    X_test, Y_test = preprocess_data(X_test, Y_test)

    train_data = (X_train, Y_train)
    val_data = (X_val, Y_val)
    test_data = (X_test, Y_test)

    results = grid_search(train_data, val_data, test_data)

    with open('grid_search_results.json', 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    main()
