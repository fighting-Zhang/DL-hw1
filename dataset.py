import numpy as np
import requests
import gzip
import os

def download_file(url, save_path):
    """下载文件并保存到指定路径"""
    response = requests.get(url, stream=True)
    response.raise_for_status()  # 确保请求成功
    with open(save_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

def load_fashion_mnist(data_dir, training=True, val_split=0.1):
    """下载并加载Fashion-MNIST数据集到指定目录"""
    base_url = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
    files = [
        "train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz"
    ]
    
    # 确保数据目录存在
    os.makedirs(data_dir, exist_ok=True)

    # 下载文件
    for file_name in files:
        file_path = os.path.join(data_dir, file_name)
        if not os.path.exists(file_path):  # 避免重复下载
            print(f"Downloading {file_name}...")
            download_file(base_url + file_name, file_path)
        else:
            print(f"{file_name} already exists.")

    # 加载数据
    def load_images(file_name):
        with gzip.open(os.path.join(data_dir, file_name), 'rb') as f:
            return np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)  #flatten

    def load_labels(file_name):
        with gzip.open(os.path.join(data_dir, file_name), 'rb') as f:
            return np.frombuffer(f.read(), np.uint8, offset=8)

    if training:
        x_train = load_images("train-images-idx3-ubyte.gz")
        y_train = load_labels("train-labels-idx1-ubyte.gz")
        # 随机划分验证集
        indices = np.arange(x_train.shape[0])
        np.random.shuffle(indices)
        n_train = int(x_train.shape[0] * (1 - val_split))
        train_indices = indices[:n_train]
        val_indices = indices[n_train:]

        x_val = x_train[val_indices]
        y_val = y_train[val_indices]
        x_train = x_train[train_indices]
        y_train = y_train[train_indices]

        return x_train, y_train, x_val, y_val
    else:
        x_test = load_images("t10k-images-idx3-ubyte.gz")
        y_test = load_labels("t10k-labels-idx1-ubyte.gz")
        return x_test, y_test


def preprocess_data(x, y):
    # Normalize the images
    x = x.astype(np.float32) / 255.0
    
    # One-hot encode labels
    y_one_hot = np.zeros((y.size, 10))
    y_one_hot[np.arange(y.size), y] = 1

    x = x.T
    y_one_hot = y_one_hot.T
    
    return x, y_one_hot



if __name__ == "__main__":
    data_dir = './fashion-mnist'
    x_train, y_train, x_val, y_val = load_fashion_mnist(data_dir, training=True, val_split=0.1)
    print("Data loaded successfully!")

    x_train, y_train = preprocess_data(x_train, y_train)
    x_val, y_val = preprocess_data(x_val, y_val)