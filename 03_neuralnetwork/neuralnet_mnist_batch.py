import sys, os
import numpy as np
import pickle
sys.path.append(os.pardir) # 親ディレクトリを読み込む
from dataset.mnist import load_mnist  # datasetディレクトリのload_mnist関数を使う
from common.functions import sigmoid, softmax # commonディレクトリの(同上)

# MNISTデータを取得する
def get_data():
    # flatten: 入力画像を1次元に配列にする(Falseなら1x28x28の3次元配列)
    # normalize: 入力画像を0〜1の値に正規化する(Falseなら0〜255)
    # one_hot_label: 正解のラベルだけ1の配列にする(Falseなら2,7など正解の値が入る)
    (x_train, t_train), (x_test, t_test) = \
         load_mnist(flatten=True, normalize=True, one_hot_label=False)
    return x_test, t_test

# 学習済の重みパラメータ(sample_weight.pkl)を使ってネットワークを初期化する
# このファイルには重みとバイアスがディクショナリ型の変数として保存されている
def init_network():
    with open("sample_weight.pkl", "rb") as f:
        # pickle(実行中のオブジェクトをファイルとして保存する機能)を使う
        network = pickle.load(f) 

    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    
    return y

x, t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0 # 正解数
for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    # 配列yの中で最大の(最も確率の高い)インデックスを取得
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

# 正解率を精度として出力
print("Accuracy: " + str(float(accuracy_cnt) / len(x)))
