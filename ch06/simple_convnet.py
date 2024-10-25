# coding: utf-8          
# Simple CNN 클래스 선언
# =======간단한 합성곱 신경망=======
# conv -> relu -> pool -> affine -> relu -> affine -> softmax
#################################### visualizing filters.ipynb파일에서 에러난거 임포트 해결
import sys
import os
import numpy as np
import pickle

from collections import OrderedDict
from common.layers import Convolution, Relu, Pooling, Affine, SoftmaxWithLoss  # 올바른 경로로 임포트
sys.path.append(r'C:\projects\jupyterProjects\DeepLearning-MS-AI\DL3_20241006\common')

####################################

class SimpleConvNet:
    """ 
    Parameters:
    input_size : 입력 크기(MNIST의 경우 784)
    hidden_size_list : 은닉층의 뉴런 수 리스트 (e.g. [100, 100, 100])
    output_size : 출력 크기(분류 대상 수)
    activation : 활성화 함수 - 'relu' 또는 'sigmoid'
    weight_init_std : 가중치 초기화 시 정규분포 표준편차 (e.g. 0.01)
        'relu'를 사용할 때는 'He' 초깃값 추천
        'sigmoid'를 사용할 때는 'Xavier' 초깃값 추천
    """
    
    def __init__(self, input_dim=(1, 28, 28), 
                 conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1}, 
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size / 2) * (conv_output_size / 2))
        
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * \
            np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * \
            np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * \
            np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)
        
        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'], self.params['b3'])
        
        self.last_layer = SoftmaxWithLoss()
    
    # 순전파 함수
    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    # 손실 값을 구하는 함수
    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)
    
    # 정확도를 계산하는 함수
    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)
        
        return acc / x.shape[0]
    
    # 수치미분 방식으로 기울기를 구하는 함수
    def numerical_gradient(self, x, t):
        loss_w = lambda w: self.loss(x, t)
        
        grads = {}
        grads['W1'], grads['b1'] = numerical_gradient(loss_w, self.params['W1']), numerical_gradient(loss_w, self.params['b1'])
        grads['W2'], grads['b2'] = numerical_gradient(loss_w, self.params['W2']), numerical_gradient(loss_w, self.params['b2'])
        grads['W3'], grads['b3'] = numerical_gradient(loss_w, self.params['W3']), numerical_gradient(loss_w, self.params['b3'])
        
        return grads
    
    # 오차역전파 방식으로 기울기를 구하는 함수
    def gradient(self, x, t):
        # forward
        self.loss(x, t)
        
        # backward
        dout = 1
        dout = self.last_layer.backward(dout)
        
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        
        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        
        return grads
    
    # 학습된 가중치를 저장하는 함수
    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)
    
    # 저장된 가중치를 로딩하는 함수
    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val
        
        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]
