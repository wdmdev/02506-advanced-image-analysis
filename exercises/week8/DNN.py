import numpy as np

EPS = 10**(-6)

class ThreeHiddenDNN():
    def __init__(self, step_size=0.05) -> None:
        # Initialize random weights
        self.step_size = step_size
        self.W1 = np.random.randn(3,3)*(1/np.sqrt(3))
        self.W2 = np.random.randn(4,2)*(1/np.sqrt(4))

    def backpropagation(self, X, Y, T):
        '''
        Backpropagation function for updating network weights
        according to predictions and actual target labels.

        :param X:      The data to classify
        :param Y:      The predicted labels
        :param T:      The true labels of the given X data
        '''

        # Update values for weights[1]
        delta_2 = Y-T
        # Update values for weights[1]
        dW2 = np.outer(np.c_[self.h, np.ones((self.h.shape[0],1))].T, delta_2)
        dW2 = np.sum(dW2.diagonal().reshape((X.shape[0], -1)), axis=0)
        # Derivatives of activation
        a_prime = (self.h > 0).astype(int).T

        # Weights without bias row (last row)
        W2_hat = self.W2[:-1]
        
        delta_1 = a_prime * (W2_hat @ delta_2.T)
        # Update values for weights[0]
        dW1 = np.outer(np.c_[X, np.ones((X.shape[0],1))].T, delta_1)
        dW1 = np.sum(dW1.diagonal().reshape((X.shape[0], -1)), axis=0)

        self.W1 -= self.step_size*dW1
        self.W2 -= self.step_size*dW2

    def softmax(self, y_hat):
        y_hat_exp = np.exp(y_hat)
        return y_hat_exp/np.sum(y_hat_exp, axis=1, keepdims=True)

    def forward(self, X):
        ''' 
        Forward function for making classifications.
        Node values between input and hidden layer 
        We don't need to store z because we can use h
        for the backpropagation since we use ReLU activation function

        :param X:   The data to classify

        :returns:   The predicted labels
        '''
        z = np.c_[X, np.ones((X.shape[0],1))]@self.W1
        # Node values between hidden layer and output, ReLU
        self.h = np.maximum(z, 0)
        y_hat = np.c_[self.h, np.ones((X.shape[0],1))]@self.W2
        # Using subtraction of np.max to avoid overflow, Goodfellow: Deep Learning page 81
        max_y_hat = np.max(y_hat, axis=1, keepdims=True)
        new_y_hat = y_hat-max_y_hat
        y = self.softmax(new_y_hat)

        return y

    def cross_entropy(y,y_pre):
        loss=-np.sum(y*np.log(y_pre))
        return loss/float(y_pre.shape[0])
