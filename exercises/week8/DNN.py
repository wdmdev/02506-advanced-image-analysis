import numpy as np

EPS = np.finfo(float).eps

class ThreeHiddenDNN():
    def __init__(self, step_size=0.05) -> None:
        # Initialize random weights
        self.step_size = step_size
        self.weights = np.asarray([
            np.random.randn(3,3),
            np.random.randn(4,2)
        ])


    def backpropagation(self, x, y_pred, y_true, i):
        '''
        Backpropagation function for updating network weights
        according to predictions and actual target labels.

        :param x:           The data to classify
        :param y_pred:      The predicted labels
        :param y_true:      The true labels of the given X data
        :param i:           Current row in X data
        '''

        # Update values for weights[1]
        delta_2 = y_pred-y_true
        h = self.h[i]
        # Partial derivatives for updating weights[1]
        Q1 = np.outer(np.append(h,1).T, delta_2)

        # Derivatives of activation
        a_prime = (h > 0).astype(int).T
        # Weights without bias row (last row)
        W_hat = self.weights[1][:-1]
        W_hat[:,0] = a_prime * W_hat[:,0]
        W_hat[:,1] = a_prime * W_hat[:,1]
        
        # Update values for weights[0]
        delta_1 = W_hat @ delta_2.T
        # Partial derivatives for updating weights[0]
        Q0 = np.outer(np.append(x, 1).T, delta_1)

        self.weights[0] -= self.step_size*Q0 + EPS
        self.weights[1] -= self.step_size*Q1 + EPS

    def forward(self, x):
        ''' 
        Forward function for making classifications.
        Node values between input and hidden layer 
        We don't need to store z because we can use h
        for the backpropagation since we use ReLU activation function

        :param x:   The data to classify

        :returns:   The predicted labels
        '''
        z = np.c_[x, np.ones((x.shape[0],1))]@self.weights[0]
        # Node values between hidden layer and output
        self.h = np.maximum(z, 0)
        y_hat = np.c_[self.h, np.ones((self.h.shape[0],1))]@self.weights[1]
        # Softmax to perform classification
        y = np.exp(y_hat)/np.sum(np.exp(y_hat), axis=1, keepdims=True)

        return y
