import numpy as np

EPS = 10**(-6)

class ThreeHiddenDNN():
    def __init__(self, step_size=0.05) -> None:
        # Initialize random weights
        self.step_size = step_size
        self.W1 = np.random.randn(3,3)*(1/np.sqrt(3))
        self.W2 = np.random.randn(4,2)*(1/np.sqrt(4))

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
        W_hat = self.W2[:-1]
        W_hat[:,0] = a_prime * W_hat[:,0]
        W_hat[:,1] = a_prime * W_hat[:,1]
        W_hat = (1./x.shape[0])*W_hat
        
        # Update values for weights[0]
        delta_1 = W_hat @ delta_2.T
        # Partial derivatives for updating weights[0]
        Q0 = np.outer(np.append(x, 1).T, delta_1)

        self.W1 -= self.step_size*Q0
        self.W2 -= self.step_size*Q1

    def softmax(self, y_hat):
        '''
            Softmax to perform classification
        '''
        y_hat_exp = np.exp(y_hat)
        y_hat_sum = np.sum(y_hat, axis=1)
        y_hat_sum = np.reshape(y_hat_sum, (y_hat_sum.shape[0],1))
        y_hat_softmax = y_hat_exp/y_hat_sum
        return y_hat_softmax

    def forward(self, x):
        ''' 
        Forward function for making classifications.
        Node values between input and hidden layer 
        We don't need to store z because we can use h
        for the backpropagation since we use ReLU activation function

        :param x:   The data to classify

        :returns:   The predicted labels
        '''
        z = np.c_[x, np.ones((x.shape[0],1))]@self.W1
        # Node values between hidden layer and output, ReLU
        self.h = np.maximum(z, 0)
        y_hat = np.c_[self.h, np.ones((x.shape[0],1))]@self.W2
        # Using subtraction of np.max to avoid overflow, Goodfellow: Deep Learning page 81
        max_y_hat = np.max(y_hat, axis=1, keepdims=True)
        new_y_hat = y_hat-max_y_hat
        y = self.softmax(new_y_hat)

        return y
