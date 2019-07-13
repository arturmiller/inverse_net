from sklearn import datasets
import numpy as np
import matplotlib.pylab as plt
from sklearn.preprocessing import OneHotEncoder


class InverseNet:
    def __init__(self):
        pass

    def fit(self, X, y): 
        self.model_params = np.zeros((10, X.shape[1]))

        for k in range(10):
            grad = np.empty_like(self.model_params)

            for i in range(self.model_params.shape[0]):
                for j in range(self.model_params.shape[1]):
                    h = 0.001
                    pos_model_params = np.copy(self.model_params)
                    pos_model_params[i, j] += h
                    neg_model_params = np.copy(self.model_params)
                    neg_model_params[i, j] -= h

                    grad[i, j] = (self.T(X, y, pos_model_params) - self.T(X, y, neg_model_params)) / 2.0 / h

            beta = 100.0
            self.model_params -= beta * grad
            print(str(k) + ' ,err: ' + str(self.T(X, y, self.model_params)))

    def T(self, X, y, model_params):
        return np.sum((y - self.predict(X, model_params=model_params))**2) / y.shape[0]


    def predict(self, X, model_params=None):
        if model_params is None:
            model_params = self.model_params
        alpha = 5e-6
        cur_causal_factors = np.zeros((X.shape[0], 10))
        for _ in range(100):
            dL = -2 * np.einsum('ki,ji->kj', (X - self.forward(cur_causal_factors, model_params)), model_params)
            cur_causal_factors = cur_causal_factors - alpha * dL
        return cur_causal_factors

    def forward(self, causal_factors, model_params):
        simulation = np.einsum('ki,ij->kj', causal_factors, model_params)
        return simulation


if __name__ == '__main__':
    (images, target) = datasets.load_digits(10, return_X_y=True)
    inverse_net = InverseNet()

    enc = OneHotEncoder()
    enc.fit(target.reshape(-1, 1))
    y = enc.transform(target.reshape(-1, 1)).toarray()

    inverse_net.fit(images[:100], y[:100])

    print('error: ' + str(inverse_net.T(images, y, inverse_net.model_params)))
    y_pred = inverse_net.predict(images[:10].reshape((10, -1)))
    print('class: ' + str(np.argmax(y_pred, 1)))
    for i in range(10):
        X_pred = inverse_net.forward(np.reshape(y_pred[i, :], (1, -1)), inverse_net.model_params)
        plt.imshow(inverse_net.model_params[i].reshape(8,8))
        plt.figure()
        plt.imshow(X_pred.reshape(8, 8))
        plt.figure()
        plt.imshow(images[i].reshape(8, 8))
        plt.show()

# 0 ,err: 0.8936954711511097
# 1 ,err: 0.8498081152211131
# 2 ,err: 0.820228803651684
# 3 ,err: 0.7950465992790031
# 4 ,err: 0.7720252159730479
# 5 ,err: 0.7505637924441816
# 6 ,err: 0.730447244926543
# 7 ,err: 0.7115584200249419
# 8 ,err: 0.69380744388594
# 9 ,err: 0.677114121959552
# error: 0.7550790055235298