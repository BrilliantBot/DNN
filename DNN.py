class DNN():
    def __init__(self,layers):
        self.layers = layers
        self.weights = []
        for i in range(len(layers)-1):
            layers_weights = np.random.rand(layers[i+1],layers[i]+1)
            self.weights.append(layers_weights)

    def sigmoid(self,x):
        return 1/(1+np.exp(-.01*x))

    def predict(self,data):
        x_s = [data]

        for i in range(len(self.layers)-1):
          """add bias"""
          x_s[-1] = np.concatenate((x_s[-1],[1]))
          z = np.dot(self.weights[i],x_s[i])
          x_s.append(self.sigmoid(z))

        return x_s[-1]

    def train(self,data,y_true):
        x_s = [data]

        for i in range(len(self.layers)-1):
          """add bias"""
          x_s[-1] = np.concatenate((x_s[-1],[1]))
          z = np.dot(self.weights[i],x_s[i])
          x_s.append(self.sigmoid(z))

        psi = []
        for i in range(len(y_true)):
          output = x_s[-1][i]
          psi.append(-2*(y_true[i] - output) * (output * (1-output)))
        psi = np.array(psi)
        psi = np.reshape(psi,(psi.shape[0],1))

        gradients = []
        gradients.append(psi*x_s[-2])

        for i in range(len(self.layers) - 2, 0,-1):
            w = self.weights[i][:,:-1]
            x = x_s[i][:-1]
            term = w * x * (1-x)
            term = np.transpose(term)

            psi = np.dot(term, psi)
            psi = np.reshape(psi,(psi.shape[0],1))

            gradients.append(psi*x_s[i-1])

        for i in range(len(gradients)):
            self.weights[i] -= .1*gradients[-(i+1)]
        return sum((y_true-x_s[-1])**2)
