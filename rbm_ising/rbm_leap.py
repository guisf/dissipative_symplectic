import numpy as np
import progressbar
import pickle


bar_widgets = ['Training: ', progressbar.Percentage(), 
    ' ', progressbar.Bar(marker="-", left="[", right="]"), 
    ' ', progressbar.ETA()
]

class Sigmoid():
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.__call__(x) * (1 - self.__call__(x))

def batch_iterator(X, y=None, batch_size=64):
    """ Simple batch generator """
    n_samples = X.shape[0]
    for i in np.arange(0, n_samples, batch_size):
        begin, end = i, min(i+batch_size, n_samples)
        if y is not None:
            yield X[begin:end], y[begin:end]
        else:
            yield X[begin:end]

sigmoid = Sigmoid()

class RBM():
    
    def __init__(self, n_hidden=128, learning_rate=0.1, mu=0.9,
                    batch_size=10, n_iterations=100):
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.lr = learning_rate
        self.mu = mu
        self.cosh = np.cosh(-np.log(mu))
        self.n_hidden = n_hidden
        self.progressbar = progressbar.ProgressBar(widgets=bar_widgets)

    def _initialize_weights(self, X):
        n_visible = X.shape[1]
        self.W = np.random.normal(scale=0.1, size=(n_visible, self.n_hidden))
        #self.v0 = np.zeros(n_visible)       # Bias visible
        #self.v0 = np.ones(n_visible)       # Bias visible
        #self.v0 = np.random.choice([0,1], n_visible)       # Bias visible
        self.v0 = np.random.normal(loc=0.5, scale=0.5, size=n_visible)
        #self.h0 = np.zeros(self.n_hidden)   # Bias hidden
        #self.h0 = np.ones(self.n_hidden)   # Bias hidden
        #self.h0 = np.random.choice([0,1], self.n_hidden)   # Bias hidden
        self.h0 = np.random.normal(loc=0.5, scale=0.5, size=self.n_hidden)

        # momentum variables
        self.pW = np.zeros((n_visible, self.n_hidden))
        self.pv0 = np.zeros(n_visible)
        self.ph0 = np.zeros(self.n_hidden)

    def fit(self, X, y=None):
        """Contrastive Divergence training procedure with k=1."""

        self._initialize_weights(X)

        self.training_errors = []
        self.training_reconstructions = []
        
        for _ in self.progressbar(range(self.n_iterations)):
            
            batch_errors = []
            for batch in batch_iterator(X, batch_size=self.batch_size):

                # Positive phase
                positive_hidden = sigmoid(batch.dot(self.W) + self.h0)
                hidden_states = self._sample(positive_hidden)
                positive_associations = batch.T.dot(positive_hidden)
                # Negative phase
                negative_visible = sigmoid(hidden_states.dot(self.W.T)+self.v0)
                negative_visible = self._sample(negative_visible)
                negative_hidden = sigmoid(negative_visible.dot(self.W)+self.h0)
                negative_associations = negative_visible.T.dot(negative_hidden)
                # compute gradients
                g1 = -(positive_associations-negative_associations)
                g2 = -(positive_hidden.sum(axis=0)-negative_hidden.sum(axis=0))
                g3 = -(batch.sum(axis=0)-negative_visible.sum(axis=0))

                self.pW = self.mu*(self.mu*self.pW - (self.lr)*g1)
                self.ph0 = self.mu*(self.mu*self.ph0 - (self.lr)*g2)
                self.pv0 = self.mu*(self.mu*self.pv0 - (self.lr)*g3)

                self.W = self.W + self.lr*self.cosh*self.pW
                self.h0 = self.h0 + self.lr*self.cosh*self.ph0
                self.v0 = self.v0 + self.lr*self.cosh*self.pv0

                batch_errors.append(np.mean((batch-negative_visible)**2))

            self.training_errors.append(np.mean(batch_errors))
            # Reconstruct a batch of images from the training set
            idx = np.random.choice(range(X.shape[0]), self.batch_size)
            self.training_reconstructions.append(self.reconstruct(X[idx]))

    def _sample(self, X):
        return X > np.random.random_sample(size=X.shape)

    def reconstruct(self, X):
        positive_hidden = sigmoid(X.dot(self.W) + self.h0)
        hidden_states = self._sample(positive_hidden)
        negative_visible = sigmoid(hidden_states.dot(self.W.T)+self.v0)
        return negative_visible


if __name__ == '__main__':
    
    import config

    temps = np.linspace(config.tmin, config.tmax, config.nt)
    train_temps = [temps[i] for i in range(0, len(temps), config.lf_skip)]
    N = config.N
    for i, T in enumerate(train_temps):
       
        print("step %i/%i ..."%(i+1, len(train_temps)))
        data = pickle.load(open(config.monte_carlo_output%(T, N), 'rb'))

        # vectorize, binarize, shuffle
        Z = np.array([d.reshape(N*N) for d in data])
        Z[Z==-1] = 0
        idx = np.random.choice(range(Z.shape[0]), config.lf_num_train_points,
                               replace=False)
        X = Z[idx]

        # training RBM
        rbm = RBM(n_hidden=config.lf_n_hidden,
                  batch_size=config.lf_batch_size,
                  learning_rate=config.lf_learning_rate,
                  mu=config.lf_mu,
                  n_iterations=config.lf_iterations)
        rbm.fit(X)
        pickle.dump(rbm.training_errors,
                    open(config.lf_conv_format%(T,N), 'wb'))

        # getting samples from RBM and making into an Ising state
        #idx = np.random.choice(range(Z.shape[0]), config.lf_num_train_points)
        Xp = rbm.reconstruct(X)
        Xp = np.rint(Xp)
        Xp[Xp==0.] = -1.
        states = [x.reshape((N,N)) for x in Xp]
        pickle.dump(states, open(config.lf_file_format%(T, N), 'wb'))
        
