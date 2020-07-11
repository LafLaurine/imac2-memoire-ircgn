class Autoencoder:
    def __init__(self, D, d):
        # D is the input data (image dimension)
        # d is the encoding dimension
        # network has 3 layers of dimensions D,d,D
        # input placeholder, "None" here means any size
        self.X = tf.placeholder(tf.float32, shape=(None, D))
        
        # encoding phase
        # input to hidden (D -> d)
        # first layer of weights with bias
        # shape of weight is Dxd, going from the higher to lower dimension
        self.W1 = tf.Variable(tf.random_normal(shape=(D,d)))
        self.b1 = tf.Variable(np.zeros(d).astype(np.float32))
        
        # tensor for bottleneck layer (ReLU activation)
        self.Z = tf.nn.relu( tf.matmul(self.X, self.W1) + self.b1 )
        
        # decoding phase
        # hidden -> output (d -> D)
        # identical from encoding but we're goint from the lower to higher dimension
        self.W2 = tf.Variable(tf.random_normal(shape=(d,D)))
        self.b2 = tf.Variable(np.zeros(D).astype(np.float32))

        # output tensor
        logits = tf.matmul(self.Z, self.W2) + self.b2 
        # prediction variable (sigmoid activation because interval is [0,1])
        self.X_hat = tf.nn.sigmoid(logits)
        
        # define loss function : sigmoid cross entropy : binary classification at the pixel level
        self.loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(
                # expected result
                labels=self.X,
                logits=logits
            )
        )

        # optimizer       
        self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.005).minimize(self.loss)
        
        self.init_op = tf.global_variables_initializer()
        self.sess = tf.get_default_session() 
        if(self.sess == None):
            self.sess = tf.Session()
        self.sess.run(self.init_op)
        
        
    def fit(self, X, epochs=10, bs=64):
        n_batches = len(X) // bs
        print("Training {} batches".format(n_batches))
        
        for i in range(epochs):
            print("Epoch: ", i)
            X_perm = np.random.permutation(X)
            for j in range(n_batches):
                batch = X_perm[j*bs:(j+1)*bs]
                _, _ = self.sess.run((self.optimizer, self.loss),
                                    feed_dict={self.X: batch})


    def predict(self, X):
        return self.sess.run(self.X_hat, feed_dict={self.X: X})
        
    def encode(self, X):
        return self.sess.run(self.Z, feed_dict={self.X: X})
    
    def decode(self, Z):
        return self.sess.run(self.X_hat, feed_dict={self.Z: Z})
    
    def terminate(self):
        self.sess.close()
        del self.sess