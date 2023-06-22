import numpy as np 
class Perceptron:
    """
    params
    -----
    eta: float 
    learning rate (between 0.0 and 1.0) 
    n_iter: int
    passes ofver the training dataset (epochs)
    
    random_state: int 
    random number generator seed for random weight 
    initialization
    
    Attributes
    ----------
    w_: 1d-array
    weights after fitting
    
    b_: scalar
    bias unit after fitting.add()
    
    errors_: list
    number of misclassification (updates) in each epoch
    
    """
    
    
    
    def _init_(self,eta=0.01, n_iter=50, random_state=1):
        self.eta=eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit(self,x,y):
        """
        fit training data.
        Parameters
        ========
        x: (array-like), shape [n_examples, n_features]
        Training vectors , where nexamples is the number of examples and n_features is the number of features. 
        y: (array like), shape [n examples] target values
        returns 
        ------
        self: object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_=rgen.normal(loc=0.0,scale=0.01,size=x.shape[1])
        self.b_=np.float(0.)
        self.errors_=[]
        
        for _ in range (self.n_iter):
            errors =0
            for xi,target in zip(x,y):
                update=self.eta * (target-self.predict(xi))
                self.w_ += update * xi
                self.b_ +=update
                errors += int(update !=0.0)
            self.errors_.append(errors)
        return self
    
    def net_input(self,x):
        """calculate net input """
        return np.dot(x,self.w_)+ self.b_
    def predict(self,x):
        """return class label after unit step"""
        return np.where(self.net_input(x) >= 0.0,1,0)