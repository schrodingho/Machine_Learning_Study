import numpy as np

class SimpleLinearRegression:
    def __init__(self):
        self.a=None
        self.b=None

    def fit(self,x_train,y_train):

        assert x_train.ndim==1
        assert len(x_train)==len(y_train)

        x_mean=np.mean(x_train)
        y_mean=np.mean(y_train)

        num=0.0
        dom