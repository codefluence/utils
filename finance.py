import numpy as np

class NeutralizeTransform:

    '''
    Not my code. Picked from https://www.kaggle.com/snippsy/jane-street-densenet-neutralizing-features
                             https://www.kaggle.com/snippsy

    Feature neutralization is a technique to keep as much infomration as possible of a vector, simultaneously reducing linear-dependent information on another vector.
    Essentially we take the residuals of a vector by linear regression with another vector as a feature.

    How is it useful?
    It is useful because by applying the feature neutralization to the features on the target, we can get a set of features that contain as much original information as possible but decorrelate with the target.
    Imagine a situation where one feature is very correlated with the target (which is often the case with ML in finance). When you train a model, the model jumps to the feature and ignores else.
    This is fine as long as you are validating your model with historical data. You may get a very high utility score.
    However, when you deploy the model for forecasting, there may be a situation in the future where that very strong feature becomes useless. Like pandemic, war, political affairs, you name it. What if that happens?
    Maybe the sign of your model prediction gets flipped...sell when you should buy, buy when you should sell.
    Apparently this is a catastrophic scenario for an investor...but this is often the consequence when you use an overfitting model just because no feature is perfect in a financial market.
    You might want to have a model where it is not dependent on single features!
    Feature neutralization helps you to have that kind of model.

    https://www.kaggle.com/code1110/janestreet-avoid-overfit-feature-neutralization?scriptVersionId=53012639
    https://docs.numer.ai/numerai-signals/signals-overview#neutralization
    https://www.kaggle.com/c/jane-street-market-prediction/discussion/215305

    #how to use:
    #neutralizer = NeutralizeTransform(proportion=0.25).fit(features, targets)
    #neutral = neutralizer.transform(features)
    '''



    def __init__(self,proportion=1.0):
        self.proportion = proportion
    
    def fit(self,X,y):

        self.lms = []
        self.mean_exposure = np.mean(y,axis=0)
        self.y_shape = y.shape[-1]

        for x in X.T:
            scores = x.reshape((-1,1))
            exposures = y
            exposures = np.hstack((exposures, np.array([np.mean(scores)] * len(exposures)).reshape(-1, 1)))
            
            transform = np.linalg.lstsq(exposures, scores, rcond=None)[0]
            self.lms.append(transform)

        return self
            
    def transform(self,X):

        out = []
        
        for i,transform in enumerate(self.lms):
            x = X[:,i]
            scores = x.reshape((-1,1))
            exposures = np.repeat(self.mean_exposure,len(x),axis=0).reshape((-1,self.y_shape))
            exposures = np.concatenate([exposures,np.array([np.mean(scores)] * len(exposures)).reshape((-1,1))],axis=1)
            correction = self.proportion * exposures.dot(transform)
            out.append(x - correction.ravel())
            
        return np.asarray(out).T
    
    def fit_transform(self,X,y):

        self.fit(X,y)
        return self.transform(X,y)