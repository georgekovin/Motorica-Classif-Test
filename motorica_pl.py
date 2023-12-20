import numpy as np
import pandas as pd
import joblib as jl


class MotoricaPipeline:
    def __init__(self) -> None:
        pass
    
    def fit(self, X):
        self.pipeline = jl.load('motorica_pl.pkl')
        return self 
    
    def predict(self, X, as_df=False):
        y = np.split(self.pipeline.predict(X), X.shape[0])
    
        if as_df:
            y_inxcol = pd.Series(range(len(y)), name='sample_id')
            y_header = pd.Series(range(len(y[0])), name='timestep')

            y = pd.DataFrame(y)
            
            return y.set_index(y_inxcol).T.set_index(y_header).T
        
        return np.array(y) 
    
    def fit_predict(self, X, as_df=False):
        return self.fit(X).predict(X, as_df=as_df)


