import numpy as np
import pandas as pd
import joblib as jl


pipeline = jl.load('model/sgd_pipeline.pkl')


class MotoricaPipeline:
    def __init__(self) -> None:
        self.pipeline = pipeline
    
    def fit(self, X):
        return self 
    
    def predict(self, X, as_df=False):
        y = np.split(self.pipeline.predict(X), X.shape[0])
    
        if as_df:
            y_inxcol = pd.Series(range(len(y)), name='sample_id')
            y_header = pd.Series(range(len(y[0])), name='timestep')

            y = pd.DataFrame(y)
            
            return y.set_index(y_inxcol).T.set_index(y_header).T
        
        return np.array(y) 
    
    