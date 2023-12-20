import numpy as np
import pandas as pd
import pickle
from sklearn.base import BaseEstimator


class MotoricaPipeline(BaseEstimator):
    def __init__(self) -> None:
        """Пайплайн для предсказания жеста человека по датчикам на его руке."""
        
        with open('./motorica_pl.pkl', 'rb') as pl_file:
            self.pipeline = pickle.load(pl_file)
    
    def fit(self, X):
        """Обучение пайплайна 
        (на самом деле он уже обученный, данный метод - просто формальность)

        Args:
        ---
            X (`ArrayLike`): трехмерный массив с размерностью (`наблюдения`, `датчики`, `время`)

        Returns:
        ---
            `self`: обученный пайплайн
        """
        return self 
    
    def predict(self, X, as_df=False):
        """Предсказание жестов по датчикам на руке.

        Args:
        ---
            X (`ArrayLike`): трехмерный массив с размерностью (`наблюдения`, `датчики`, `время`)
            as_df (bool, optional): при `True` метод возвращает результат в формате DataFrame, по умолчанию - `False`

        Returns:
        ---
            `DataFrame` | `NDArray`: двумерный массив с закодированными жестами с размерностью (`наблюдения`, `время`)
        """
        
        y = np.split(self.pipeline.predict(X), X.shape[0])
    
        if as_df:
            y_inxcol = pd.Series(range(len(y)), name='sample_id')
            y_header = pd.Series(range(len(y[0])), name='timestep')

            y = pd.DataFrame(y)
            
            return y.set_index(y_inxcol).T.set_index(y_header).T
        
        return np.array(y) 
    
    def fit_predict(self, X, as_df=False):
        """Обучение пайплайна и затем предсказание жестов.

        Args:
        ---
            X (`ArrayLike`): трехмерный массив с размерностью (`наблюдения`, `датчики`, `время`)
            as_df (bool, optional): при `True` метод возвращает результат в формате DataFrame, по умолчанию - `False`

        Returns:
        ---
            `DataFrame` | `NDArray`: двумерный массив с закодированными жестами с размерностью (`наблюдения`, `время`)
        """
        
        return self.fit(X).predict(X, as_df=as_df)
