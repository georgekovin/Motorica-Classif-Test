import numpy as np
import pandas as pd
import pickle
from sklearn.base import BaseEstimator



class MotoricaPipeline(BaseEstimator):
    
    def __init__(self) -> None:
        """Пайплайн для предсказания жеста человека по датчикам на его руке."""
        
        with open('./motorica_pl.pkl', 'rb') as pl_file:
            self.pipeline = pickle.load(pl_file)
        
        self.data = np.load('./data/X_train.npy')
        self.target = np.load('./data/y_train.npy')
        
            
    def _check_params(self, arr, name, dim):
        """Технический метод для проверки корректности формата данных.

        Args:
            arr (`ArrayLike`): данные в виде массива
            name (`str`): название переменной
            dim (`int`): требуемая размерность

        Raises:
            `ValueError`: массив `arr` должен иметь требуемую размерность `dim`.
        """
        arr_dim = len(arr.shape)
        
        if arr_dim != dim:
            raise ValueError(f'{name} must be {str(dim)}d array, got {arr_dim}d')
    
    
    def _update_data(self, X, y):
        """Технический метод для обновления данных.

        Args:
            X (`ArrayLike`): новые данные
            y (`ArrayLike`): новая целевая переменная

        Returns:
            `tuple`: кортеж из обновленных данных
        """
        X_new = np.concatenate([self.data, X], axis=0)
        y_reshaped = np.append(self.target, y.flatten(), axis=0)
        
        return X_new, y_reshaped
    
    
    def fit(self, X, y):
        """Обучение пайплайна.

        Args:
        ---
            X (`ArrayLike`): трехмерный массив с размерностью (`наблюдения`, `датчики`, `время`) для обучения
            y (`ArrayLike`): целевая переменная в формате (`наблюдения`, `время`)

        Returns:
        ---
            `self`: обученный пайплайн
        """
        self._check_params(X, 'X', 3)     
        self._check_params(y, 'y', 2)
        
        X_new, y_new = self._update_data(X, y)
        self.pipeline.fit(X_new, y_new)
        
        return self 
    
    
    def predict(self, X, as_df=False):
        """Предсказание жестов по датчикам на руке.

        Args:
        ---
            X (`ArrayLike`): трехмерный массив с размерностью (`наблюдения`, `датчики`, `время`)
            as_df (`bool`, optional): при `True` метод возвращает результат в формате DataFrame, по умолчанию - `False`

        Returns:
        ---
            `DataFrame` | `NDArray`: двумерный массив с закодированными жестами с размерностью (`наблюдения`, `время`)
        """
        self._check_params(X, 'X', 3)
        
        y = np.split(self.pipeline.predict(X), X.shape[0])
    
        if as_df:
            y_inxcol = pd.Series(range(len(y)), name='sample_id')
            y_header = pd.Series(range(len(y[0])), name='timestep')

            y = pd.DataFrame(y)
            
            return y.set_index(y_inxcol).T.set_index(y_header).T
        
        return np.array(y) 
    
    
    def fit_predict(self, X, y=None, as_df=False):
        """Обучение пайплайна и затем предсказание жестов.

        Args:
        ---
            X (`ArrayLike`): трехмерный массив с размерностью (`наблюдения`, `датчики`, `время`) для обучения
            y (`ArrayLike`, optional): целевая переменная в формате (`наблюдения`, `время`), не обязательна
            as_df (bool, optional): при `True` метод возвращает результат в формате DataFrame, по умолчанию - `False`

        Returns:
        ---
            `DataFrame` | `NDArray`: двумерный массив с закодированными жестами с размерностью (`наблюдения`, `время`)
        """
        
        return self.fit(X, y).predict(X, as_df=as_df)
    
    
    def update_pipeline_and_data(self, X, y):
        """Обновление пайплайна, если он обучался на новых данных.

        Args:
        ---
            X (`ArrayLike`): данные, которые нужно добавить к старым
            y (`ArrayLike`): новая целевая переменная, которую нужно соединить со старой
        """
        self._check_params(X, 'X', 3)     
        self._check_params(y, 'y', 2)
        
        pl_updated = self.pipeline.fit(X, y)
        data_updated = self._update_data(X, y)
        
        with open('./motorica_pl.pkl', 'wb') as pl_file:
            pickle.dump(pl_updated, pl_file)
        
        with open('./data/X_train.npy', 'wb') as X_file:
            np.save(X_file, data_updated[0])

        with open('./data/y_train.npy', 'wb') as y_file:
            np.save(y_file, data_updated[1])
    
    
    def get_pipeline_step(self, step):
        """Получение конкретного шага из пайплайна.

        Args:
        ---
            step (`str`): название шага

        Raises:
        ---
            `ValueError`: пайплайн не имеет шага, котроый вы указали

        Returns:
        ---
            `Any`: шаг пайплайна
        """
        steps = ['reshaper', 'scaler', 'model']
        
        if step not in steps:
            raise ValueError(f'Pipeline have no such step - {step}.')
        
        return self.pipeline.named_steps[step]