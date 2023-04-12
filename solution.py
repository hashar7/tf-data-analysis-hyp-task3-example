import pandas as pd
import numpy as np
from scipy.stats import permutation_test

chat_id = 458704720 # Ваш chat ID, не меняйте название переменной

def solution(x: np.array, y: np.array) -> bool: 
    alpha = 0.07
    result = permutation_test((x, y), lambda x, y, axis: np.mean(x, axis=axis) - np.mean(y, axis=axis), 
                 vectorized=True, 
                 n_resamples=5000,
                 alternative='greater').pvalue < alpha
    return result 
