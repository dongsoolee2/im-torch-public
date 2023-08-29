"""
Various functions for imaging data analysis using PyTorch
"""

import numpy as np
import h5py

__all__ = ['rolling_window', 'rolling_window_pair']

def rolling_window(array, window):
    # input: (stimulus) array (T, X1, X2)
    #         window size (int)
    # output: rolling_window (stimulus) array (T - (window - 1), window, X1, X2)
    # With stimulus (T, X1, X2) and response (T, ),
    # output stimulus (T - (window - 1),) time should be matched by response
    T, X1, X2 = array.shape
    return np.squeeze(np.lib.stride_tricks.sliding_window_view(array, (window, X1, X2)))

def rolling_window_pair(array_stim, array_resp, window):
    # input: array_stim (T, X1, X2)
    #        array_resp (T, N?)
    #        window size (int)
    # output: rolling_window array_stim (T - window, window, X1, X2)
    #         corresponding array_resp (T - window, N)
    # With stimulus (T, X1, X2) and response (T, ),
    # output stimulus T[(window - 1):-1] time is matched by response[(window):,], size = (T - window)
    return np.array(rolling_window(array_stim, window)[:-1, :, :, :]), np.array(array_resp[window:, :])

def h5file_loader(filepath, p):
    # input: h5 filepath to read 
    #        json object (config.json)
    # output: X_train, y_train, X_test, y_test
    idx_repeat = 450
    idx_norepeat = 2750
    buffer = 150
    with h5py.File(filepath, 'r') as f:
        X_train_temp = np.double(f[p['train_stim_key']])  # (T(tr), X1, X2)
        y_train_temp = np.double(f[p['train_resp_key']])  # (N, T(tr)
        X_test_temp = np.double(f[p['test_stim_key']])    # (T(te), X1, X2) or np.nan
        y_test_temp = np.double(f[p['test_resp_key']])    # (trial, N, T(te)) or np.nan
    test_repeated = 0 if len(X_test_temp.shape) == 0 else 1
    if test_repeated:
        X_train = X_train_temp[idx_repeat:, :, :]
        y_train = y_train_temp[:, idx_repeat:]
        X_test = X_test_temp[:, :, :]
        y_test = np.nanmean(y_test_temp[:, :, :], 0)      # trial average (N, T(te))
    else:
        X_train = X_train_temp[idx_norepeat:, :, :]
        y_train = y_train_temp[:, idx_norepeat:]
        X_test = X_train_temp[idx_repeat:idx_norepeat - buffer, :, :]
        y_test = y_train_temp[:, idx_repeat:idx_norepeat - buffer]
    return X_train, y_train, X_test, y_test