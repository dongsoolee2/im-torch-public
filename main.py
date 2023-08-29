# import modules
import os
import sys
import numpy as np
import scipy.stats
import h5py
import json
import time

import torch
import torch.nn as nn
from torch.nn import MSELoss, PoissonNLLLoss
import torch.optim as optim
import torch.linalg as LA
import torch.utils.data

import utils
import models
from models import *

# project directory
if sys.platform == 'darwin':
    home_dir = '/Users'
    MAC0_OR_LINUX1 = 0
else:
    home_dir = '/home'
    MAC0_OR_LINUX1 = 1
proj_dir = home_dir + '/dlee/im-torch/'

# load config.json
with open(proj_dir + 'config.json', 'r') as js:
    p = json.load(js)
p['_start_time'] = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
p['_MAC0_OR_LINUX1'] = MAC0_OR_LINUX1
p['_proj_dir'] = proj_dir

# device 
if MAC0_OR_LINUX1:
    num_gpu = p['num_gpu']
    DEVICE = torch.device('cuda:' + str(num_gpu) if torch.cuda.is_available() else 'cpu')
else:
    DEVICE = torch.device('mps' if torch.has_mps else 'cpu')
p['_torch_DEVICE'] = str(DEVICE)

# load data
# X_train/X_test (T, X1, X2), y_train/y_test (n_roi, T)
h5file_dir = proj_dir + 'data-extract/' if MAC0_OR_LINUX1 else '/Users/dlee/imaging/data-extract/'
h5file_name = p['h5file']
X_train, y_train, X_test, y_test = utils.h5file_loader(h5file_dir + h5file_name, p)
with h5py.File(h5file_dir + h5file_name, 'r') as f:
    te_cc_val_rank = np.double(f[p['test_cc_key']])
    p['_te_cc_val'] = list(te_cc_val_rank[:, 0])
    p['_tbins_tr_first_10'] = list(np.double(f['train/tbins'])[:10])
    p['_tbins_te_first_10'] = list(np.double(f['test/tbins'])[:10])
p['_h5file_dir+name'] = h5file_dir + h5file_name

# total number of roi
n_roi = y_train.shape[0]
p['_n_roi'] = n_roi

# hyperparameters
num_frame_window = p['num_frame_window']
learning_rate = p['learning_rate']
batch_size = p['batch_size']
num_epochs = p['num_epochs']
num_cells = n_roi if p['num_cells'] == 'all' else p['num_cells']
select_mode = p['select_mode']
p['_num_cells'] = num_cells

# selection
if p['select_mode'] == 'large_cc':
    with h5py.File(h5file_dir + h5file_name, 'r') as f:
        te_cc_val_rank = np.double(f[p['test_cc_key']])
    te_cc = te_cc_val_rank[:, 0]
    te_cc_largeidx = np.argsort(te_cc)[::-1]
    idx_select = te_cc_largeidx[:num_cells]
elif p['select_mode'] == 'random':                  # randomly select
    np.random.seed(0)
    idx_arr = np.arange(n_roi)
    np.random.shuffle(idx_arr)
    idx_select = idx_arr[:num_cells]
elif p['select_mode'] == 'manual':                  # manually select
    idx_select = p['select_manual'][:num_cells]
else:                                               # [0, 1, 2, ...]
    idx_select = np.arange(n_roi)[:num_cells]
p['_idx_select'] = idx_select.tolist()

# rolling window
# X_train/test_rw (T - window, X1, X2), y_train/test_rw (T - window, N)
X_train_rw, y_train_rw = utils.rolling_window_pair(X_train, y_train[idx_select, :].T, window=num_frame_window)
X_test_rw, y_test_rw = utils.rolling_window_pair(X_test, y_test[idx_select, :].T, window=num_frame_window) 

# create torch.Tensor 
X_tr = torch.Tensor(X_train_rw)
y_tr = torch.Tensor(y_train_rw)
X_te = torch.Tensor(X_test_rw)
y_te = torch.Tensor(y_test_rw)
print(X_tr.shape, y_tr.shape, X_te.shape, y_te.shape)
p['_X_tr_shape'] = str(X_tr.shape)
p['_y_tr_shape'] = str(y_tr.shape)
p['_X_te_shape'] = str(X_te.shape)
p['_y_te_shape'] = str(y_te.shape)

# dataset and dataloader
# train:
dataset_tr = torch.utils.data.TensorDataset(X_tr, y_tr)
dataloader_tr = torch.utils.data.DataLoader(dataset=dataset_tr, batch_size=batch_size, shuffle=True)
# test:
#   dataset_te = torch.utils.data.TensorDataset(X_te, y_te)
#   dataloader_te = torch.utils.data.DataLoader(dataset=dataset_te, batch_size=100, shuffle=False)

# model
model = vars()[p['model']](out=num_cells).to(DEVICE)
p['_model'] = str(model)

# loss function, optimizer, and scheduler
loss_fn = vars()[p['loss_fn']]
if p['loss_fn'] == 'PoissonNLLLoss':
    loss_fn = loss_fn(log_input=False)
else:
    loss_fn = loss_fn()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-2)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2)
p['_loss_fn'] = str(loss_fn)
p['_optimizer'] = str(optimizer)
p['_scheduler'] = str(scheduler)

# create directory if not exist
save_dir = proj_dir + 'archive/' + h5file_name[:-3] + '_' + p['model'] + '_' + p['_start_time'] + '/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# train
def train(model):
    num_batches = len(dataloader_tr)
    loss_tr_list = []
    cc_tr_list = []
    cc_te_list = []
    best_accuracy = -1
    
    for epoch in range(num_epochs):

        # train mode
        model.train(mode=True)
        for batch, (movies, response) in enumerate(dataloader_tr):
           
            # to device
            movies = movies.to(DEVICE)       
            response = response.to(DEVICE)

            # forward pass
            optimizer.zero_grad()
            pred = model(movies)    # (T, n_roi, 1, 1)
            
            # loss & regularization
            loss_main = loss_fn(torch.squeeze(pred), response)
            #activity_l1 = 1e-3 * torch.mean(LA.vector_norm(torch.squeeze(pred), dim=0, ord=1))

            # loss backward and optimize
            loss = loss_main #+ activity_l1 
            if batch == 0:
                loss_tr_list.append(loss.item())
            loss.backward()
            optimizer.step()

            # print
            if ((batch + 1) % 5 == 0) or ((batch + 1) == num_batches):
                print('Epoch [{}/{}], Iteration [{}/{}], Loss: {:.9f}'
                      .format(epoch+1, num_epochs, batch+1, num_batches, loss.item()))

        # eval mode
        model.eval()
        with torch.no_grad():
            
            # detach train & forward pass (validation)
            pred_tr = torch.squeeze(pred).cpu().detach().numpy()
            resp_tr = response.cpu().detach().numpy()

            pred_te = torch.squeeze(model(X_te.to(DEVICE))).cpu().detach().numpy()
            resp_te = y_te.detach().numpy()

            # calculate pearson correlation
            cc_tr = np.stack([scipy.stats.pearsonr(pred_tr[:, i], resp_tr[:, i])[0] for i in range(num_cells)])
            cc_te = np.stack([scipy.stats.pearsonr(pred_te[:, i], resp_te[:, i])[0] for i in range(num_cells)])
            cc_tr_list.append(cc_tr)
            cc_te_list.append(cc_te)

            # print results
            print("Train correlation coefficient is: {}".format(cc_tr))
            print("Train correlation coefficient (average) is: {}".format(np.nanmean(cc_tr)))
            print("Validation/Test correlation coefficient is: {}".format(cc_te))
            print("Validation/Test correlation coefficient (average) is: {}".format(np.nanmean(cc_te)))

            # validation/test accuracy
            accuracy_te = np.nanmean(cc_te)

            # update best accuracy and save model
            if best_accuracy < accuracy_te:
                # update best accuracy
                best_accuracy = accuracy_te

                # save model
                ptfile_name = h5file_name[:-3] + '_' + 'E' + format(epoch, '03d') + '_' + format(accuracy_te, '.3f') + '_' + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()) + '.pt'
                torch.save(model.state_dict(), save_dir + ptfile_name)

    # save other parameters
    p['_save_dir'] = save_dir
    p['_num_batches'] = num_batches
    p['_loss_tr_list'] = loss_tr_list
    p['_cc_tr_list'] = np.stack(cc_tr_list).tolist()
    p['_cc_te_list'] = np.stack(cc_te_list).tolist()
    p['_best_accuracy'] = best_accuracy
    p['_ptfile_name'] = ptfile_name
    p['_end_time'] = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())

    # save new json file
    newjsonfile_name = 'config_' + h5file_name[:-3] + '_' + p['model'] + '_' + p['_start_time'] + '.json'
    with open(save_dir + newjsonfile_name, 'w') as js:
        json.dump(p, js, indent=4, sort_keys=True)

def main():
    train(model)

if __name__ == "__main__":
    main()