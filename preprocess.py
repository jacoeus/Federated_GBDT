import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.utils.data

TRAINDATA_DIR = './train/train/'
TESTDATA_PATH = './test/testing-X.pkl'
ATTACK_TYPES = {
    'snmp': 0,
    'portmap': 1,
    'syn': 2,
    'dns': 3,
    'ssdp': 4,
    'webddos': 5,
    'mssql': 6,
    'tftp': 7,
    'ntp': 8,
    'udplag': 9,
    'ldap': 10,
    'netbios': 11,
    'udp': 12,
    'benign': 13,
}


class CompDataset(object):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

        self._data = [(x, y) for x, y in zip(X, Y)]

    def __getitem__(self, idx):
        return self._data[idx]

    def __len__(self):
        return len(self._data)


def extract_features(data, has_label=True):

    data['SimillarHTTP'] = 0.
    if has_label:
        return data.iloc[:, -80:-1]

    return data.iloc[:, -79:]

def get_user_data(user_idx):
    fpath = ""
    for root, dirs, fnames in os.walk(TRAINDATA_DIR):
        #select_files = [a for a in fnames if a.find('one')==-1]
        fname = fnames[user_idx]
        fpath = os.path.join(root, fname)
        break

    if not fpath.endswith('csv'):
        return

    print('Load User {} Data: '.format(user_idx), os.path.basename(fpath))
    data = pd.read_csv(fpath, skipinitialspace=True, low_memory=False)
    x = extract_features(data)
    x = x.iloc[:49500, :]
    y = np.array([
        ATTACK_TYPES[t.split('_')[-1].replace('-', '').lower()]
        for t in data.iloc[:, -1]
    ])
    y = y[:49500]
    x = x.to_numpy().astype(np.float32)
    return (
        x,
        y,
    )

def get_test_data():
    with open(TESTDATA_PATH, 'rb') as fin:
        data = pickle.load(fin)

    return data['X']
