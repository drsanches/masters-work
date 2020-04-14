import numpy as np
import json


__DATASET_PATH = '../dataset/'

def get_dataset(filename):
    dataset = []
    with open(__DATASET_PATH + filename) as f:
        dataset = json.load(f)

    X = np.array(dataset[0])
    Y = np.array(dataset[1])
    return X, Y


def convert_Y_to_class_numbers(Y):
    tmp = []
    for y in Y:
        tmp.append(convert_y_to_class_number(y))
    return tmp


def convert_y_to_class_number(y):
    return np.argmax(y);


def get_feature_names():
    return [
    'AVG_TIME_BETWEEN_TRANS',
    'AVG_TIME_BETWEEN_SENT_TRANS',
    'AVG_TIME_BETWEEN_RECEIVED_TRANS',
    'DEVIATION_TIME_BETWEEN_TRANS',
    'DEVIATION_TIME_BETWEEN_SENT_TRANS',
    'DEVIATION_TIME_BETWEEN_RECEIVED_TRANS',
    'AVG_TRANS_ETH',
    'AVG_ETH_SENT',
    'AVG_ETH_RECEIVED',
    'DEVIATION_TRANS_ETH',
    'DEVIATION_ETH_SENT',
    'DEVIATION_ETH_RECEIVED',
    'PERCENT_OF_SMART_CONTRACT_TRANS',
    'PERCENT_OF_TRANS_RECEIVED_FROM_SMART_CONTRACTS',
    'PERCENT_OF_TRANS_SENT_TO_SMART_CONTRACTS',
    'PERCENT_OF_SMART_CONTRACT_ETH',
    'PERCENT_OF_ETH_RECEIVED_FROM_SMART_CONTRACTS',
    'PERCENT_OF_ETH_SENT_TO_SMART_CONTRACTS'
    ]


def remove_feature(X, featureIndex):
    res = [];
    for x in X:
        tmp = x
        tmp = np.delete(tmp, featureIndex)
        res.append(tmp)        
    return np.array(res)