import function as fc
from sgd import SGD
from ls_plm import LSPLM
from ftrl import FTRL
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve,roc_auc_score
import  matplotlib.pyplot as plt
import pickle
import time

if __name__ == '__main__':

    # name = "parameters" + str(time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time())))
    # # name = "parameters" + str(time.localtime(time.time())) + "txt"
    # print (name)
    # with open("save/"+ "parameters20170816_152824",'a') as f:
    #     f.write(str(time.time()))
    _tp,_p,_tn,_n = 1,2,3,4
    print ("tp = %s p = %s tn = %s n = %s" % (_tp, _p, _tn, _n))