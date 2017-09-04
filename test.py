
import function as fc
from sgd import SGD
from ls_plm import LSPLM
from ftrl import FTRL
import random
import  matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve,roc_auc_score
import time
import copy

def calculateY(X, w):
    Y = []
    for x in X:
        y = -(w[2] + x * w[0]) / w[1]
        Y.append(y)
    return Y




if __name__ == "__main__":
    # data = [
    #         (1, {1: 1.0, 2:1.0, 3:1.0, 5:1.0, 8:1.0}),
    #         (1, {1: 1.0, 3: 1.0, 4:1.0, 7: 1.0}),
    #         (-1, {2:1.0, 5:1.0, 9:1.0}),
    #         (-1, {0:1.0, 3:1.0, 4:1.0, 5:1.0}),
    #         (-1, {0:1.0, 2:1.0, 3:1.0, 6:1.0, 7:1.0}),
    #         (-1, {3:1.0, 6:1.0}),
    #         (-1, {0:1.0, 2:1.0, 7:1.0}),
    #         (-1, {6:1.0, 7:1.0}),
    #         (1, {1:1.0, 2:1.0, 3:1.0, 4:1.0, 5:1.0, 8:1.0}),
    #         (1, {1:1.0, 3:1.0, 5:1.0, 6:1.0}),
    #         (-1, {0:1.0, 2:1.0, 9:1.0}),
    #         (-1, {0:1.0, 4:1.0, 7:1.0})
    #     ]

    #generate test data
    # data = [(1, {0:x+0.1, 1:y+0.1}) if abs(x-y) < 2 else (-1, {0:x+0.1, 1:y+0.1}) for x in range(-2,3) for y in range(-2,3)]
    # _data = np.transpose(np.array([(y[0], y[1], x) for x,y in data]))
    # plt.scatter(_data[0], _data[1], s = (_data[2]+2)*30, c = (_data[2]+1)/2, linewidths=0, alpha=0.7)
    # plt.show()


    data = [(1, { 0:4*random.random() - 2, 1:4*random.random() - 2}) for _ in range(10000)]
    data = [(-1, f) if abs(f[0])+abs(f[1]) < 2 else (1, f) for _, f in data]
    _data = np.transpose(np.array([(y[0], y[1], x) for x,y in data]))
    # plt.scatter(_data[0], _data[1], s = (_data[2]+2)*5, c = (_data[2]+1)/2, linewidths=0, alpha=0.7)
    # plt.show()

    # print (data)
    #2,step = 0.1, norm_u_2=0.1, norm_w_2=0.01, iterNum=200,M=2
    #2,step = 0.001, norm_u_2=0.1, norm_w_2=0.01, iterNum=400,M=2

    classNum = 4
    # _sgd = SGD(2,step = 0.001, norm_u_2=0.1, norm_w_2=0.01, iterNum=100,M=classNum)
    # ww, wu = _sgd.do_sgd(data)
    stamp = time.time()
    # _ftrl = FTRL(2, intercept=True, classNum=classNum, u_stdev = 0.1, w_stdev = 0.1, u_alpha = 0.05, u_beta = 1.0,
    #              norm1 = 0.1, norm21 = 5.0, w_alpha = 0.05, w_beta = 1.0, it = 20, save = False, load=False,stamp = stamp)
    test_data = copy.deepcopy(data)
    # ww, wu = _ftrl.do_ftrl(data,test_data)

    _lsplm = LSPLM(feaNum = 2,
                 classNum = classNum,
                 norm21 = 0.01,
                 norm1 = 0.01,
                 tol = 0.001,
                 iterNum = 100,
                 intercept = True,
                 memoryNum = 10,
                 beta = 0.0001,
                 lamb = 0.001)
    ww, wu = _lsplm.train(data,test_data)

    print ("------------- w --------------")
    for i in range(classNum):
        print (ww[i])
    print ("------------- u --------------")
    for i in range(classNum):
        print(wu[i])

    print ("============ result ===========")
    count = 0.0
    labels = []
    scores = []
    for i in range(len(data)):
        d = data[i]
        result = fc.mlr(ww, wu, d)
        labels.append(d[0])
        scores.append(result)
        if abs(result - (1+d[0])/2) < 0.5:
            count += 1
    print (count / len(data))

    # 画散点图和分类器
    _data = np.transpose(np.array([(y[0], y[1], x) for x,y in data]))
    plt.scatter(_data[0], _data[1], s = (_data[2]+2)*2, c = (_data[2]+1)/2, linewidths=0, alpha=0.7)
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    for w in ww:
        plt.plot([-2,0,2], calculateY([-2,0,2],w))

    # #画ROC
    # fpr, tpr, thresholds = roc_curve(np.array(labels), np.array(scores), pos_label=1)
    # roc_auc = roc_auc_score(labels, scores)
    # # plt.figure()
    # lw = 2
    # plt.plot(fpr, tpr, color='darkorange',
    #          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.0])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    # plt.legend(loc="lower right")
    plt.show()

# acc = [0.7491, 0.836, 0.8901, 0.9336, 0.9462, 0.9523, 0.9604, 0.9588, 0.9643, 0.9656, 0.9671, 0.9697, 0.9711, 0.9714, 0.9736, 0.9741, 0.9748, 0.9745, 0.9745, 0.9764, 0.9769, 0.975, 0.9751, 0.9771, 0.9772, 0.9785, 0.9788, 0.9799, 0.9798, 0.9791, 0.98, 0.981, 0.9805, 0.9816, 0.9804, 0.9815, 0.9821, 0.9818, 0.9828, 0.9833, 0.9831, 0.9828, 0.9832, 0.9832, 0.9832, 0.9833, 0.9835, 0.9833, 0.9843, 0.9841, 0.9842, 0.9843, 0.9847, 0.9849, 0.9841, 0.9843, 0.9847, 0.9853, 0.9855, 0.9859, 0.9863, 0.9858, 0.9862, 0.9864, 0.9865, 0.9867, 0.9866, 0.9865, 0.9869, 0.9869, 0.9869, 0.987, 0.9867, 0.9871, 0.987, 0.9871, 0.9872, 0.9871, 0.9874, 0.9874, 0.9875, 0.9876, 0.9874, 0.9875, 0.9875, 0.9876, 0.9876, 0.9877, 0.9877, 0.9877, 0.9877, 0.9878, 0.9878, 0.988, 0.9882, 0.9881, 0.9882, 0.9882, 0.9882, 0.9883]


