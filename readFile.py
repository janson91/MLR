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




if __name__ == "__main__":
    data = []
    labels = []
    features = []
    with open('sample_feature_test_20170814') as f:
        print("-------read file-------")
        pos = 0
        for i, line in enumerate(f.readlines()):
            # if i > 10000: break
            ds = line.strip().split(" ")
            label = int(float(ds[0]))
            if label == 0:
                label = -1
            feature = {}

            if label == 1:
                pos +=1

            for fea in ds[1:]:
                key, value = fea.split(':')
                feature[int(key)] = float(value)
            if not(1 in feature and feature[1] == 0.0):
                labels.append(label)
                features.append(feature)
        print ("-----read file done-----")
    #

    # print (len(data))
    print ("data number :", len(labels))
    print ("pos number :", pos)
    # labels, features = list(zip(*data))
    # # maxfeaNum = 0
    # # for f in features:
    # #     maxfeaNum = max(maxfeaNum, max(f.keys()))
    # # print (maxfeaNum)
    # # print(features)
    # # print (labels)
    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2,random_state=0)
    # # pickle.dump((features_train, features_test, labels_train, labels_test), open("save\\split_data.pkl", "wb"))

    #用pretrain之后的数据：
    # features, labels = pickle.load(open("pretrain_data.pkl","rb"))
    # features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.2,random_state=0)
    # pickle.dump((features_train, features_test, labels_train, labels_test), open("save\\split_pretrain_data.pkl", "wb"))


    # features_train, features_test, labels_train, labels_test = pickle.load(open("save\\split_data.pkl","rb"))
    print (len(features_train))
    print (len(features_test))
    print (len(labels_train))
    print (len(labels_test))
    # for d in data[:100]:
    #     print (d)
    print ("-----pretrain file done-----")

    classNum = 2
    positive = np.sum((np.array(labels_train) + 1)/2)
    weight = (len(labels_train) - positive)/positive
    # _sgd = SGD(2,step = 0.001, norm_u_2=0.1, norm_w_2=0.01, iterNum=100,M=classNum)
    # ww, wu = _sgd.do_sgd(data)
    # 225621
    #57137
    # stamp = str(time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time())))
    # _ftrl = FTRL(225621, intercept=True, classNum=classNum, u_stdev=0.1, w_stdev=0.1, u_alpha=0.05, u_beta=1.0,
    #              norm1=0.0001, norm21=.005, w_alpha=0.05, w_beta=1.0, it=300, stamp = stamp, weight = weight, load=False, load_stamp = "20170818_172643", Mrate = 10)
    # ww, wu = _ftrl.do_ftrl(list(zip(labels_train, features_train)),list(zip(labels_test, features_test)))

    #
    _lsplm = LSPLM(feaNum=225621,
                   classNum=classNum,
                   norm21=0.01,
                   norm1=0.01,
                   tol=0.001,
                   iterNum=200,
                   intercept=True,
                   memoryNum=10,
                   beta=0.000001,
                   lamb=0.000001)
    ww, wu = _lsplm.train(list(zip(labels_train, features_train)),list(zip(labels_test, features_test)))



    print("------------- w --------------")
    for i in range(classNum):
        print(ww[i])
    print("------------- u --------------")
    for i in range(classNum):
        print(wu[i])

    print("============ result ===========")
    test_data = list(zip(labels_test, features_test))
    count = 0.0
    labels = []
    scores = []
    _tp,_tn,_p,_n = 0,0,0,0
    for i in range(len(labels_test)):
        d = test_data[i]
        result = fc.mlr(ww, wu, d)
        labels.append(d[0])
        scores.append(result)
        if d[0] == 1: _p += 1
        else: _n += 1

        if abs(result - (1 + d[0]) / 2) < 0.5:
            count += 1
            if result > 0.5:_tp += 1
            else: _tn += 1
    print ("tp = ",_tp," p = ", _p," tn = ",_tn," n = ", _n," tp/p = ",_tp/_p," tn/n = ", _tn/_n," acc: ",count / len(test_data))
    print(count / len(test_data))

    # 计算AUC
    fpr, tpr, thresholds = roc_curve(np.array(labels), np.array(scores), pos_label=1)
    roc_auc = roc_auc_score(labels, scores)

    # with open("save/result" + stamp, "a") as fw:
    #     fw.write("tp = %s p = %s tn = %s n = %s \n" % (_tp, _p, _tn, _n))
    #     fw.write("auc = %s" % roc_auc)

    #画ROC
    # plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

