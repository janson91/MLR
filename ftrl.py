import function as fc
import numpy as np
import random
import  matplotlib.pyplot as plt
import pickle
import time
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve,roc_auc_score

class FTRL:
    def __init__(self,
                 feaNum,
                 intercept = True,
                 classNum = 4,
                 u_stdev = 0.1,
                 w_stdev = 0.1,
                 u_alpha = 0.05,
                 u_beta = 1.0,
                 norm1 = 0.1,
                 norm21 = 5.0,
                 w_alpha = 0.05,
                 w_beta = 1.0,
                 it = 30,
                 save = True,
                 load = False,
                 stamp = "",
                 load_stamp = "",
                 weight = 1,
                 Mrate = 100
                 ):
        """
            参数设置参考了https://github.com/CastellanZhang/alphaPLM
        -classNum: 分片数。	default:4
        -intercept: 是否加入bias
        -u_stdev: u的初始化使用均值为0的高斯分布，u_stdev为标准差。	default:0.1
        -w_stdev: w的初始化使用均值为0的高斯分布，w_stdev为标准差。	default:0.1
        -u_alpha: u的FTRL超参数alpha。	default:0.05
        -u_beta: u的FTRL超参数beta。	default:1.0
        -norm1: w,u的L1正则。	default:0.1
        -norm21: w,u的L2正则。	default:5.0
        -w_alpha: w的FTRL超参数alpha。	default:0.05
        -w_beta: w的FTRL超参数beta。	default:1.0
        -feaNum: 特征维度
        -it: 迭代轮数
        -save: 是否保存模型
        -load： 是否从文件中提取模型
        -stamp：文件时间戳
        """
        self.intercept = intercept
        self.classNum = classNum
        self.u_stdev = u_stdev
        self.w_stdev = w_stdev
        self.u_alpha = u_alpha
        self.u_beta = u_beta
        self.norm1 = norm1
        self.norm21 = norm21
        self.w_alpha = w_alpha
        self.w_beta = w_beta
        self.feaNum = feaNum + 1 if self.intercept else feaNum
        self.it = it
        self.save = save
        self.load = load
        self.stamp = stamp
        self.load_stamp = load_stamp
        self.weight = weight
        self.Mrate = Mrate

    def do_ftrl(self,data, test_data):
        self.saveParameters()
        if self.intercept:
            data = list(map(self.addBias, data))
            test_data = list(map(self.addBias, test_data))

        #储存每一代的ACC和LOSS
        ACC = []
        LOSS = []
        TEST_ACC = []
        TEST_LOSS = []
        AUC = []
        TEST_AUC = []

        ZW = [{} for _ in range(self.classNum)]
        ZU = [{} for _ in range(self.classNum)]
        NW = [{} for _ in range(self.classNum)]
        NU = [{} for _ in range(self.classNum)]
        WW = [{} for _ in range(self.classNum)]
        for w in WW:
            for i in range(self.feaNum):
                w[i] = np.random.normal(0,self.w_stdev)
        WU = [{} for _ in range(self.classNum)]
        for u in WU:
            for i in range(self.feaNum):
                u[i] = np.random.normal(0,self.u_stdev)
        if self.load:
            (WW, WU) = pickle.load(open("save/weight"+self.load_stamp+".kpl" ,"rb"))

        it = 0
        acc = []

        pos_data = []
        neg_data = []
        for item in data:
            if item[0]<=0:
                neg_data.append(item)
            else:
                pos_data.append(item)
        M = len(pos_data)/self.Mrate

        while it < self.it:
            start_time = time.time()


            shuffle_index = list(range(len(neg_data)))
            random.shuffle(shuffle_index)

            shuffle_neg_data = []
            for sii in range(len(pos_data)):
                shuffle_neg_data.append(neg_data[shuffle_index[sii]])

            data = pos_data + shuffle_neg_data

            for label, feaDic in data:
                # label, feaDic = data[si]

                #计算梯度：
                GW, GU = fc.cal_derivative(WW, WU, (label, feaDic))

                #更新Z,N
                for i in range(self.classNum):
                    for index in feaDic:
                        zw = ZW[i].get(index, 0)
                        nw = NW[i].get(index, 0)
                        zu = ZU[i].get(index, 0)
                        nu = NU[i].get(index, 0)
                        gw = GW[i].get(index, 0)/M
                        gu = GU[i].get(index, 0)/M
                        ww = WW[i].get(index, 0)
                        wu = WU[i].get(index, 0)

                        sw = ((nw + gw ** 2) ** 0.5 - nw ** 0.5) / self.w_alpha
                        su = ((nu + gu ** 2) ** 0.5 - nu ** 0.5) / self.u_alpha

                        ZW[i][index] = zw + gw - sw * ww
                        ZU[i][index] = zu + gu - su * wu

                        NW[i][index] = nw + gw ** 2
                        NU[i][index] = nu + gu ** 2

                #更新w,u
                for i in range(self.classNum):
                    for index in feaDic:
                        #计算W
                        z = ZW[i].get(index, 0)
                        n = NW[i].get(index, 0)
                        if abs(z) < self.norm1:
                            WW[i][index] = 0
                        else:
                            WW[i][index] = -(z - fc.sign(z) * self.norm1) / \
                                           (self.norm21 + (self.w_beta + n ** 0.5) / self.w_alpha)
                        #计算U

                        z = ZU[i].get(index, 0)
                        n = NU[i].get(index, 0)
                        if abs(z) < self.norm1:
                            WU[i][index] = 0
                        else:
                            WU[i][index] = -(z - fc.sign(z) * self.norm1) / \
                                           (self.norm21 + (self.u_beta + n ** 0.5) / self.u_alpha)
            print("iteration: ", it)
            it += 1
            if self.save :
                pickle.dump((WW,WU), open("save/weight"+self.stamp+".kpl", "wb"))

            # 计算train 相关参数：
            loss = fc.calLoss(data, WW, WU, self.norm21, self.norm1, self.feaNum)

            count = 0.0
            labels = []
            scores = []
            _tp, _tn, _p, _n = 0, 0, 0, 0
            for i in range(len(data)):
                d = data[i]
                result = fc.mlr(WW,WU, d)
                labels.append(d[0])
                scores.append(result)
                if d[0] == 1:
                    _p += 1
                else:
                    _n += 1

                if abs(result - (1 + d[0]) / 2) < 0.5:
                    count += 1
                    if result > 0.5:
                        _tp += 1
                    else:
                        _tn += 1
            acc = count / len(data)
            print ("train:")
            print("tp = ", _tp, " p = ", _p, " tn = ", _tn, " n = ", _n, " tp/p = ", _tp / _p, " tn/n = ", _tn / _n)
            ACC.append(str(acc))
            LOSS.append(str(loss))
            # 计算AUC
            roc_auc = roc_auc_score(labels, scores)
            AUC.append(str(roc_auc))

            if (it) % 10 == 0:
                #计算test相关量
                count = 0.0
                labels = []
                scores = []
                _tp, _tn, _p, _n = 0, 0, 0, 0
                for i in range(len(test_data)):
                    d = test_data[i]
                    result = fc.mlr(WW,WU, d)
                    labels.append(d[0])
                    scores.append(result)
                    if d[0] == 1:
                        _p += 1
                    else:
                        _n += 1

                    if abs(result - (1 + d[0]) / 2) < 0.5:
                        count += 1
                        if result > 0.5:
                            _tp += 1
                        else:
                            _tn += 1
                test_loss = fc.calLoss(test_data, WW, WU, self.norm21, self.norm1, self.feaNum)
                test_acc = count / len(test_data)
                print ("test:")
                print("tp = ", _tp, " p = ", _p, " tn = ", _tn, " n = ", _n, " tp/p = ", _tp / _p, " tn/n = ", _tn / _n,)

                TEST_ACC.append(str(test_acc))
                TEST_LOSS.append(str(test_loss))
                # 计算AUC
                # fpr, tpr, thresholds = roc_curve(np.array(labels), np.array(scores), pos_label=1)
                test_roc_auc = roc_auc_score(labels, scores)
                TEST_AUC.append(str(test_roc_auc))
                print("loss ", loss, " acc ", acc," auc ", roc_auc, " test loss ", test_loss, " test acc ", test_acc, " test auc ", test_roc_auc)
            print("use time: ", time.time() - start_time)
            print ("------------------------------------------------------\n")

        with open("save/result"+self.stamp,"a") as fw:
            fw.write("train_acc:" + " ".join(ACC)+"\n")
            fw.write("train_loss:" + " ".join(LOSS) + "\n")
            fw.write("train_auc:" + " ".join(AUC) + "\n")
            fw.write("test_acc:" + " ".join(TEST_ACC) + "\n")
            fw.write("test_loss:" + " ".join(TEST_LOSS) + "\n")
            fw.write("test_auc:" + " ".join(TEST_AUC) + "\n")

            # acc.append(count / len(data))
        #     if it%30 == 0:
        #
        #         _data = np.transpose(np.array([(y[0], y[1], x) for x, y in data]))
        #         plt.scatter(_data[0], _data[1], s=(_data[2] + 2) * 2, c=(_data[2] + 1) / 2, linewidths=0, alpha=0.7)
        #         plt.xlim(-3, 3)
        #         plt.ylim(-3, 3)
        #         for w in WW:
        #             plt.plot([-2, 0, 2], calculateY([-2, 0, 2], w))
        #         plt.savefig('%s'%it)
        #         # plt.show()
        #         plt.close(0)
        # print (acc)
        return WW, WU

    def addBias(self, item):
        """
        add bias
        :param item:
        :param fN:
        :return:
        """
        label, featureDic = item
        featureDic[self.feaNum - 1] = 1.0
        return (label, featureDic)

    def saveParameters(self):
        with open("save/para"+self.stamp, "w") as fp:
            for k,v in vars(self).items():
                fp.write(k + " " + str(v) + "\n")


def calculateY(X, w):
    Y = []
    for x in X:
        y = -(w[2] + x * w[0]) / w[1]
        Y.append(y)
    return Y