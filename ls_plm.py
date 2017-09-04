import numpy as np
import function as fc
import copy
import pickle
import time
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve,roc_auc_score
import random

class LSPLM:

    def __init__(self,
                 feaNum,
                 classNum,
                 iterNum = 100,
                 intercept = True,
                 memoryNum = 10,
                 beta = 0.1,
                 lamb = 0.1,
                 u_stdev=0.1,
                 w_stdev=0.1,
                 ):
        """
        :param feaNum:  特征数
        :param classNum:    类别数
        :param iterNum:
        :param intercept:
        :param memoryNum:
        :param beta:
        :param lamb:
        :param u_stdev:
        :param w_stdev:
        """
        self.classNum = classNum
        self.iterNum = iterNum
        self.intercept = intercept
        self.memoryNum = memoryNum
        self.feaNum = feaNum + 1 if self.intercept else feaNum
        self.lossList = []
        self.beta = beta
        self.lamb = lamb
        self.aucList = []
        self.w_stdev = w_stdev
        self.u_stdev = u_stdev

    def train(self, data,test_data):
        """
            训练ls-plm large scale piece-wise leanir model
        :param data:
        :return:
        """
        ACC = []
        LOSS = []
        TEST_ACC = []
        TEST_LOSS = []
        AUC = []
        TEST_AUC = []

        # np.random.seed(0)
        if self.intercept:
            data = list(map(self.addBias, data))
            test_data = list(map(self.addBias, test_data))
        it = 0
        WW = [{} for _ in range(self.classNum)]
        for w in WW:
            for i in range(self.feaNum):
                w[i] = np.random.normal(0,self.w_stdev)
        WU = [{} for _ in range(self.classNum)]
        for u in WU:
            for i in range(self.feaNum):
                u[i] = np.random.normal(0,self.u_stdev)

        gradient_W = [{} for _ in range(self.classNum)]
        gradient_U = [{} for _ in range(self.classNum)]
        loss = 0.0
        sList = [[] for _ in range(self.classNum * 2)]
        roList = [[] for _ in range(self.classNum * 2)]
        yList = [[] for _ in range(self.classNum * 2)]
        alphaList = [[0] * self.memoryNum for _ in range(self.classNum * 2)]
        # #初始化计算 一阶梯度
        # gradient_W, gradient_U = fc.cal_derivative(data, weight_W, weight_U, self.norm21, self.norm1, self.feaNum)
        # 计算loss 和 auc
        loss = fc.calLoss(data, WW, WU, self.lamb, self.beta, self.feaNum)

        # print("loss: %s" % loss)
        # print("gradient_w: is")
        # for w in weight_W:
        #     print (w)
        # print("gradient_u: is")
        # for u in weight_U:
        #     print(u)
        # self.firstLoss = loss
        # self.lossList.append(loss)

        pos_data = []
        neg_data = []
        for item in data:
            if item[0] <= 0:
                neg_data.append(item)
            else:
                pos_data.append(item)
        # M = len(pos_data) / self.Mrate


        while it < self.iterNum:
            print("============iterator : %s ==========" % it)
            start_time = time.time()

            if len(neg_data)>len(pos_data):
                print("pos_data + shuffle_neg_data")
                shuffle_index = list(range(len(neg_data)))
                random.shuffle(shuffle_index)

                shuffle_neg_data = []
                for sii in range(len(pos_data)):
                    shuffle_neg_data.append(neg_data[shuffle_index[sii]])

                data = pos_data + shuffle_neg_data



            # 1. 计算虚梯度
                #计算梯度
            LW, LU = fc.sumCalDerivative(WW, WU, data, weight = 1)
            vGW, vGU = fc.virtualGradient(self.feaNum, WW, WU, LW, LU,self.beta, self.lamb)

            # 2. 保存虚梯度方向，用于后续确定搜索方向是否跨象限
            vG = vGW + vGU
            dir = copy.deepcopy(vG)
            # dirW = copy.deepcopy(vGW)
            # dirU = copy.deepcopy(vGU)

            # 3. 利用LBFGS算法的两个循环计算下降方向, 这里会直接修改vGradient, 并确定下降方向是否跨象限
            fc.lbfgs(self.feaNum, vG, sList, roList, yList, alphaList,dir)

            # # 4. 确定下降方向是否跨象限， 这里也会直接修改vGradient
            # fc.fixDirection(vG, dir)

            # 5. 线性搜索最优解
            newLoss, newW = fc.backTrackingLineSearch(self.feaNum, it, loss, data, WW+WU, LW+LU, vG, dir,self.lamb, self.beta)

            # 打印结果
            newWW = newW[:len(newW)//2]
            newWU = newW[len(newW)//2:]
            # if self.save :
            #     pickle.dump((WW,WU), open("save/weight"+self.stamp+".kpl", "wb"))

            # 计算train 相关参数：
            loss = fc.calLoss(data, newWW, newWU, self.lamb, self.beta, self.feaNum)

            count = 0.0
            labels = []
            scores = []
            _tp, _tn, _p, _n = 0, 0, 0, 0
            for i in range(len(data)):
                d = data[i]
                result = fc.mlr(newWW, newWU, d)
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
            print("train:")
            print("tp = ", _tp, " p = ", _p, " tn = ", _tn, " n = ", _n, " tp/p = ", _tp / _p, " tn/n = ", _tn / _n)
            ACC.append(str(acc))
            LOSS.append(str(loss))
            # 计算AUC
            roc_auc = roc_auc_score(labels, scores)
            AUC.append(str(roc_auc))

            # 计算test相关量
            count = 0.0
            labels = []
            scores = []
            _tp, _tn, _p, _n = 0, 0, 0, 0
            for i in range(len(test_data)):
                d = test_data[i]
                result = fc.mlr(newWW, newWU, d)
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
            test_loss = fc.calLoss(data, newWW, newWU, self.lamb, self.beta, self.feaNum)
            test_acc = count / len(test_data)
            print("test:")
            print("tp = ", _tp, " p = ", _p, " tn = ", _tn, " n = ", _n, " tp/p = ", _tp / _p, " tn/n = ", _tn / _n, )

            TEST_ACC.append(str(test_acc))
            TEST_LOSS.append(str(test_loss))
            # 计算AUC
            # fpr, tpr, thresholds = roc_curve(np.array(labels), np.array(scores), pos_label=1)
            test_roc_auc = roc_auc_score(labels, scores)
            TEST_AUC.append(str(test_roc_auc))
            print("loss ", loss, " acc ", acc, " auc ", roc_auc, " test loss ", test_loss, " test acc ", test_acc,
                  " test auc ", test_roc_auc)


                # 6. 判断是否提前终止
            if self.check(it, test_roc_auc):
                break
            else:
                # 7. 更新各种参数
                self.shift(data, sList, yList, roList, WW + WU, newW, LW, LU)
                WW = newWW
                WU = newWU
            print("loss: %s" % loss)
            print("============iterator : %s end ==========" % it)
            print("")
            it += 1

            print("use time: ", time.time() - start_time)
            print("------------------------------------------------------\n")



        # with open("save/result"+self.stamp,"a") as fw:
        #     fw.write("train_acc:" + " ".join(ACC)+"\n")
        #     fw.write("train_loss:" + " ".join(LOSS) + "\n")
        #     fw.write("train_auc:" + " ".join(AUC) + "\n")
        #     fw.write("test_acc:" + " ".join(TEST_ACC) + "\n")
        #     fw.write("test_loss:" + " ".join(TEST_LOSS) + "\n")
        #     fw.write("test_auc:" + " ".join(TEST_AUC) + "\n")
        return WW, WU

    def shift(self,data, sList, yList, roList, W, newW, LW, LU):
        newLW, newLU = fc.sumCalDerivative(newW[:len(newW)//2],newW[len(newW)//2:],data)
        newGradient = newLW+newLU
        gradient = LW + LU
        for i in range(len(sList)):
            slist = sList[i]
            ylist = yList[i]
            rolist = roList[i]
            w = W[i]
            neww = newW[i]
            g = gradient[i]
            newg = newGradient[i]

            size = len(slist)
            if size == self.memoryNum:
                # print >> sys.stdout, "pop 老的S, Y, RO"
                slist.pop(0)
                ylist.pop(0)
                rolist.pop(0)

            nextS = {}
            nextY = {}
            fc.addMultInto(self.feaNum, nextS, neww, w, -1)
            # print "newG: %s" % newGradient
            fc.addMultInto(self.feaNum, nextY, newg, g, -1)
            # print "nextS: %s" % nextS
            # print "nextY: %s" % nextY
            ro = fc.dotProduct(nextS, nextY)
            slist.append(nextS)
            ylist.append(nextY)
            rolist.append(ro)

    def check(self, it, auc):
        # if len(self.lossList) <= 5:
        #     self.lossList.append(newLoss)
        #     return False
        # firstLoss = self.lossList[0]
        #
        # lastLoss = newLoss
        # reduceLoss = (firstLoss - lastLoss )
        # averageReduce = reduceLoss / len(self.lossList)
        #
        # reduceRatio = averageReduce / newLoss
        # if len(self.lossList) == 10:
        #     self.lossList.pop(0)
        # self.lossList.append(lastLoss)
        #
        # if reduceRatio <= self.tol:
        #     return True
        # else:
        #     return False
        self.aucList.append(auc)
        # if it < 5 or (self.aucList[-1] < self.aucList[-2] and self.aucList[-2] < self.aucList[-3]):
        if it < 100:
            return False
        else:
            return True

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