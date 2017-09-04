
import numpy as np
import random
import function as fc

class SGD():
    def __init__(self,
                 feaNum,
                 norm_u_2 = 0.1,
                 norm_w_2 = 0.01,
                 step = 0.1,
                 iterNum = 2,
                 intercept = True,
                 M = 2):
        self.feaNum = feaNum
        self.norm_u_2 = norm_u_2
        self.norm_w_2 = norm_w_2
        self.step = step
        self.iterNum = iterNum
        self.intercept = intercept
        self.M = M
        if self.intercept:
            self.feaNum += 1

    def do_sgd(self, data):
        """
        run sgd algorithm, only with L2 norm
        :return:
        """
        if self.intercept:
            data = list(map(self.addBias, data))
        print (data)
        it = 0
        weight_w = [{} for i in  range(self.M)]
        weight_u = [{} for i in  range(self.M)]
        for w in weight_w:
            for i in range(self.feaNum):
                w[i] = np.random.normal(0,1)
        for u in weight_u:
            for i in range(self.feaNum):
                u[i] = np.random.normal(0,1)

        shuffle_index = list(range(len(data)))

        while it < self.iterNum:
            print ("-----------------iterator : %s-----------------" % it)
            #calculate derivative

            random.shuffle(shuffle_index)
            for i in shuffle_index:
                item = data[i]

                #calculate the direction of gradient respectively
                dir_W, dir_U = self.cal_derivative(weight_w, weight_u, item)
                for i in range(self.M):
                    #update w and u
                    for d in dir_W[i]:
                        old_w = weight_w[i].get(d, 2 * random.random()-1)
                        # old_w = weight_w[i].get(d, 0)
                        new_w = old_w - self.step * (dir_W[i][d] + self.norm_w_2 * old_w)
                        if abs(new_w) > np.exp(-30):
                            weight_w[i][d] = new_w
                        elif d in weight_w[i]:
                            weight_w[i].pop(d)
                    for d in dir_U[i]:
                        old_u = weight_u[i].get(d, 2 * random.random()-1)
                        # old_u = weight_u[i].get(d, 0)
                        new_u = old_u - self.step * (dir_U[i][d] + self.norm_u_2 * old_u)
                        if abs(new_u) > np.exp(-30):
                            weight_u[i][d] = new_u
                        elif d in weight_u[i]:
                            weight_u[i].pop(d)
            self.cal_loss(weight_w, weight_u, data)

            it += 1
        print ("----------------------  end  ----------------------")
        return weight_w, weight_u


    def cal_derivative(self, W_w, W_u, item):
        """
        calculate derivative only with l2 norm
        :param weight:
        :return:
        """
        label, featureDic = item
        dir_W = []
        dir_U = []
        temp_eux = []
        temp_sywx = []
        sum_eux = 0.0
        sum_eux_sywx = 0.0
        for i in range(self.M):
            #get all the temp exp(uj * x) and sigmoid(y * wj * x)
            #and get sum at the same time
            eux = np.exp(fc.dotProduct(W_u[i], featureDic))
            sywx = fc.sigmoid(label * fc.dotProduct(W_w[i], featureDic))
            temp_eux.append(eux)
            temp_sywx.append(sywx)
            sum_eux += eux
            sum_eux_sywx += eux * sywx
        for i in range(self.M):
            #calculate array uj and array wj
            dir_w = {}
            dir_u = {}
            for index in featureDic:
                dir_u[index] = temp_eux[i] * featureDic[index] / sum_eux - \
                    temp_eux[i] * temp_sywx[i] * featureDic[index] / sum_eux_sywx
                dir_w[index] = label * temp_sywx[i] * ( temp_sywx[i] - 1 ) * featureDic[index] / sum_eux_sywx
            dir_W.append(dir_w)
            dir_U.append(dir_u)

        return dir_W, dir_U


    def cal_loss(self, W_w, W_u, data):
        """
        calculate the loss over all data
        :param w_w:
        :param w_u:
        :param data:
        :return:
        """
        loss = 0.0
        for label, featureDic in data:
            #sum_u is sum(exp(uj * x))
            #sum_us is sum(exp(uj * x) * sigmoid(y * wi * x))
            sum_u = 0
            sum_us = 0
            for i in range(self.M):
                wx = fc.dotProduct(W_w[i], featureDic)
                eux = np.exp(fc.dotProduct(W_u[i], featureDic))
                sum_u += eux
                sum_us += eux * fc.sigmoid(label * wx)
            loss += np.log(sum_u) - np.log(sum_us)
        print("loss is:  %s" % loss)



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





