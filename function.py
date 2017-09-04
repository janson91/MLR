import numpy as np
import time
import copy

def performance(f): #定义装饰器函数，功能是传进来的函数进行包装并返回包装后的函数
    def fn(*args, **kw):       #对传进来的函数进行包装的函数
        t_start = time.time()  #记录函数开始时间
        r = f(*args, **kw)     #调用函数
        t_end = time.time()    #记录函数结束时间
        print ('call %s() in %fs' % (f.__name__, (t_end - t_start)))  #打印调用函数的属性信息，并打印调用函数所用的时间
        return r               #返回包装后的函数
    return fn #调用包装后的函数

def mlr(W, U, item):
    """
    calculate mixture logistic regression
    :param U:
    :param W:
    :param x:
    :return:
    """
    label, x = item
    prob = 0.0
    ux = []
    for u in U:
        ux.append(dotProduct(u, x))
    ux = softmax(ux)
    # print(ux)
    for index, w in enumerate(W):
        prob += ux[index] * sigmoid(dotProduct(w, x))
    # print (label, " ", prob, x)
    return prob


def dotProduct(weight, featureDic):
    """
    calculate w * x
    :param featureDic:
    :param weight:
    :return:
    """
    result = 0.0
    for index in featureDic:
        x = featureDic[index]
        w = weight.get(index, 0)
        result += x * w
    return result

def sigmoid(z):
    """
    calculate sigmoid
    :param z:
    :return:
    """
    return 1 / (1 + np.exp( -max(min(z, 35), -35) ))

def softmax(x):
    """
    softmax a array
    :param x:
    :return:
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def calculateY(X, w):
    Y = []
    for x in X:
        y = -(w[2] + x * w[0]) / w[1]
        Y.append(y)
    return Y

def calLoss(data, weight_W, weight_U, norm21, norm1, feaNum):
    """
        计算loss
    :param data:
    :param weight_W:
    :param weight_U:
    :return:
    """
    #混合逻辑回归的loss
    functionLoss = calFunctionLoss(weight_W, weight_U, data)
    #L21正则的loss
    norm21Loss = calNorm21(weight_W + weight_U, feaNum)
    #L1正则的loss
    norm1Loss = calNorm1(weight_W + weight_U)
    print( functionLoss , norm21 * norm21Loss , norm1 * norm1Loss)
    return functionLoss + norm21 * norm21Loss + norm1 * norm1Loss

def calFunctionLoss(W_w, W_u, data):
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
        for i in range(len(W_w)):
            wx = dotProduct(W_w[i], featureDic)
            eux = np.exp(dotProduct(W_u[i], featureDic))
            sum_u += eux
            sum_us += eux * sigmoid(label * wx)
        loss += np.log(sum_u) - np.log(sum_us)
    return loss
    # print("loss is:  %s" % loss)

def calNorm21(weight, feaNum):
    '''
        计算norm21
    :param weight:
    :return:
    '''
    loss = 0.0
    for i in range(feaNum):
        d = 0.0
        #计算所有weight的第i个维度上的平方和
        for w in weight:
            d += w.get(i, 0) ** 2
        loss += d ** 0.5
    return loss

def calNorm1(weight):
    """
        计算norm1
    :param weight:
    :return:
    """
    loss = 0.0
    for w in weight:
        for v in w.values():
            loss += abs(v)
    return loss

def calDimension21(W, feaNum):
    """
        计算每一个维度的L2
    :param W:
    :return:{dimension1:std1, dimension2:std2 ......}
    """
    D21 = {}
    for index in range(feaNum):
        sum = 0
        for w in W:
            sum += w.get(index, 0) ** 2
        if sum != 0:
            D21[index] = sum ** 0.5
    return D21

def calGradient(data, weight_W, weight_U, norm2, norm1, feaNum):
    """
        计算 gradient,
        w,u每个都有calssNum个向量
    :param data:
    :param weight_W:
    :param weight_U:
    :param norm2:
    :param norm1:
    :param feaNum:
    :return:
    """
    gW = []
    gU = []


def cal_derivative(W_w, W_u, item, weight = 1):
    """
    calculate derivative
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
    for i in range(len(W_w)):
        #get all the temp exp(uj * x) and sigmoid(y * wj * x)
        #and get sum at the same time
        eux = np.exp(dotProduct(W_u[i], featureDic))
        sywx = sigmoid(label * dotProduct(W_w[i], featureDic))
        temp_eux.append(eux)
        temp_sywx.append(sywx)
        sum_eux += eux
        sum_eux_sywx += eux * sywx
    for i in range(len(W_w)):
        #calculate array uj and array wj
        dir_w = {}
        dir_u = {}
        for index in featureDic:
            dir_u[index] = temp_eux[i] * featureDic[index] / sum_eux - \
                temp_eux[i] * temp_sywx[i] * featureDic[index] / sum_eux_sywx
            dir_w[index] = label * temp_eux[i] *  temp_sywx[i] * ( temp_sywx[i] - 1 ) * featureDic[index] / sum_eux_sywx
            if label > 0:
                dir_u[index] *= weight
                dir_w[index] *= weight
        dir_W.append(dir_w)
        dir_U.append(dir_u)

    return dir_W, dir_U

def sumCalDerivative(WW, WU, data,weight = 1):
    # 计算所有样本的梯度和(所有样本的一阶导数和），weight为负样本数/正样本数。
    LW = [{} for _ in range(len(WW))]
    LU = [{} for _ in range(len(WW))]
    for item in data:
        lw, lu = cal_derivative(WW, WU, item)
        #如果是正样本，乘以权重，这样正样本更重要
        if item[0] > 0:
            pos_weight = weight
        else:
            pos_weight = 1
        for i in range(len(WW)):
            for index in lw[i]:
                LW[i].setdefault(index, 0)
                LW[i][index] += lw[i][index] * pos_weight
            for index in lu[i]:
                LU[i].setdefault(index, 0)
                LU[i][index] += lu[i][index] * pos_weight
    for lw in LW:
        for k in lw:
            lw[k] /= len(data)
    for lu in LU:
        for k in lu:
            lu[k] /= len(data)

    return LW, LU




def virtualGradient(feaNum, WW, WU, GW, GU,beta,lamb):
    """
    计算虚梯度,也就是论文中的d_ij
    :param feaNum:
    :param weight_W:
    :param weight_U:
    :param gradient_W:
    :param gradient_U:
    :param norm21:
    :param norm1:
    :return:
    """
    #计算θ_i·
    D21 = calDimension21(WW + WU, feaNum)
    #计算v：
    VW = calV(GW, beta)
    VU = calV(GU, beta)
    #计算v_i·
    VD21 = calDimension21(VW + VU, feaNum)
    sumVD21 = sum(VD21.values())

    #
    #计算d_ij
    DW = calDij(GW, WW, VW, D21, sumVD21, beta, lamb, feaNum)
    DU = calDij(GU, WU, VU, D21, sumVD21, beta, lamb, feaNum)
    return DW, DU

def calV(L, beta):
    """
        计算v，包括wv， uv，这里是分别计算的
        （可以和到一起算，因为w，u一直都是分着算的，所以这里也分着算了。重构的时候再优化吧）
    :param LW:
    :param LU:
    :param beta:
    :param lamb:
    :return:
    """
    V = copy.deepcopy(L)
    for v in V:
        for index in v:
            v[index] = max(abs(v[index]) -  beta, 0) * sign(-v[index])
    return V

def calDij(L, W, V, D21, sumVD21, beta, lamb, feaNum):
    """
    分三种情况讨论，并计算d_i
    :param L:  loss of θ, matrix
    :param W:  weight,θ, matrix
    :param V: v , matrix
    :param D21: norm21, W_i·  of W , vector
    :param sumVD21:  norm21, value
    :param beta:
    :param lamb:
    :param feaNum:
    :return:
    """
    D = [{} for _ in range(len(W))]
    for i,d in enumerate(D):
        for index in range(feaNum):
            if D21.get(i,0) == 0:
                temp = V[i].get(index, 0) * max(sumVD21 - lamb, 0) / sumVD21

            elif W[i].get(index, 0) == 0:
                s = -L[i].get(index,0)
                temp = max(abs(s) - beta, 0) * sign(s)

            else:
                s = -L[i].get(index, 0) - lamb * W[i].get(index, 0) / D21.get(index)
                temp = s - beta * sign(W[i].get(index, 0))

            if temp != 0:
                d[index] = temp
    return D


def lbfgs(feaNum, vG, sList, roList, yList, alphaList, DIR):
    """
        两个循环计算下降方向,拟合Hessian矩阵的 逆H 和梯度负方向的乘积，即 -H * f'
    :param feaNum:
    :param vG: matrix, 2m*d
    :param sList:matrix, 2m*d
    :param roList:matrix, 2m*d
    :param yList:matrix, 2m*d
    :param alphaList:matrix, 2m*d
    :return:
    """
    for _i in range(len(sList)):
        vg = vG[_i]
        slist = sList[_i]
        rolist = roList[_i]
        ylist = yList[_i]
        alist = alphaList[_i]
        dir = DIR[_i]
        count = len(slist)
        if count > 0:
            indexList = list(range(0, count))
            indexList.reverse()
            for i in indexList:
                alist[i] = -1.0 * dotProduct(vg,slist[i]) / rolist[i]
                addMult(feaNum, vg, ylist[i], alist[i])

            lastY = ylist[-1]
            yDotY = dotProduct(lastY, lastY)
            scalar = rolist[-1] / yDotY
            scale(vg, scalar);

            for i in range(0, count):
                beta = dotProduct(vg, ylist[i],) / rolist[i]
                addMult(feaNum, vg, slist[i], -alist[i] - beta);

        #判断y(k)T * s(k) > 0
        if count > 0 and dotProduct(ylist[-1], slist[-1]) > 0:
            for index in vg:
                if sign(vg[index] != sign(dir.get(index,0))):
                    vg[index] = dir.get(index,0)
        else:
            vG[_i] = dir

def addMult(paramCount, vecDic1, vecDic2, c):
    for index in range(0, paramCount):
        v1 = vecDic1.get(index, 0)
        vecDic1[index] = v1 + vecDic2.get(index, 0) * c

def addMultInto(paramCount, vec1, vec2, vec3, c):
    for index in range(0, paramCount):
        vec1[index] = vec2.get(index, 0) + vec3.get(index, 0) * c


def scale(vecDic1, c):
    for index in vecDic1:
        vecDic1[index] *= c

def fixDirection(vGW, vGU, dirW, dirU):
    """
        检查下降是否跨象限
    :param vGW:
    :param vGU:
    :param dirW:
    :param dirU:
    :return:
    """
    pass

def backTrackingLineSearch(feaNum, it, loss, data, W, L, vG, dir,norm21, norm1):
    """
        线性搜索，得到最佳步长并更新权重
    :param it:
    :param oldLoss:
    :param data:
    :param WW:
    :param WU:
    :param GW:
    :param GU:
    :param vGW:
    :param vGU:
    :return:
    """
    alpha = 1.0
    backoff = 0.5
    if it == 0:
        # normalDir = dotProduct(vG, vG) ** 0.5
        # alpha =  1.0 / normalDir
        backoff = 0.1
    gamma = 1e-5
    loss_it = 0;
    while True:


        newW = getNewWeight(feaNum, W, vG, alpha, dir  )

        new_loss = calLoss(data, newW[:len(newW)//2], newW[len(newW)//2:], norm21, norm1, feaNum)

        #论文中的阈值项
        threshold = calThreshold(dir, W, newW)

        if new_loss <= loss + gamma * threshold or(loss_it > 0 and new_loss > pre_loss):
            return new_loss, newW
        pre_loss = new_loss
        alpha *= backoff
        loss_it += 1

def calThreshold(dir, W, newW):
    """
        计算论文中阈值项
    :param dir:
    :param W:
    :param newW:
    :return:
    """
    threshold = 0
    for i,d in enumerate(dir):
        for index in d:
            threshold += -d[index] * (newW[i].get(index, 0) - W[i].get(index, 0))
    return threshold

def getNewWeight(feaNum, W, vG, alpha, dir  ):
    """
        计算新的参数
    :param feaNum:
    :param W:
    :param vG:
    :param alpha:
    :return:
    """
    new_W = [{} for _ in range(len(W))]
    for i, w in enumerate(W):
        for index in range (feaNum):
            _w = w.get(index, 0)
            if _w == 0:
                _sign = sign(dir[i].get(index, 0))
            else:
                _sign = sign(_w)
            _new_w = _w + alpha * vG[i].get(index,0)
            if sign(_new_w) == sign(_sign):
                new_W[i][index] = _new_w
    return new_W


def check(it, loss, newLoss, WW, WU):
    """
        检查是否提前终止
    :param it:
    :param loss:
    :param newLoss:
    :param WW:
    :param WU:
    :return:
    """
    pass

def shift(sList, yList, roList, weight, newWeight, gradient, newGradient):
    """
        更新
    :param sList:
    :param yList:
    :param roList:
    :param weight:
    :param newWeight:
    :param gradient:
    :param newGradient:
    :return:
    """
    pass

def sign(x):
    """
    return 1,0,-1
    :param x:
    :return:
    """
    if x < 0:
        return -1
    elif x > 0:
         return 1
    else:
        return 0

def calAcc(data, WW, WU):
    count = 0.0
    for i in range(len(data)):
        d = data[i]
        result = mlr(WW, WU, d)
        if abs(result - (1 + d[0]) / 2) < 0.5:
            count += 1
    return count / len(data)