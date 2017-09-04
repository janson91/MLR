## Large Scale Piece-wise linear model

论文为 Learning Piece-wise Linear Models from Large Scale Data for Ad Click Prediction.pdf



### 算法实现

mlr的代码不难写，不过mlr可以通过多种优化方法进行训练，下面一共有三种方法，其中包括不同的范数

|        | L1   | L2   | L21  |
| ------ | ---- | ---- | ---- |
| SGD    |      | √    |      |
| FTRL   | √    | √    |      |
| LS-PLM | √    |      | √    |

- **ls_plm.py** : LS-PLM算法，通过函数 _lsplm.train(data,test_data) 进行训练
- **ftrl.py** : FTRL算法，通过函数 _ftrl.train(data,test_data) 进行训练
- **sgd.py** : SGD算法，通过函数 _sgd.train(data,test_data) 进行训练
- **function.py** : 以上三个算法能用到的功能函数。包括mlr计算、梯度计算、loss计算等等

### 运行

- **test.py** : 测试样例，运行并绘图
- **readFile.py** : 读入电商数据，运行并绘图

### 预处理

- **pre_train.py** : 预处理电商数据
- **pretrain_data.pkl** : 预处理好的数据