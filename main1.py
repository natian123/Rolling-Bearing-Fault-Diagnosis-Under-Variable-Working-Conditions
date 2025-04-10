# coding:utf-8
#调用库
import scipy.io as scio
import numpy as np
from keras.optimizer_v2.adam import Adam
from sklearn.manifold import TSNE
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from model import MyModel
from model_DCNN import MyModel_DCNN
from keras.layers import * 

def test():
    #导入数据
    normal = scio.loadmat('98.mat')['X098_DE_time']

    IR007 = scio.loadmat('106.mat')['X106_DE_time']
    B007 = scio.loadmat('119.mat')['X119_DE_time']
    OR007 = scio.loadmat('131.mat')['X131_DE_time']

    IR014 = scio.loadmat('170.mat')['X170_DE_time']
    B014 = scio.loadmat('186.mat')['X186_DE_time']
    OR014 = scio.loadmat('198.mat')['X198_DE_time']

    IR021 = scio.loadmat('210.mat')['X210_DE_time']
    B021 = scio.loadmat('223.mat')['X223_DE_time']
    OR021 = scio.loadmat('235.mat')['X235_DE_time']

    #用滑窗截取样本集
    win_len = 400
    step = 100
    def data_sample(data,win_len,step):
        i = 0
        data_sample=[]
        while i<len(data)-win_len:
            data_sample.append(data[i:i+win_len])
            i = i + step
        return data_sample
    #用上述定义的函数制作样本集
    normal_sample = np.array(data_sample(normal,win_len,step))
    normal_sample = normal_sample[:1200, :, :]
    IR007_sample = np.array(data_sample(IR007,win_len,step))
    B007_sample = np.array(data_sample(B007,win_len,step))
    OR007_sample = np.array(data_sample(OR007,win_len,step))

    IR014_sample = np.array(data_sample(IR014,win_len,step))
    B014_sample = np.array(data_sample(B014 ,win_len,step))
    OR014_sample = np.array(data_sample(OR014,win_len,step))

    IR021_sample = np.array(data_sample(IR021,win_len,step))
    B021_sample = np.array(data_sample(B021 ,win_len,step))
    OR021_sample = np.array(data_sample(OR021,win_len,step))

    #制作样本标签
    normal_label = [0 for i in range(normal_sample.shape[0])] 
    IR007_label = [1 for i in range(IR007_sample.shape[0])]
    B007_label = [2 for i in range(B007_sample.shape[0])]
    OR007_label = [3 for i in range(OR007_sample.shape[0])]

    IR014_label = [4 for i in range(IR014_sample.shape[0])]
    B014_label = [5 for i in range(B014_sample.shape[0])]
    OR014_label = [6 for i in range(OR014_sample.shape[0])]

    IR021_label = [7 for i in range(IR021_sample.shape[0])]
    B021_label = [8 for i in range(B021_sample.shape[0])]
    OR021_label = [9 for i in range(OR021_sample.shape[0])]

    #将样本和标签对应放在一起
    x_data = np.vstack((normal_sample, IR007_sample, B007_sample, OR007_sample, IR014_sample, B014_sample, OR014_sample, IR021_sample, B021_sample, OR021_sample))#閹稿鐎惄瀛樻煙閸氭埊绱欑悰宀勩?庢惔蹇ョ礆閸棗褰旈弫鎵矋閺嬪嫭鍨氭稉?娑擃亝鏌婇惃鍕殶缂??
    y_data = normal_label+IR007_label+B007_label+OR007_label+IR014_label+B014_label+OR014_label+IR021_label+B021_label+OR021_label

    # 将标签one-hot
    y_data = to_categorical(y_data,num_classes=10)

    #随机划分训练和验证集
    X_train,X_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.2)

    #将main1.m文件中的模型选择代表放入model.txt中取出来，对应模型
    m = open('model.txt')
    m = m.read()
    m = int(m) 
    if m == 1:
        model = MyModel_DCNN()
    elif m == 2:
        model = MyModel()

    #学习率
    adam_lr = 0.001
    #设置训练的参数
    model.compile(optimizer=Adam(lr=adam_lr),
                      loss='categorical_crossentropy',metrics=['accuracy'])    

    f = open('epoch.txt')
    epoch_str = f.read()
    epoch = int(epoch_str)

    b = open('batchsize.txt')
    batchsize_str = b.read()
    batchsize = int(batchsize_str)

    ###开始训练，history输出的是训练过程
    history=model.fit( X_train, y_train, batch_size=batchsize, epochs=epoch, verbose=1, validation_data=[X_test,y_test])

    train_loss, train_acc = model.evaluate(X_train, y_train)
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('train_acc', train_acc)
    print('test_acc', test_acc)

    ###可视化部分
    #绘制准确率曲线
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('1.png', dpi=120)
    plt.close()
    #plt.show()
    #绘制loss曲线
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig('2.png', dpi=120)
    plt.close()
    #plt.show()

    if m == 1:
        layer = keras.backend.function([model.layers[0].input], [model.layers[12].output])
    elif m == 2:
        layer = keras.backend.function([model.layers[0].input], [model.layers[10].output])
   
    f1 = layer([X_train])[0]#输出layer层后的输出
    tsne = TSNE(n_components=2, random_state=35)#TSNE降维成2维，可视化
    X_tsne = tsne.fit_transform(f1)
    ax=plt.figure(figsize=(7, 5))#设置图片大小
    plt.title('t-SNE')
    y_ = np.array([item.index(1) for item in y_train.tolist()])#
    scatter=plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_)#画聚类图
    legend1 = ax.legend(*scatter.legend_elements(),    #添加图例
                        loc="center right", title="Classes")
    plt.legend()#将上面定义的图例显示出来
    plt.savefig('tsne.png', dpi=120)
    #plt.show()
    print()
    return history
history = test()
