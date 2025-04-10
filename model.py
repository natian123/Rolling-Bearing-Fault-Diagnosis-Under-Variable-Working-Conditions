###模型搭建
from keras.layers import * #将keras api中的特征处理层引入，包括cnn多种
from keras.models import Model #用于搭建模型

def MyModel():

    data_input=Input(shape=(400,1))
    conv1=convolutional.Conv1D(64,3,strides=3,padding="same")(data_input)
    conv1=BatchNormalization(momentum=0.8)(conv1)
    conv1=MaxPool1D(pool_size=4)(conv1)

    # conv2=convolutional.Conv1D(128,3,strides=3,padding="same")(conv1)
    # conv2=BatchNormalization(momentum=0.8)(conv2)
    # conv2=MaxPool1D(pool_size=4)(conv2)

    conv3=convolutional.Conv1D(128,3,strides=3,padding="same")(conv1)
    conv3=BatchNormalization(momentum=0.8)(conv3)
    conv3=MaxPool1D(pool_size=4)(conv3)

    flatten=Flatten()(conv3)
    dense_1=Dense(1000)(flatten)
    dense_1=Dropout(0.5)(dense_1)
    dense_1=Dense(100)(dense_1)
    output = Dense(10, activation='softmax')(dense_1)

    cnn_model= Model(inputs=data_input, outputs=output)
    cnn_model.summary() #打印模型结构与参数
    return cnn_model