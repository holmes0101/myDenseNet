#！/usr/bin/env python
# _*_ coding:utf-8 _*_

import numpy as np
from PIL import Image
import os
import copy
import tensorflow as tf

def preprocess_for_train(image, height, width, bbox=None):
    # 如果没有提供标注框，则认为整个图像就是需要关注的部分

    if bbox is None:
        bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])

    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # 随即截取图像，减小需要关注的物体大小对图像识别算法的影响
    # bbox_begin, bbox_size, _ = tf.image.sample_distorted_bounding_box(tf.shape(image), bounding_boxes = bbox)
    # distorted_image = tf.slice(image, bbox_begin, bbox_size)

    distorted_image = tf.image.random_hue(image, 0.3, seed=2018)

    distorted_image = tf.image.random_saturation(distorted_image, 1, 5, seed=2108)

    distorted_image = tf.image.random_brightness(distorted_image, 0.1, seed=2801)

    # 将随即截取的图片调整为神经网络的输入大小，method可以选择插值的方式, 0为双线性1为最近邻2为双三次3为面积
    #distorted_image = tf.image.resize_images(distorted_image, [height, width], method=0)
    # 随机左右翻转
    distorted_image = tf.image.random_flip_left_right(distorted_image)
    return distorted_image


class LoadDatas(object):
    '''
    将所有的图像与label保存为文件，然后从文件中读入
    '''
    def __encode__(self, imageName):
        '''

        :param imageName:  the name of category in real world
        :return:   the encode of label
        '''
        for ii in range(len(self.classes)):
            if imageName == self.classes[ii]:
                return ii
        raise("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!label error!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    def __init__(self, dataFilePath, targetHeight, targetWidth, labelDataDir, flag):
        '''
        dataFilePath: is the data path. its like this: train/ 0 class File, 1 calss File
        targetHeight: is the height of image you want
        targetWidth: is the width of image you want
        flag: is a sign to show 'train data' or 'test data'
        self.data is a list of [image_file_path, class_label]
        '''
        if (flag == "train") or (flag == 'train'):
            self.flag = 'train'
        elif(flag == 'test' or (flag == "val")):
            self.flag = 'test'
        else:
            raise("！！！！！！！！！！！！！！！！Data Type is Error!!!!!!!!!!!!!")

        self.classes = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
        self.data = []
        self.targetHeight = targetHeight
        self.targetWidth = targetWidth
        self.length = 0
        self.counter = 0
        for className in os.listdir(dataFilePath):
            for imageName in os.listdir( dataFilePath + '/' + str(className)):
                tempImageName = dataFilePath + '/' + str(className) + '/' + str(imageName)
                self.data.append([tempImageName, str(className)])
                self.length += 1
        if self.flag == 'train':
            print("read Train Data is Done! train Data number is: {}".format(self.length))
        elif self.flag == 'test':
            print("read Test Data is Done! test Data number is: {}".format(self.length))


        self.usedData = []       ## shuffle Batch 时使用
        self.labelList = []  ### 对应的数值，最终转为np.array， 当某一个属性均为同一个数值时，不对该属性进行训练
        self.labelDict = {} ### labelName : 对应的label数值
        self.__get_label(labelDataDir)

        self.labelArray = np.array(copy.deepcopy(self.labelList))

        #print("############################################labelDict")
        #print(self.labelDict)





    ### 生成label对应的Dict
    def __get_label(self, labelDataDir):
        with open(labelDataDir, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.replace('\n', ' ')
            lineList = line.split()
            if lineList[0] not in self.labelDict:
                self.labelDict[ lineList[0] ] = [float(x) for x in lineList[1:]]
            self.labelList.append( self.labelDict[ lineList[0] ] )





    def next_batch(self, columnNumber, batch_size=16):
        '''

        :param batch_size:
        :return:  images is a np.ndarray with shape = [None, targetHeight, targetWidth, 1/3]
                  labes: is a np.ndarray with shape = [None, ]
        '''
        if self.flag == 'train':
            # np.random.shuffle(dataset)

            ###  每次均打乱，但是不放回抽样
            images = []
            labels = []
            if len(self.usedData) < batch_size:
                self.usedData = copy.deepcopy(self.data)
            #print("data number: {}, usedData number: {}".format(len(self.data), len(self.usedData)))
            np.random.shuffle(self.usedData)
            tempData = self.usedData[0:batch_size]
            #print(self.usedData[batch_size])
            for item in tempData:
                #print(item)
                img_raw = Image.open(item[0])
                img_raw = img_raw.convert("RGB")
                img_raw = img_raw.resize((self.targetHeight, self.targetWidth))
                img_rawArr = np.array(img_raw)
                #img_rawArr = preprocess_for_train(img_raw, self.targetHeight, self.targetWidth)
                img_rawArr = img_rawArr.astype(dtype=np.float32) * (1. / 255)
                #img_rawArr = img_rawArr * (1. / 255)
                images.append(img_rawArr)
                #print( img_rawArr )
                tempList = self.labelDict[ item[1] ]
                labels.append( self.__oneHotLabel(tempList[columnNumber]) )
                #print(tempList[columnNumber])
                #print( labels )

            tempCounter = 0
            while(tempCounter < batch_size):
                tempCounter += 1
                del self.usedData[0]
            #print(self.usedData[0])
            #raise("!!!!!!!!!!!!!!!!!!!!test!!!!!!!!!!!!!!!!!!!!!!!")
            return images, labels



        elif self.flag == 'test':
            images = []
            labels = []
            if( self.counter + batch_size >= self.length ):
                tempData = self.data[self.counter : self.length]
                self.counter = 0
            else:
                tempData = self.data[self.counter: (self.counter + batch_size)]
                self.counter += batch_size
            for item in tempData:
                img_raw = Image.open(item[0])
                img_raw = img_raw.convert("RGB")
                img_raw = img_raw.resize((self.targetHeight, self.targetWidth), Image.ANTIALIAS)
                img_rawArr = np.array(img_raw)
                img_rawArr = img_rawArr.astype(dtype=np.float32) * (1. / 255)
                images.append(img_rawArr)
                tempList = self.labelDict[ item[1] ]
                labels.append( self.__oneHotLabel(tempList[columnNumber]) )
            return images, labels

    def __oneHotLabel(self, label):
        label_oht = [0 for ii in range(len(self.classes))]
        for ii in range(len(self.classes)):
            #print( float(label) )

            if float( self.classes[ii] ) == float(label):
                label_oht[ii] = 1
                return label_oht
        raise('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!label error!!!!!!!!!!!!!!!!!')

