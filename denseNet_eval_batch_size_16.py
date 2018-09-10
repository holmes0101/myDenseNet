# ！/usr/bin/env python
# _*_ coding:utf-8 _*_

import numpy as np
from PIL import Image
import tensorflow as tf
import os
import copy
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

import operator
import time


class LoadDatasForTest(object):
    '''
    将所有的图像与label保存为文件，然后从文件中读入
    '''

    def __init__(self, testDataDir, dataFilePath, targetHeight, targetWidth, labelDataDir):
        '''
        dataFilePath: is the data path. its like this: train/ 0 class File, 1 calss File
        targetHeight: is the height of image you want
        targetWidth: is the width of image you want
        flag: is a sign to show 'train data' or 'test data'
        self.data is a list of [image_file_path, class_label]
        '''

        self.classes = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
        self.data = []
        self.targetHeight = targetHeight
        self.targetWidth = targetWidth
        self.length = 0
        self.counter = 0
        self.testDataDir = testDataDir

        for imageName in os.listdir(dataFilePath):
            self.data.append(imageName)
            self.length += 1

        '''
        with open(dataFilePath, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.replace('\n', ' ')
            lineList = line.split()
            imageName = lineList[0]
            imageName = imageName.replace('.jpeg', '.jpg')
            self.data.append( imageName )
            self.length += 1
        '''

        print("read Test Data is Done! test Data number is: {}".format(self.length))

        self.labelList = []  ### 对应的数值，最终转为np.array， 当某一个属性均为同一个数值时，不对该属性进行训练
        self.labelDict = {}  ### labelName : 对应的label数值
        self.__get_label(labelDataDir)

        self.labelArray = np.array(copy.deepcopy(self.labelList))

        # print("############################################labelDict")
        # print(self.labelDict)

    ### 生成label对应的Dict
    def __get_label(self, labelDataDir):
        with open(labelDataDir, 'r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.replace('\n', ' ')
            lineList = line.split()
            if lineList[0] not in self.labelDict:
                self.labelDict[lineList[0]] = [float(x) for x in lineList[1:]]
            self.labelList.append(self.labelDict[lineList[0]])

    def next_batch(self, batch_size=16):
        '''
        :param batch_size:
        :return:  images is a np.ndarray with shape = [None, targetHeight, targetWidth, 1/3]
                  labes: is a np.ndarray with shape = [None, ]
        '''

        images = []
        if (self.counter + batch_size >= self.length):
            tempData = self.data[self.counter: self.length]
            self.counter = 0
        else:
            tempData = self.data[self.counter: (self.counter + batch_size)]
            # print(tempData)
            self.counter += batch_size
        for item in tempData:
            img_raw = Image.open(self.testDataDir + '/' + item)
            img_raw = img_raw.convert("RGB")
            img_raw = img_raw.resize((self.targetHeight, self.targetWidth), Image.ANTIALIAS)
            img_rawArr = np.array(img_raw)
            img_rawArr = img_rawArr.astype(dtype=np.float32) * (1. / 255)
            images.append(img_rawArr)
        return images, tempData

    def getPredicteLabel(self, pred_list):
        '''

        :param pred_list: is a list of prediction like this: [0 0.1 0.2   0.3]
        :return:   return the lable string name (encode) like ZJL1
        '''
        distanceDict = self.__calculateDis(pred_list)

        ## 按照距离数值从小到大排序， 如果想从大到小排序，加上reverse=True,
        ## 排序后为list类型，list中每个元素为key-value组成的Turple
        distanceDict_Sorted = sorted(distanceDict.items(), key=operator.itemgetter(1), reverse=False)
        return (distanceDict_Sorted[0])[0]

    def __calculateDis(self, pred_list):
        '''

        :param pred_list: is a list of prediction like this: [0 0.1 0.2   0.3]
        :return:  the Eule Distance of
        '''
        pred_list = [pred_list]
        distanceDict = {}
        for index, key in enumerate(self.labelDict):
            tempDis = euclidean_distances(pred_list, [self.labelDict[key]])
            distanceDict[key] = (tempDis[0])[0]
        return distanceDict


if __name__ == "__main__":
    tf.app.flags.DEFINE_string('testDataDir', 'test', 'the Dir of test Images')
    tf.app.flags.DEFINE_integer('targetHeight', 64, 'height of the image you want')
    tf.app.flags.DEFINE_integer('targetWidth', 64, 'width of the image you want')
    tf.app.flags.DEFINE_integer('batch_size', 16, 'batch_size')
    tf.app.flags.DEFINE_string('modelDir', 'model/', 'teh Dir of model')
    FLAGS = tf.app.flags.FLAGS

    os.chdir('../')
    currentDir = os.getcwd()
    # submitDataDir = currentDir + '/DatasetA_train_20180813/submit.txt'

    testDataDir = currentDir + '/DatasetA_test_20180813/DatasetA_test/test'

    testResultDir = currentDir + '/result_batch_size_regu'
    if not os.path.exists(testResultDir):
        os.mkdir(testResultDir)

    label_data_dir = currentDir + '/DatasetA_train_20180813/attributes_per_class.txt'

    testData = LoadDatasForTest(testDataDir, dataFilePath=testDataDir, targetHeight=FLAGS.targetHeight,
                                targetWidth=FLAGS.targetWidth, labelDataDir=label_data_dir)

    columnTotalNumber = testData.labelArray.shape[1]

    curTime = time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))

    testIterNumber = 0
    if testData.length % FLAGS.batch_size == 0:
        testIterNumber = testData.length / FLAGS.batch_size
    else:
        testIterNumber = (testData.length - (testData.length % FLAGS.batch_size)) / FLAGS.batch_size + 1

    with tf.Session() as sess:

        for columnNumber in range(columnTotalNumber):
            if not os.path.exists(currentDir + '/data/model-DenseNet121-regu' + '/' + str(columnNumber)):
                f = open(testResultDir + '/' + str(curTime) + '-columnNumber-' + str(columnNumber) + '-result.txt', 'w')
                for index in range(int(testIterNumber)):
                    X_test, X_testImageName = testData.next_batch(batch_size=FLAGS.batch_size)
                    for ii in range(len(X_testImageName)):
                        f.write(str(X_testImageName[ii]) + ' ')
                        for jj in range(11):
                            if jj == 10:
                                f.write(str(list(set(testData.labelArray[:, columnNumber]))[0]) + '\n')
                            else:
                                f.write(str(list(set(testData.labelArray[:, columnNumber]))[0]) + ' ')
                f.close()
                continue

            f = open(testResultDir + '/' + str(curTime) + '-columnNumber-' + str(columnNumber) + '-result.txt', 'w')
            model_file = tf.train.latest_checkpoint(currentDir + '/data/model-DenseNet121-regu' + '/' + str(columnNumber))
            # print(model_file)
            saver = tf.train.import_meta_graph(model_file + '.meta')  ## load the Graph without weights
            saver.restore(sess,
                          tf.train.latest_checkpoint(currentDir + '/data/model-DenseNet121-regu' + '/' + str(columnNumber)))

            tempResult = []
            for index in range(int(testIterNumber)):
                X_test, X_testImageName = testData.next_batch(batch_size=FLAGS.batch_size)


                graph = tf.get_default_graph()
                prediction = graph.get_tensor_by_name("prediction/prediction:0")
                inputs = graph.get_tensor_by_name("input/inputs:0")

                is_training = graph.get_tensor_by_name('input/is_training:0')

                pred = sess.run(prediction, feed_dict={inputs: X_test, is_training: False})
                #print(pred.shape)
                #pred = pred[0]
                # print(pred.shape[0])
                #print(pred)
                for ii in range(len(X_testImageName)):
                    f.write(str(X_testImageName[ii]) + ' ')
                    tempPred = pred[ii]
                    for jj in range(tempPred.shape[0]):
                        if jj == (tempPred.shape[0] - 1):
                            f.write(str(tempPred[jj]) + '\n')
                        else:
                            f.write(str(tempPred[jj]) + ' ')
                print('columnNumber:{} :{} is done, batch_size: {}'.format(columnNumber, (index + 1), FLAGS.batch_size))
            f.close()








