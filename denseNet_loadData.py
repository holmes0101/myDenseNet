#！/usr/bin/env python
# _*_ coding:utf-8 _*_


import os
import numpy as np
import tensorflow as tf


from loadData import LoadDatas



import time
from tensorflow.python.framework import graph_util
import copy

def print_Layer(layer):
    print(layer.op.name, ' ', layer.get_shape().as_list())

def myConv2d(input_tensor, conv_size, stride_size ,output_channel, name, regu=None, padding='SAME', act=tf.nn.relu, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        input_channel = input_tensor.get_shape()[-1].value
        weights = tf.get_variable(name='weights', shape=[conv_size, conv_size, input_channel, output_channel],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.01))
        biases = tf.get_variable(name='biases', shape=[output_channel], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.001))
        conv = tf.nn.conv2d(input=input_tensor, filter=weights, strides=[1, stride_size, stride_size, 1], padding=padding,
                            use_cudnn_on_gpu=True, name=name)
        if regu != None and reuse != True:  ## reuse = False 当重用时，只计算一次
            tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regu)(weights))
        print_Layer(conv)
        if act == None:
            return tf.nn.bias_add(value=conv, bias=biases)
        conv_relu = act(tf.nn.bias_add(value=conv, bias=biases))
        print_Layer(conv_relu)
        return conv_relu


def myMaxPooling2D(input_tensor, ksize, strides, padding, name, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        max_pool = tf.nn.max_pool(input_tensor, ksize=[1, ksize, ksize, 1],
                                  strides=[1, strides, strides, 1], padding=padding, name=name)
        print_Layer(max_pool)
        return max_pool


def myFc(input_tensor, output_channel, name, regu=None, act=tf.nn.relu, reuse=False):
    input_channel = input_tensor.get_shape()[-1].value
    with tf.variable_scope(name, reuse=reuse):
        weights = tf.get_variable(name='weights', shape=[input_channel, output_channel], dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.01))
        biases = tf.get_variable(name='biases', shape=[output_channel], dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.001))
        if regu != None and reuse != True:
            tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regu)(weights))
        fc = tf.matmul(input_tensor, weights) + biases
        if act != None:
            act_fc = act(fc)
            print_Layer(act_fc)
            return act_fc
        else:
            print_Layer(fc)
            return fc


def myDropout(input_tensor, rate, name, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        drop = tf.nn.dropout(input_tensor, rate, name=name)
        print_Layer(drop)
        return drop


def myBN( input_tensor, name, is_training, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        bn = tf.contrib.layers.batch_norm(input_tensor,
                                          scale=True, is_training=is_training,
                                          updates_collections=None)
        print_Layer(bn)
        return bn


def myAvgPool(input_tensor, ksize, stride_size, name, padding='VALID', reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        avg_pool = tf.nn.avg_pool(input_tensor, ksize=[1, ksize, ksize, 1],
                                  strides=[1, stride_size, stride_size, 1],
                                  padding=padding, name=name
                                  )
        print_Layer(avg_pool)
        return avg_pool

def denseConv(input_tensor, conv_size,stride_size, is_training, output_channel, name, regu=None, padding='SAME', act=tf.nn.relu, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        conv = myConv2d(input_tensor, conv_size, stride_size, output_channel,
                        name='conv', regu=regu, padding=padding,act=None, reuse=reuse)

        conv_bn = myBN(conv, 'BN', is_training=is_training, reuse=reuse)
        conv_bn_relu = act(conv_bn)
        print_Layer(conv_bn_relu)
        return conv_bn_relu

def denseBN_Relu_Conv(input_tensor, conv_size, is_training, stride_size ,output_channel, name, regu=None, padding='SAME', act=tf.nn.relu, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        bn = myBN(input_tensor, name='BN', is_training=is_training, reuse=reuse)
        bn_relu = act(bn, name='Relu')
        print_Layer(bn_relu)
        bn_relu_conv = myConv2d(bn_relu, conv_size=conv_size, stride_size=stride_size,
                                output_channel=output_channel, name='conv', act=None,
                                regu=regu, padding=padding, reuse=reuse)
        return bn_relu_conv

######### output_channel: 是指每次增加的 channel数目
def denseBlock( input_tensor, stride_size,is_training, output_channel, name, regu=None, padding='SAME', act=tf.nn.relu, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        input_channel = input_tensor.get_shape()[-1].value
        conv1 = denseBN_Relu_Conv(input_tensor, conv_size=1, is_training=is_training,
                                  stride_size=stride_size, output_channel=input_channel,
                         name='1_1_conv', regu=regu, padding=padding, act=act,
                         reuse=reuse)

        conv3 = denseBN_Relu_Conv(input_tensor=conv1, conv_size=3, is_training=is_training,
                                  stride_size=stride_size, output_channel=output_channel,
                          name='3_3_conv', regu=regu, padding=padding, act=act,
                          reuse=reuse)
        conv3_con = tf.concat([input_tensor, conv3], axis=3, name='concat')
        print_Layer(conv3_con)
        return conv3_con

def denseTransition(input_tensor, stride_size, is_training, name,regu=None, padding='VALID', act=tf.nn.relu, reuse=False):
    with tf.variable_scope(name, reuse=reuse):
        input_channel = input_tensor.get_shape()[-1].value
        conv1 = denseBN_Relu_Conv(input_tensor, conv_size=1, is_training=is_training,
                                  stride_size=stride_size, output_channel=input_channel,
                          name='1_1_conv', act=act, regu=regu, padding=padding, reuse=reuse)

        #conv2 = myMaxPooling2D(conv1, ksize=2, strides=2, padding='VALID', name='Maxpool', reuse=reuse)
        conv2 = myAvgPool(conv1, ksize=2, stride_size=2, name='2_2_avgpool', padding='VALID', reuse=reuse)
        return conv2

## 0.0005
def defmyDensetNet(input_tensor, is_training, regu=0.005):
    '''

    :param intpu_tensor:
    :return:
    '''
    # input: 64 * 64 *3  output: 64 * 64 * 64
    x = denseConv(input_tensor, conv_size=3, is_training=is_training, stride_size=1, output_channel=32, padding='SAME',
                  reuse=False, regu=regu, name='Conv1')



    ## input: 64 * 64 * 64  output: 32 * 32 * 64
    x = myMaxPooling2D(x, ksize=2, strides=2, padding='VALID',
                       name='MaxPooling1', reuse=False)

    # input: 32 * 32 * 64  output: 32 * 32 * 128
    x = denseConv(x, conv_size=3, stride_size=1, is_training=is_training, output_channel=64, padding='SAME',
                  reuse=False, regu=regu, name='Conv2')

    x = denseConv(x, conv_size=3, stride_size=1, is_training=is_training, output_channel=128, padding='SAME',
                  reuse=False, regu=regu, name='Conv3')

    # input: 32 * 32 * 64  output: 16 * 16 * 128
    x = myMaxPooling2D(x, ksize=2, strides=2, padding='VALID',
                       name='MaxPooling2', reuse=False)


    ############################################## 6 Dense Block(1)
    ## input: 16 * 16 * 128 output: 16 * 16 * 416(128 + 48 * 6)
    blockCircleNumber = 6
    for ii in range(blockCircleNumber):
        tempName = 'DenseBlock1-{}'.format(str(ii + 1))
        x = denseBlock(x, stride_size=1, is_training=is_training, output_channel=48, name=tempName,
                       padding='SAME', regu=regu, reuse=False)

    ############################################# Transition Layer(1)
    ## input: 32 * 32 * 416   output: 16 * 16 * 416
    x = denseTransition(x, stride_size=1, is_training=is_training, regu=regu, name='Transition1', padding='VALID', reuse=False)

    ############################################# 8 Dense Block (2)
    ## input: 16 * 16 * 416  output: 16 * 16 * 800 (416 + 48 * 8)
    blockCircleNumber = 8
    for ii in range(blockCircleNumber):
        tempName = 'DenseBlock2-{}'.format(str(ii + 1))
        x = denseBlock(x, stride_size=1, is_training=is_training, output_channel=48, name=tempName,
                       padding='SAME', regu=regu, reuse=False)

    ############################################# Transition Layer(2)
    ## input: 16 * 16 * 800 output: 8 * 8 * 800
    x = denseTransition(x, stride_size=1, is_training=is_training, regu=regu, name='Transition2', padding='VALID', reuse=False)

    ############################################# 8 Dense Block (3)
    ## input: 16 * 16 * 416  output: 16 * 16 * 800 (416 + 48 * 8)
    blockCircleNumber = 8
    for ii in range(blockCircleNumber):
        tempName = 'DenseBlock3-{}'.format(str(ii + 1))
        x = denseBlock(x, stride_size=1, is_training=is_training, output_channel=48, name=tempName,
                       padding='SAME', regu=regu, reuse=False)

    ############################################ Classification Layer
    ## input: 8 * 8 * 800 output: 800
    x = myAvgPool(x, ksize=4, stride_size=1, name='Classification-avgpool',
                  padding='VALID', reuse=False)



    input_channel = x.get_shape()[-1].value
    x = tf.reshape(x, [-1, input_channel], name='reshape')
    print_Layer(x)

    #x = myFc(x, output_channel=512, regu=regu, name='Fc1', act=tf.nn.relu, reuse=False)

    logist = myFc(x, output_channel=11,
                  name='Fc-logist', act=None, reuse=False)

    return logist

## 0.0005
def defmyDensetNet2(input_tensor, is_training, regu=0.005):
    '''

    :param intpu_tensor:
    :return:
    '''
    # input: 64 * 64 *3  output: 64 * 64 * 64
    x = denseConv(input_tensor, conv_size=3, is_training=is_training, stride_size=1, output_channel=32, padding='SAME',
                  reuse=False, regu=regu, name='Conv1')



    ## input: 64 * 64 * 64  output: 32 * 32 * 64
    x = myMaxPooling2D(x, ksize=2, strides=2, padding='VALID',
                       name='MaxPooling1', reuse=False)

    ############################################## 6 Dense Block(1)
    ## input: 16 * 16 * 128 output: 16 * 16 * 416(128 + 48 * 6)
    blockCircleNumber = 6
    for ii in range(blockCircleNumber):
        tempName = 'DenseBlock1-{}'.format(str(ii + 1))
        x = denseBlock(x, stride_size=1, is_training=is_training, output_channel=48, name=tempName,
                       padding='SAME', regu=regu, reuse=False)

    ############################################# Transition Layer(1)
    ## input: 32 * 32 * 416   output: 16 * 16 * 416
    x = denseTransition(x, stride_size=1, is_training=is_training, regu=regu, name='Transition1', padding='VALID', reuse=False)

    ############################################# 8 Dense Block (2)
    ## input: 16 * 16 * 416  output: 16 * 16 * 800 (416 + 48 * 8)
    blockCircleNumber = 8
    for ii in range(blockCircleNumber):
        tempName = 'DenseBlock2-{}'.format(str(ii + 1))
        x = denseBlock(x, stride_size=1, is_training=is_training, output_channel=48, name=tempName,
                       padding='SAME', regu=regu, reuse=False)

    ############################################# Transition Layer(2)
    ## input: 16 * 16 * 800 output: 8 * 8 * 800
    x = denseTransition(x, stride_size=1, is_training=is_training, regu=regu, name='Transition2', padding='VALID', reuse=False)

    ############################################# 8 Dense Block (3)
    ## input: 16 * 16 * 416  output: 16 * 16 * 800 (416 + 48 * 8)
    blockCircleNumber = 8
    for ii in range(blockCircleNumber):
        tempName = 'DenseBlock3-{}'.format(str(ii + 1))
        x = denseBlock(x, stride_size=1, is_training=is_training, output_channel=48, name=tempName,
                       padding='SAME', regu=regu, reuse=False)

    ############################################ Classification Layer
    ## input: 8 * 8 * 800 output: 800
    x = myAvgPool(x, ksize=8, stride_size=1, name='Classification-avgpool',
                  padding='VALID', reuse=False)



    input_channel = x.get_shape()[-1].value
    x = tf.reshape(x, [-1, input_channel], name='reshape')
    print_Layer(x)

    #x = myFc(x, output_channel=512, regu=regu, name='Fc1', act=tf.nn.relu, reuse=False)

    logist = myFc(x, output_channel=11,
                  name='Fc-logist', act=None, reuse=False)

    return logist

######### 验证加入BN不会影响收敛，反而加快收敛
def defBNNet(input_tensor, is_training, regu = 0.001, reuse=False):
    conv1 = denseConv(input_tensor, conv_size=3, stride_size=1, is_training=is_training, output_channel=32,
                      name='conv1', regu=regu, padding='SAME', act=tf.nn.relu,
                      reuse=reuse)
    conv1_p1 = myMaxPooling2D(conv1, ksize=2, strides=2, padding='VALID', name='conv1_p1',
                              reuse=reuse)

    conv2 = denseConv(conv1_p1, conv_size=3, stride_size=1, is_training=is_training, output_channel=64, name='conv2',
                      padding='SAME',regu=regu,  act=tf.nn.relu,
                      reuse=reuse)
    conv2_p2 = myMaxPooling2D(conv2, ksize=2, strides=2, padding="VALID", name='conv2_p2', reuse=reuse)

    conv3 = denseConv(conv2_p2, conv_size=3, stride_size=1, is_training=is_training, output_channel=128, name='conv3',
                      padding='SAME', regu=regu, act=tf.nn.relu, reuse=reuse)
    conv3_p3 = myMaxPooling2D(conv3, ksize=2, strides=2, padding='VALID', name='conv3_p3', reuse=reuse)

    flatten_conv = tf.contrib.layers.flatten(conv3_p3)
    print_Layer(flatten_conv)

    fc4 = myFc(flatten_conv, output_channel=128, name='fc4')
    # fc5_drop = myDropout( fc5, 0.8, name = 'fc5_drop' )
    # fc5 = myFc(fc4, output_channel=256, name='fc6')
    # fc6_drop = myDropout( fc6, 0.8, name = 'fc6_drop' )
    logits = myFc(fc4, output_channel=11, name='logits', act=None)
    return logits

if __name__ == "__main__":
    # def train():
    # tf.app.flags.DEFINE_boolean("fake_data", False, "If true, uses fake data for unit testing.")
    tf.app.flags.DEFINE_integer('max_steps', 10, 'Number of steps to run trainer.')
    tf.app.flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
    tf.app.flags.DEFINE_string('summaries_dir', 'summaries_denseNet_Regu_shuffle2/', 'Summaries directory')
    tf.app.flags.DEFINE_integer('epoch_iter', 1000, 'number of train iterations for each epoch')
    tf.app.flags.DEFINE_integer('epoch_iter_val', 360, 'number of val iterations for each epoch')
    tf.app.flags.DEFINE_integer('batch_size', 32, 'batch size')
    tf.app.flags.DEFINE_integer('height', 64, 'target height')
    tf.app.flags.DEFINE_integer('width', 64, 'target_width')

    FLAGS = tf.app.flags.FLAGS

    os.chdir('../')
    current_dir = os.getcwd()

    data_dir = current_dir + '/data'
    train_data_dir = data_dir + '/train'
    test_data_dir = data_dir + '/test'
    label_data_dir = current_dir + '/DatasetA_train_20180813/trainWordVector100d.txt'

    trainData = LoadDatas(train_data_dir, targetHeight=FLAGS.height, targetWidth=FLAGS.width, labelDataDir=label_data_dir, flag='train')


    testData = LoadDatas(test_data_dir, targetHeight=FLAGS.height, targetWidth=FLAGS.width, labelDataDir=label_data_dir, flag='test')


    ###  tensorboard 显示时所需要的数据的输出路径
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    if not os.path.exists(current_dir + '/' + FLAGS.summaries_dir):
        os.makedirs(current_dir + '/' + FLAGS.summaries_dir)

    ######## 定义网络
    ##  网络的输入
    with tf.variable_scope('input'):
        inputs = tf.placeholder(tf.float32, shape=[None, FLAGS.height, FLAGS.width, 3], name='inputs')
        tf.summary.image('image', inputs, max_outputs=FLAGS.batch_size)
        y_ = tf.placeholder(tf.int16, shape=[None, 11], name='y-input')
        ##将每一个属性按照不同比例，划分为11个类别, 0.0 0.1 0.2 0.3 ... 1.0, 然后采用one_Hot编码
        is_training = tf.placeholder(tf.bool, name='is_training')

    print("################################## test DenseNet121#############################")
    print_Layer(inputs)

    log_dir = FLAGS.summaries_dir

    logits = defmyDensetNet2(inputs, is_training)
    # logits = defDenseNet121(inputs)
    # logits = defNet(inputs)  ### 收敛
    #logits = defBNNet(inputs, is_training)  ### 更快收敛
    ############################  定义loss , 精度，, 训练OP
    with tf.variable_scope('prediction'):
        y = tf.nn.softmax(logits, name='prediction')

    with tf.variable_scope('loss'):
        # y = tf.clip_by_value(y, 0.005, 0.995)
        loss_p = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=y_, logits=logits))
        tf.add_to_collection('losses', loss_p)
        loss = tf.add_n(tf.get_collection('losses'))
        tf.summary.scalar('loss', loss)

    with tf.variable_scope('accuracy'):
        accuracy_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        acc = tf.reduce_mean(tf.cast(accuracy_prediction, tf.float32))
        tf.summary.scalar('acc', acc)

    with tf.variable_scope('train_op'):
        train_step = tf.train.MomentumOptimizer(learning_rate=0.001, momentum=0.9, use_nesterov=True).minimize(loss)

    #########################  保存tensorboard 所需要的数据
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir + '/train', tf.get_default_graph())
    test_writer = tf.summary.FileWriter(log_dir + '/test')


    #########################  保存训练的model, 如果存在则删除
    modelDir = data_dir + '/model-myDenseNet2-epoch10-100d'
    if tf.gfile.Exists(modelDir):
        tf.gfile.DeleteRecursively(modelDir)
    if not os.path.exists(modelDir):
        os.mkdir(modelDir)

    #saver = tf.train.Saver(max_to_keep=100000)

    maxMinLoss = 10000
    colunmTotalNumber = trainData.labelArray.shape[1]
    print('columnTotalNumber is {}'.format(colunmTotalNumber))


    with tf.Session() as sess:




        ### 如果给出的一个epoch的迭代次数过小，则自动计算每个epoch迭代的次数

        FLAGS.epoch_iter = int(trainData.length / FLAGS.batch_size) + 100
        global_step = FLAGS.max_steps

        for columnNumber in range(0, colunmTotalNumber ):

            if len(set(trainData.labelArray[:, columnNumber])) == 1:
                for ii in range(5):
                    print("#############################################################################")
                print("columnNumber {} has only one value".format(columnNumber))
                continue

            init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init_op)
            saver = tf.train.Saver(max_to_keep=3)

            minValLoss = maxMinLoss
            if not os.path.exists(modelDir+'/' + str(columnNumber)):
                os.mkdir(modelDir+'/' + str(columnNumber))
            f = open(modelDir + '/' + str(columnNumber) + '/' + str(columnNumber) + '-trainLog.txt', 'w')

            for epoch in range(global_step):

                train_loss, train_acc,  n_batch_train = 0., 0., 0.
                for epoch_iter in range(FLAGS.epoch_iter):
                    x_trains, y_trains = trainData.next_batch(columnNumber=columnNumber, batch_size=FLAGS.batch_size)
                    summary, _, err, ac= sess.run([merged, train_step, loss, acc],
                                                   feed_dict={inputs: x_trains, y_: y_trains, is_training: True})
                    train_loss += err
                    train_acc += ac
                    n_batch_train += 1
                    train_writer.add_summary(summary, epoch * FLAGS.epoch_iter + epoch_iter + 1)
                    print("\t", end='')
                    print("currentIter: loss:{:.6f}, acc:{:.6f}".format(err, ac))
                    f.write( "\t" + "currentIter: loss:{:.6f}, acc:{:.6f}".format(err, ac) + '\n' )
                    print("epoch{}:iter{}, loss:{:.6f}, acc:{:.6f}".format(epoch + 1, epoch_iter + 1,
                                                                           train_loss / (epoch_iter + 1),
                                                                           train_acc / (epoch_iter + 1)))
                    f.write( "epoch{}:iter{}, loss:{:.6f}, acc:{:.6f}".format(epoch + 1, epoch_iter + 1,
                                                                           train_loss / (epoch_iter + 1),
                                                                           train_acc / (epoch_iter + 1)) + '\n')

                ############################# val #####################
                FLAGS.epoch_iter_val = int(testData.length / FLAGS.batch_size)
                val_loss, val_acc, n_batch = 0.,  0., 0.
                testData.counter = 0
                epoch_iter_val = 0
                for epoch_iter_val in range(FLAGS.epoch_iter_val):
                    x_vals, y_vals = testData.next_batch(columnNumber=columnNumber, batch_size=FLAGS.batch_size)
                    summary, err, ac = sess.run([merged, loss, acc],
                                                feed_dict={inputs: x_vals, y_: y_vals, is_training: False})
                    val_loss += err
                    val_acc += ac
                    n_batch += 1
                    print("\t", end='')
                    print("test: currentIter: loss:{:.6f}, acc:{:.6f}".format(err, ac))
                    f.write('\t' + "test: currentIter: loss:{:.6f}, acc:{:.6f}".format(err, ac) + '\n')
                    print("epoch:{}, iter:{}, loss: {:.6f}, acc:{:.6f}".format(epoch + 1, epoch_iter_val + 1,
                                                                               val_loss / (epoch_iter_val + 1),
                                                                               val_acc / (epoch_iter_val + 1)))
                    f.write( "epoch:{}, iter:{}, loss: {:.6f}, acc:{:.6f}".format(epoch + 1, epoch_iter_val + 1,
                                                                               val_loss / (epoch_iter_val + 1),
                                                                               val_acc / (epoch_iter_val + 1)) + '\n' )

                ################# 只保存val_loss 下降的训练参数
                if minValLoss > (val_loss / (epoch_iter_val + 1)):
                    minValLoss = (val_loss / (epoch_iter_val + 1))
                    saveModelDir = modelDir + '/' + str(columnNumber) + '/model'+ str(columnNumber) + '-' + str(epoch + 1) + '.ckpt'
                    saver.save(sess, saveModelDir, global_step=epoch * FLAGS.epoch_iter + epoch_iter + 1)
                test_writer.add_summary(summary, epoch * FLAGS.epoch_iter_val + epoch_iter_val + 1)
            f.close()









