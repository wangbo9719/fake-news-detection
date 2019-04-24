'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from method import method
import pickle
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from tensorflow.contrib import learn

tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos",
                       "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg",
                       "Data source for the negative data.")

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding (default: 128)")
tf.flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes (default: '3,4,5')")
tf.flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size (default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps (default: 100)")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store (default: 5)")
# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS

class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # embedding_table就是词向量表
            self.W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), #随机初始化词向量表
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            # 这个tf.nn.embedding_lookup()的作用就是从词向量表中去找input_x所对应的词向量；
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            # 由于CNN输入都是四维，所以在最后一维添加一个维度，与CNN的输入维度对照起来。

        # Create a convolution + maxpool layer for each filter size
        #对不同窗口尺寸的过滤器都创造一个卷积层和池化层
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes): #遍历卷积核的各个size
            with tf.name_scope("conv-maxpool-%s" % filter_size):#建立一个名魏conv-maxpool的模块
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                # 卷积核参数：高*宽*通道*卷积核个数
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                # W表示截断产生正态分布数据，方差为0.1，变量维度为filter_shape的张量
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                # b表示变量维度为卷积核个数，数值为0.1的张量
                conv = tf.nn.conv2d(#实现卷积
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)
                # 一个pooled是一种卷积核处理一个样本之后得到的一个值，如果有三种卷积核，则append了三次

        # Combine all the pooled features，将max-pooling层的各种特征整合在一起
        # 每种卷积核的个数与卷积核的种类乘积，等于全部的卷积核个数
        num_filters_total = num_filters * len(filter_sizes)
        # 将pooled_outputs在第四维度上进行拼接
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout ，随机失活，缓解过拟合
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions，产生最后的预测和输出
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss，定义模型的损失函数
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy，定义模型的准确率
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

class MethodTextCNN(method):
    fold = None
    sample_ratio = None
    data_type = None
    data = None
    learning_rate = 0.01

    def preprocess(self):  # 打乱样本；分为训练集和测试集两部分
        # Data Preparation
        # ==================================================

        # Load data
        '''
        print("Loading data...")
        content_dic = open('./data_samples/article_content_dic','rb')
        credi_dic = open('./data_samples/article_credibility_dic', 'rb')
        content = list(pickle.load(content_dic).values()) #内容列表
        credi = list(pickle.load(credi_dic).values()) #rating列表  顺序是对应的
        #label处理为0，1值
        for i,_ in enumerate(credi):
            if credi[i] > 3: #0为正
                credi[i] = [0,1]
            elif credi[i] < 4:
                credi[i] = [1, 0]
            else:
                print(i,credi[i])

        content_credi_all = dict()
        for i, _ in enumerate(credi):
            content_credi = dict()
            content_credi['content'] = content[i]
            content_credi['credi'] = credi[i]
            content_credi_all[i] = content_credi

        pos_con = []
        neg_con = []
        pos_credi = []
        neg_credi = []
        for i,_ in enumerate(content_credi_all):
            if content_credi_all[i]['credi'][0] == 0:
                pos_con.append(content_credi_all[i]['content'])
                pos_credi.append([0,1])
            else:
                neg_con.append(content_credi_all[i]['content'])
                neg_credi.append([1,0
                                  ])


        y = np.concatenate([pos_credi, neg_credi], 0)
        x_text = pos_con + neg_con
        print(x_text)
        print(len(x_text))
        print(y)
        print(len(y))
        # Build vocabulary
        max_document_length = max([len(x.split(" ")) for x in x_text])
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        #learn.preprocessing.VocabularyProcessor, 根据所有已分词好的文本建立好一个词典，然后找出每个词在词典中对应的索引，不足长度或者不存在的词补0
        x = np.array(list(vocab_processor.fit_transform(x_text)))

        # Randomly shuffle data
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        print('shuffle_indices:',shuffle_indices)
        x_shuffled = x[shuffle_indices]
        y_shuffled = np.array(y)[shuffle_indices]

        # Split train/test set
        # TODO: This is very crude, should use cross-validation
        dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
        x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
        y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

        del x, y, x_shuffled, y_shuffled

        print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
        print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
        #print("x_train:",x_train,"\n y_train:",y_train)
        return x_train, y_train, vocab_processor, x_dev, y_dev
        '''

    def train(self,x_train, y_train, vocab_processor, x_dev, y_dev):
        # Training
        # ==================================================

        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=FLAGS.allow_soft_placement,
                log_device_placement=FLAGS.log_device_placement)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                cnn = TextCNN(
                    sequence_length=x_train.shape[1],
                    num_classes=y_train.shape[1],
                    vocab_size=len(vocab_processor.vocabulary_),
                    embedding_size=FLAGS.embedding_dim,
                    filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                    num_filters=FLAGS.num_filters,
                    l2_reg_lambda=FLAGS.l2_reg_lambda)

                # Define Training procedure
                global_step = tf.Variable(0, name="global_step", trainable=False)
                optimizer = tf.train.AdamOptimizer(1e-3)
                grads_and_vars = optimizer.compute_gradients(cnn.loss)
                train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

                # Keep track of gradient values and sparsity (optional)
                grad_summaries = []
                for g, v in grads_and_vars:
                    if g is not None:
                        grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                        sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                        grad_summaries.append(grad_hist_summary)
                        grad_summaries.append(sparsity_summary)
                grad_summaries_merged = tf.summary.merge(grad_summaries)

                # Output directory for models and summaries
                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
                print("Writing to {}\n".format(out_dir))

                # Summaries for loss and accuracy
                loss_summary = tf.summary.scalar("loss", cnn.loss)
                acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

                # Train Summaries
                train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
                train_summary_dir = os.path.join(out_dir, "summaries", "train")
                train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

                # Dev summaries
                dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
                dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
                dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

                # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
                checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)

                # Write vocabulary
                vocab_processor.save(os.path.join(out_dir, "vocab"))

                # Initialize all variables
                sess.run(tf.global_variables_initializer())

                def train_step(x_batch, y_batch):
                    """
                    A single training step
                    """
                    feed_dict = {
                        cnn.input_x: x_batch,
                        cnn.input_y: y_batch,
                        cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                    }
                    _, step, summaries, loss, accuracy = sess.run(
                        [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                        feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    train_summary_writer.add_summary(summaries, step)

                def dev_step(x_batch, y_batch, writer=None):
                    """
                    Evaluates model on a dev set
                    """
                    feed_dict = {
                        cnn.input_x: x_batch,
                        cnn.input_y: y_batch,
                        cnn.dropout_keep_prob: 1.0
                    }
                    step, summaries, loss, accuracy = sess.run(
                        [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                        feed_dict)
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                    if writer:
                        writer.add_summary(summaries, step)

                # Generate batches
                batches = data_helpers.batch_iter(
                    list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
                # Training loop. For each batch...
                for batch in batches:
                    x_batch, y_batch = zip(*batch)
                    train_step(x_batch, y_batch)
                    current_step = tf.train.global_step(sess, global_step)
                    if current_step % FLAGS.evaluate_every == 0: #每100步
                        print("\nEvaluation:")
                        dev_step(x_dev, y_dev, writer=dev_summary_writer)
                        print("")
                    if current_step % FLAGS.checkpoint_every == 0: #每100步
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))

    def run(self):
        self.start_time = time.time()
        x_train, y_train, vocab_processor, x_dev, y_dev = self.preprocess()
        self.train(x_train, y_train, vocab_processor, x_dev, y_dev)

