'''
Concrete MethodModule class for a specific learning MethodModule
'''

# Copyright (c) 2017 Jiawei Zhang <jwzhanggy@gmail.com>
# License: TBD

from method import method
import collections
import time
import copy
import pickle
import random
from tensorflow.contrib import rnn
#from MethodGDUCells import MethodGDUCell
from tensorflow.python.framework import ops
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
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
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

        # Combine all the pooled features，将max-pooling层的各种特征整合在一起
        num_filters_total = num_filters * len(filter_sizes)
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
    n_hidden_rnn_layer1 = None
    n_hidden_rnn_layer2 = None
    n_hidden_gdu = None

    dictionary = None
    reverse_dictionary = None
    vocab_size = None
    max_string_length = None

    article_id_index_dict = None
    article_index_id_dict = None
    creator_id_index_dict = None
    creator_index_id_dict = None
    subject_id_index_dict = None
    subject_index_id_dict = None

    article_train_set = None
    article_test_set = None
    creator_train_set = None
    creator_test_set = None
    subject_train_set = None
    subject_test_set = None

    article_credibility_dict = None
    creator_credibility_dict = None
    subject_credibility_dict = None

    AS_matrix = None
    AC_matrix = None
    padding_x = None

    article_seed_batch_index = None
    creator_seed_batch_index = None
    subject_seed_batch_index = None

    def build_name_index_dict(self, input_dict):
        dictionary = {}
        reverse_dictionary = {}
        index = 0
        for idd in input_dict:
            dictionary[idd] = index
            reverse_dictionary[index] = idd
            index += 1
        return dictionary, reverse_dictionary

    def build_dataset(self, word_list):
        self.article_credibility_dict = self.data['setting']['ground_truth']['artilcle_crediblity_dict']
        self.creator_credibility_dict = self.data['setting']['ground_truth']['creator_credibility_dict']
        self.subject_credibility_dict = self.data['setting']['ground_truth']['subject_credibility_dict']

        self.article_train_set = \
        self.data['setting']['CV_Sampling']['article_sampling_dict'][self.fold][self.sample_ratio]['train']
        self.article_test_set = \
        self.data['setting']['CV_Sampling']['article_sampling_dict'][self.fold][self.sample_ratio]['test']
        self.creator_train_set = \
        self.data['setting']['CV_Sampling']['creator_sampling_dict'][self.fold][self.sample_ratio]['train']
        self.creator_test_set = \
        self.data['setting']['CV_Sampling']['creator_sampling_dict'][self.fold][self.sample_ratio]['test']
        self.subject_train_set = \
        self.data['setting']['CV_Sampling']['subject_sampling_dict'][self.fold][self.sample_ratio]['train']
        self.subject_test_set = \
        self.data['setting']['CV_Sampling']['subject_sampling_dict'][self.fold][self.sample_ratio]['test']

        self.article_id_index_dict, self.article_index_id_dict = self.build_name_index_dict(
            self.article_credibility_dict)
        self.creator_id_index_dict, self.creator_index_id_dict = self.build_name_index_dict(
            self.creator_credibility_dict)
        self.subject_id_index_dict, self.subject_index_id_dict = self.build_name_index_dict(
            self.subject_credibility_dict)

        count = collections.Counter(word_list).most_common()
        self.dictionary = dict()
        for word, _ in count:
            self.dictionary[word] = len(self.dictionary)
        self.reverse_dictionary = dict(zip(self.dictionary.values(), self.dictionary.keys()))
        self.vocab_size = len(self.dictionary)

        self.padding_x = self.dictionary['dummy_word_that_will_never_appear']

    def get_word__dict(self):
        word_list = []
        self.max_string_length = 0
        length_list = []
        for article in self.data['node']['article']:
            content = self.data['node']['article'][article]['content']
            words = content.lower().split(' ')
            if len(words) > 100:
                words = words[:100]
            word_list.extend(words)
            if len(words) > self.max_string_length:
                self.max_string_length = len(words)
            length_list.append(len(words))

        for creator in self.data['node']['creator']:
            content = self.data['node']['creator'][creator]['profile']
            words = content.lower().split(' ')
            if len(words) > 100:
                words = words[:100]
            word_list.extend(words)
            if len(words) > self.max_string_length:
                self.max_string_length = len(words)
            length_list.append(len(words))

        for subject in self.data['link']['subject_article']:
            words = [subject.lower()]
            word_list.extend(words)

        word_list.extend(['dummy_word_that_will_never_appear'] * 100000)
        return word_list

    def batch_generation(self):
        article_train_X = []
        article_train_y = []
        for article in self.article_train_set:
            x = self.entity_feature_dict[article]
            y = self.article_credibility_dict[article]
            article_train_X.append(x)
            article_train_y.append(y)
        article_test_X = []
        article_test_y = []
        for article in self.article_test_set:
            x = self.entity_feature_dict[article]
            y = self.article_credibility_dict[article]
            article_test_X.append(x)
            article_test_y.append(y)

        creator_train_X = []
        creator_train_y = []
        for creator in self.creator_train_set:
            x = self.entity_feature_dict[creator]
            y = self.creator_credibility_dict[creator]
            creator_train_X.append(x)
            creator_train_y.append(y)
        creator_test_X = []
        creator_test_y = []
        for creator in self.creator_test_set:
            x = self.entity_feature_dict[creator]
            y = self.creator_credibility_dict[creator]
            creator_test_X.append(x)
            creator_test_y.append(y)

        subject_train_X = []
        subject_train_y = []
        for subject in self.subject_train_set:
            x = self.entity_feature_dict[subject]
            y = self.subject_credibility_dict[subject]
            subject_train_X.append(x)
            subject_train_y.append(y)
        subject_test_X = []
        subject_test_y = []
        for subject in self.subject_test_set:
            x = self.entity_feature_dict[subject]
            y = self.subject_credibility_dict[subject]
            subject_test_X.append(x)
            subject_test_y.append(y)
        subject_train_y[-1] = 0

        return article_train_X, article_train_y, article_test_X, article_test_y, creator_train_X, creator_train_y, creator_test_X, creator_test_y, subject_train_X, subject_train_y, subject_test_X, subject_test_y
    def preprocess(self):  # 打乱样本；分为训练集和测试集两部分
        # Data Preparation
        # ==================================================

        # Load data
        print("Loading data...")
        #x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
        x_text, y = self.data.article_test_set, self.article_credibility_dict
        # Build vocabulary
        max_document_length = max([len(x.split(" ")) for x in x_text])
        vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
        x = np.array(list(vocab_processor.fit_transform(x_text)))

        # Randomly shuffle data
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = x[shuffle_indices]
        y_shuffled = y[shuffle_indices]

        # Split train/test set
        # TODO: This is very crude, should use cross-validation
        dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
        x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
        y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

        del x, y, x_shuffled, y_shuffled

        print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
        print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
        return x_train, y_train, vocab_processor, x_dev, y_dev

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
                    if current_step % FLAGS.evaluate_every == 0:
                        print("\nEvaluation:")
                        dev_step(x_dev, y_dev, writer=dev_summary_writer)
                        print("")
                    if current_step % FLAGS.checkpoint_every == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model checkpoint to {}\n".format(path))

    def run(self):
        word_list = self.get_word__dict()
        self.build_dataset(word_list)
        #x_train, y_train, vocab_processor, x_dev, y_dev = self.preprocess()
        #self.train(x_train, y_train, vocab_processor, x_dev, y_dev)

