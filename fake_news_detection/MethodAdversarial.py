from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
import time
import string
from utils.prepare_data import *
from utils.model_helper import *
from method import method

class MethodAdversarial(method):
    data_type = None
    learning_rate = 0.01

    data = None
    train_test_divide = None
    article_credibility_dict = None
    sample_ratios = None

    article_train_index_list = None
    article_test_index_list = None

    def build_dataset(self, sample_ratio):
        self.article_train_index_list = self.train_test_divide[sample_ratio]['train']
        self.article_test_index_list = self.train_test_divide[sample_ratio]['test']

    def bi_class_batch_generation(self):

        #-----------------train-----------------------
        for article_train_index in self.article_train_index_list:
            if article_train_index not in self.article_credibility_dict:
                self.article_train_index_list.remove(article_train_index)
        article_train_X = []
        article_train_y = []
        for i in range(len(self.article_train_index_list)):
            article_train_X.append('')
            article_train_y.append(0)
        order_num_train = 0
        for article_train_index in self.article_train_index_list:
            if article_train_index not in self.article_credibility_dict: continue
            y = self.article_credibility_dict[article_train_index]
            if y >= 4:
                y = 1
            else:
                y = 0
            article_train_y[order_num_train] = y
            content = self.data['node']['article'][article_train_index]['content']
            content = content.translate(str.maketrans('', '', string.punctuation))
            article_train_X[order_num_train] = content
            order_num_train += 1

        #-----------------test------------------
        for article_test_index in self.article_test_index_list:
            if article_test_index not in self.article_credibility_dict:
                self.article_test_index_list.remove(article_test_index)
        article_test_X = []
        article_test_y = []
        for i in range(len(self.article_test_index_list)):
            article_test_X.append('')
            article_test_y.append(0)
        order_num_test = 0

        for article_test_index in self.article_test_index_list:
            if article_test_index not in self.article_credibility_dict: continue
            y = self.article_credibility_dict[article_test_index]
            if y >= 4:
                y = 1
            else:
                y = 0
            article_test_y[order_num_test] = y
            content = self.data['node']['article'][article_test_index]['content']
            content = content.translate(str.maketrans('', '', string.punctuation))
            article_test_X[order_num_test] = content
            order_num_test += 1
        return article_train_X, article_train_y, article_test_X, article_test_y


    def para_setting(self, config):
        self.max_len = config["max_len"]
        self.hidden_size = config["hidden_size"]
        self.vocab_size = config["vocab_size"]
        self.embedding_size = config["embedding_size"]
        self.n_class = config["n_class"]
        self.learning_rate = config["learning_rate"]
        self.epsilon = config["epsilon"]

        # placeholder
        self.x = tf.placeholder(tf.int32, [None, self.max_len])
        self.label = tf.placeholder(tf.int32, [None])
        self.keep_prob = tf.placeholder(tf.float32) #dropout

    def scale_l2(self, x, norm_length):
        alpha = tf.reduce_max(tf.abs(x), (1, 2), keepdims=True) + 1e-12
        l2_norm = alpha * tf.sqrt(
            tf.reduce_sum(tf.pow(x / alpha, 2), (1, 2), keepdims=True) + 1e-6)
        x_unit = x / l2_norm
        return norm_length * x_unit

    #对vk正则化， emb是词k的词嵌入表示，weights是这个词出现的次数占所有词数的权重
    def normalize(self, emb, weights):
        # weights = vocab_freqs / tf.reduce_sum(vocab_freqs) ?? 这个实现没问题吗
        print("Weights: ", weights)
        mean = tf.reduce_sum(weights * emb, 0, keep_dims=True)
        var = tf.reduce_sum(weights * tf.pow(emb - mean, 2.), 0, keep_dims=True)
        stddev = tf.sqrt(1e-6 + var)
        return (emb - mean) / stddev

    #为嵌入向量增加扰动
    def _add_perturbation(self, embedded, loss):
        """Adds gradient to embedding and recomputes classification loss."""
        grad, = tf.gradients(
            loss,
            embedded,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
        grad = tf.stop_gradient(grad)
        perturb = self.scale_l2(grad, self.epsilon)
        return embedded + perturb

    #获得f
    def _get_freq(self, vocab_freq, word2idx):
        """get a frequency dict format as {word_idx: word_freq}"""
        words = vocab_freq.keys()
        freq = [0] * self.vocab_size
        for word in words:
            word_idx = word2idx.get(word)
            word_freq = vocab_freq[word]
            freq[word_idx] = word_freq
        return freq

    def build_graph(self, vocab_freq, word2idx):
        vocab_freqs = tf.constant(self._get_freq(vocab_freq, word2idx),
                                  dtype=tf.float32, shape=(self.vocab_size, 1))
        weights = vocab_freqs / tf.reduce_sum(vocab_freqs)
        # embeddings_var为值在-1到1的vocab_size * embedding_size的矩阵
        # 即每个词w的向量v构成的矩阵
        embeddings_var = tf.Variable(tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                                     trainable=True, name="embedding_var")
        # 正则化后结果
        embedding_norm = self.normalize(embeddings_var, weights)
        # 找到x对应的嵌入
        batch_embedded = tf.nn.embedding_lookup(embedding_norm, self.x)

        W = tf.Variable(tf.random_normal([self.hidden_size], stddev=0.1))
        W_fc = tf.Variable(tf.truncated_normal([self.hidden_size, self.n_class], stddev=0.1))
        b_fc = tf.Variable(tf.constant(0., shape=[self.n_class]))

        def cal_loss_logit(embedded, keep_prob, reuse=True, scope="loss"):
            with tf.variable_scope(scope, reuse=reuse) as scope:
                rnn_outputs, _ = bi_rnn(BasicLSTMCell(self.hidden_size),
                                        BasicLSTMCell(self.hidden_size),
                                        inputs=embedded, dtype=tf.float32) #RNN预训练？

                # Attention
                H = tf.add(rnn_outputs[0], rnn_outputs[1])  # fw + bw
                M = tf.tanh(H)  # M = tanh(H)  (batch_size, seq_len, HIDDEN_SIZE)
                # alpha (bs * sl, 1)
                alpha = tf.nn.softmax(tf.matmul(tf.reshape(M, [-1, self.hidden_size]),
                                                tf.reshape(W, [-1, 1])))
                r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(alpha, [-1, self.max_len,
                                                                             1]))  # supposed to be (batch_size * HIDDEN_SIZE, 1)
                r = tf.squeeze(r)
                h_star = tf.tanh(r)
                drop = tf.nn.dropout(h_star, keep_prob)

                # Fully connected layer（dense layer)
                y_hat = tf.nn.xw_plus_b(drop, W_fc, b_fc)

            return y_hat, tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y_hat, labels=self.label))

        logits, self.cls_loss = cal_loss_logit(batch_embedded, self.keep_prob, reuse=False)
        embedding_perturbated = self._add_perturbation(batch_embedded, self.cls_loss)
        adv_logits, self.adv_loss = cal_loss_logit(embedding_perturbated, self.keep_prob, reuse=True)
        self.loss = self.cls_loss + self.adv_loss

        # optimization
        loss_to_minimize = self.loss
        tvars = tf.trainable_variables()
        gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)
        grads, global_norm = tf.clip_by_global_norm(gradients, 1.0)

        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step,
                                                       name='train_step')
        self.prediction = tf.argmax(tf.nn.softmax(logits), 1)

        print("graph built successfully!")

    def run(self):

        for ratio in self.sample_ratios:
            print("ratio:",ratio)
            self.build_dataset(ratio)
            x_train, y_train, x_test, y_test = self.bi_class_batch_generation()
            # data preprocessing
            x_train, x_test, vocab_freq, word2idx, vocab_size = \
                data_preprocessing_with_dict(x_train, x_test, max_len=128)  # 32
            print("train size: ", len(x_train))
            print("vocab size: ", vocab_size)
            # split dataset to test and dev
            x_test, x_dev, y_test, y_dev, dev_size, test_size = \
                split_dataset(x_test, y_test, 0.1)  # 0.1

            print("Validation Size: ", dev_size)
            config = {
                "max_len": 128,  # 32
                "hidden_size": 64,
                "vocab_size": vocab_size,
                "embedding_size": 128,  # 128
                "n_class": 15,
                "learning_rate": 1e-3,
                "batch_size": 32,
                "train_epoch": 10,  # 10
                "epsilon": 5,  # 5
            }
            classifier = self
            self.para_setting(config)
            classifier.build_graph(vocab_freq, word2idx)

            # auto GPU growth, avoid occupy all GPU memory
            tf_config = tf.ConfigProto()
            tf_config.gpu_options.allow_growth = True
            sess = tf.Session(config=tf_config)

            sess.run(tf.global_variables_initializer())
            dev_batch = (x_dev, y_dev)
            start = time.time()
            for e in range(config["train_epoch"]):

                t0 = time.time()
                print("Epoch %d start !" % (e + 1))
                for x_batch, y_batch in fill_feed_dict(x_train, y_train, config["batch_size"]):
                    return_dict = run_train_step(classifier, sess, (x_batch, y_batch))

                t1 = time.time()

                print("Train Epoch time:  %.3f s" % (t1 - t0))
                dev_acc = run_eval_step(classifier, sess, dev_batch)
                print("validation accuracy: %.3f " % dev_acc)

            print("Training finished, time consumed : ", time.time() - start, " s")
            print("Start evaluating:  \n")
            cnt = 0
            test_acc = 0
            for x_batch, y_batch in fill_feed_dict(x_test, y_test, config["batch_size"]):
                acc = run_eval_step(classifier, sess, (x_batch, y_batch))
                test_acc += acc
                cnt += 1

            print("Test accuracy : %f %%" % (test_acc / cnt * 100))


