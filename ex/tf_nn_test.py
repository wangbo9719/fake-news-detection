import tensorflow as tf

# 1.加载数据
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# 2.定义超参数和placeholder
# 超参数
learning_rate = 0.5
epochs = 10
batch_size = 100
#placeholder，一个计算图可以参数化的接收外部的输入，作为一个placeholders(占位符)
#输入图片为28×28个像素=784, 输出为0--9的one-hot编码
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

# 3.定义参数w和b  通过random_normal函数来从正态分布中随机初始化参数w b
# hidden layer --> w1,b1
w1 = tf.Variable(tf.random_normal([784, 300],stddev=0.03,name='w1'))
b1 = tf.Variable(tf.random_normal([300], name = 'b1'))
# output layer --> w2, b2
w2 = tf.Variable(tf.random_normal([300, 10], stddev=0.03, name='w2'))
b2 = tf.Variable(tf.random_normal([10]), name='b2')

#4. 构造隐层网络
# hidden layer
hidden_out = tf.add(tf.matmul(x, w1),b1)
hidden_out = tf.nn.relu(hidden_out)

#5.构造输出（预测值） 计算输出
y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, w2), b2))

#6. bp部分，定义loss
# 这里损失为交叉熵
y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999) #对预测值y_进行压缩

cross_entropy = -tf.reduce_mean(tf.reduce_sum(y *tf.log(y_clipped) + (1 - y)*tf.log(1-y_clipped), axis=1))

#7. bp部分， 定义优化算法
# 创建优化器，确定优化目标
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cross_entropy)

#8. 定义初始化operation和准确率node
init_op = tf.global_variables_initializer()
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) #这里y和y_是n个1×10维的矩阵，因为是onehot编码，只有0，1
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#9. 开始训练
#创建session会话
with tf.Session() as sess:
    sess.run(init_op)
    total_batch = int(len(mnist.train.labels) / batch_size)
    for epochs in range(epochs):
        avg_cast = 0
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size = batch_size)
            _, c = sess.run([optimizer, cross_entropy], feed_dict={x:batch_x, y:batch_y})
            avg_cast += c/total_batch
        print("Epoch:", (epochs+1),",Cost:", avg_cast)
    print("____________________________")
    print(sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels}))










