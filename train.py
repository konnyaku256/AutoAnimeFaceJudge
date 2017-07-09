import os
import cv2
import numpy as np
import tensorflow as tf

path=os.getcwd()+'/data/'
class_count = 0
folder_list=os.listdir(path)

for folder in folder_list:
  class_count = class_count+1

NUM_CLASSES = class_count
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*3

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('label', 'label.txt', 'File name of label')
flags.DEFINE_string('train_dir', './', 'Directory to put the training data.')
flags.DEFINE_integer('max_steps', 100, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 20, 'Batch size'
                     'Must divide evenly into the dataset sizes.')
tf.app.flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')

# 予測モデルを作成する関数
def inference(images_placeholder, keep_prob):
    # 重みを標準偏差0.1の正規分布で初期化
    def weight_variable(shape):
      initial = tf.truncated_normal(shape, stddev=0.1)
      return tf.Variable(initial)

    # バイアスを標準偏差0.1の正規分布で初期化
    def bias_variable(shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    # 畳み込み層の作成
    def conv2d(x, W):
      return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    # プーリング層の作成
    def max_pool_2x2(x):
      return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')
    
    # 入力を28x28x3に変形
    x_image = tf.reshape(images_placeholder, [-1, 28, 28, 3])

    # 畳み込み層1の作成
    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_variable([5, 5, 3, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    # プーリング層1の作成
    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool_2x2(h_conv1)
    
    # 畳み込み層2の作成
    with tf.name_scope('conv2') as scope:
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    # プーリング層2の作成
    with tf.name_scope('pool2') as scope:
        h_pool2 = max_pool_2x2(h_conv2)

    # 全結合層1の作成
    with tf.name_scope('fc1') as scope:
        W_fc1 = weight_variable([7*7*64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        # dropoutの設定
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 全結合層2の作成
    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([1024, NUM_CLASSES])
        b_fc2 = bias_variable([NUM_CLASSES])

    # ソフトマックス関数による正規化
    with tf.name_scope('softmax') as scope:
        y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # 各ラベルの確率のようなものを返す
    return y_conv

# lossを計算する関数
def loss(logits, labels):
    cross_entropy = -tf.reduce_sum(labels*tf.log(logits))
    tf.summary.scalar("cross_entropy", cross_entropy)
    return cross_entropy

# 訓練のOpを定義する関数
def training(loss, learning_rate):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step

# 正解率(accuracy)を計算する関数
def accuracy(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar("accuracy", accuracy)
    return accuracy

if __name__ == '__main__':
    count=0
    folder_list=os.listdir(path)

    train_image = []
    train_label = []
    test_image = []
    test_label = []
    
    f = open(FLAGS.label, 'w')
    for folder in folder_list:
      subfolder = os.path.join(path,folder)
      file_list = os.listdir(subfolder)

      filemax = 0

      for file in file_list:
        filemax = filemax + 1

      # train : test = 9 : 1
      file_rate = int(filemax/10*9)

      i = 0

      for file in file_list:

        img = cv2.imread('./data/' + folder + '/' + file)
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        if i <= file_rate:
           train_image.append(img.flatten().astype(np.float32)/255.0)
           tmp = np.zeros(NUM_CLASSES)
           tmp[int(count)] = 1
           train_label.append(tmp)
        else:
           test_image.append(img.flatten().astype(np.float32)/255.0)
           tmp = np.zeros(NUM_CLASSES)
           tmp[int(count)] = 1
           test_label.append(tmp)

        i = i + 1

      label_name = folder + '\n'
      f.write(label_name)
      count=count+1
    f.close()

    train_image = np.asarray(train_image)
    train_label = np.asarray(train_label)
    test_image = np.asarray(test_image)
    test_label = np.asarray(test_label)
    
    with tf.Graph().as_default():
        # 画像を入れる仮のTensor
        images_placeholder = tf.placeholder("float", shape=(None, IMAGE_PIXELS))
        # ラベルを入れる仮のTensor
        labels_placeholder = tf.placeholder("float", shape=(None, NUM_CLASSES))
        # dropout率を入れる仮のTensor
        keep_prob = tf.placeholder("float")

        # inference()を呼び出してモデルを作る
        logits = inference(images_placeholder, keep_prob)
        # loss()を呼び出して損失を計算
        loss_value = loss(logits, labels_placeholder)
        # training()を呼び出して訓練
        train_op = training(loss_value, FLAGS.learning_rate)
        # 精度の計算
        acc = accuracy(logits, labels_placeholder)

        # 保存の準備
        saver = tf.train.Saver()
        # Sessionの作成
        sess = tf.Session()
        # 変数の初期化
        sess.run(tf.initialize_all_variables())
        # TensorBoardで表示する値の設定
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
        
        
        # 訓練の実行
        for step in range(FLAGS.max_steps):
            for i in range(int(len(train_image)/FLAGS.batch_size)):
                # batch_size分の画像に対して訓練の実行
                batch = FLAGS.batch_size*i
                # feed_dictでplaceholderに入れるデータを指定する
                sess.run(train_op, feed_dict={
                  images_placeholder: train_image[batch:batch+FLAGS.batch_size],
                  labels_placeholder: train_label[batch:batch+FLAGS.batch_size],
                  keep_prob: 0.5})

            # 1 step終わるたびに精度を計算する
            train_accuracy = sess.run(acc, feed_dict={
                images_placeholder: train_image,
                labels_placeholder: train_label,
                keep_prob: 1.0})
            print ("step %d, training accuracy %g"%(step, train_accuracy))

            # 1 step終わるたびにTensorBoardに表示する値を追加する
            summary_str = sess.run(summary_op, feed_dict={
                images_placeholder: train_image,
                labels_placeholder: train_label,
                keep_prob: 1.0})
            summary_writer.add_summary(summary_str, step)

    # 訓練が終了したらテストデータに対する精度を表示
    print ("test accuracy %g"%sess.run(acc, feed_dict={
        images_placeholder: test_image,
        labels_placeholder: test_label,
        keep_prob: 1.0}))

    # 最終的なモデルを保存
    save_path = saver.save(sess, "./model.ckpt")