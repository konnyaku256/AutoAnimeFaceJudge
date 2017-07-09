import tweepy
import urllib.request
import datetime
import re
import cv2
import numpy as np
import tensorflow as tf

NUM_CLASSES = 6
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*3

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

# login情報
# config.txtにTwitterApplicationの4つのキーを設定しています
f = open('config.txt')
data = f.read()
f.close()
lines = data.split('\n')

# アニメ顔検出器
cascade = cv2.CascadeClassifier("./lbpcascade_animeface.xml")
if cascade.empty():
    raise (Exception, 'cascade not found')

images_placeholder = tf.placeholder("float", shape=(None, IMAGE_PIXELS))
labels_placeholder = tf.placeholder("float", shape=(None, NUM_CLASSES))
keep_prob = tf.placeholder("float")

logits = inference(images_placeholder, keep_prob)
sess = tf.InteractiveSession()

saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())
saver.restore(sess, "./model.ckpt")

def get_oauth():
	consumer_key = lines[0]
	consumer_secret = lines[1]
	access_key = lines[2]
	access_secret = lines[3]
	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_key, access_secret)
	return auth

class StreamListener(tweepy.StreamListener):
    def on_status(self, status):
        if status.in_reply_to_screen_name=='konnyaku256':
            if 'media' in status.entities :
                text = re.sub(r'@konnyaku256 ', '', status.text)
                text = re.sub(r'(https?|ftp)(://[\w:;/.?%#&=+-]+)', '', text)
                medias = status.entities['media']
                m =  medias[0]
                media_url = m['media_url']
                print (media_url)
                now = datetime.datetime.now()
                time = now.strftime("%H%M%S")
                filename = '{}.jpg'.format(time)
                try:
                    urllib.request.urlretrieve(media_url, filename)
                except IOError:
                    print ("保存に失敗しました")

                frame = cv2.imread(filename)
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                height, width = img.shape[:2]
                faces = cascade.detectMultiScale(frame, 
                                         scaleFactor = 1.1,
                                         minNeighbors = 4,
                                         minSize = (24, 24))
                
                flag = True
                # 顔が見つかった場合は顔領域だけについて判定
                if len(faces) > 0:
                    flag = False
                    (x, y, w, h) = faces[0] # 一番大きいものだけを調べる
                    # 顔の領域のチェック
                    if y < 0 or y+h > height or x < 0 or x+w > width:
                        flag = True
                    else:
                        # 顔部分を白枠で囲む
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)
                        cv2.imwrite(filename, frame)

                if flag: # 顔が見つからない場合には全体について判定する
                    image = frame
                    cv2.imwrite("original.jpg", image)
                # 形式を変換
                img = cv2.resize(img.copy(), (28, 28))
                ximage = img.flatten().astype(np.float32)/255.0

                pred = np.argmax(logits.eval(feed_dict={
                    images_placeholder: [ximage],
                    keep_prob: 1.0 })[0])
                    
                if pred==0:
                    print ("涼風青葉")
                    name = '涼風青葉'
                elif pred==1:
                    print ("篠田はじめ")
                    name = '篠田はじめ'
                elif pred==2:
                    print ("滝本ひふみ")
                    name = '滝本ひふみ'
                elif pred==3:
                    print ("八神コウ")
                    name = '八神コウ'
                elif pred==4:
                    print ("遠山りん")
                    name = '遠山りん'
                else:
                    print ("飯島ゆん")
                    name = '飯島ゆん'
                    
                # リプライ画像送信者が自分の場合は判定結果を自分のツイートとして投稿
                if status.author.screen_name=='konnyaku256':
                    try:
                        # 画像をつけてツイート
                        message = name
                        api.update_with_media(filename, status=message)
                    except tweepy.error.TweepError as e:
                        print ("error response code: " + str(e.response.status))
                        print ("error message: " + str(e.response.reason))
                else:
                    try:
                        # 画像をつけてリプライ
                        message = '@'+status.author.screen_name+''+name
                        api.update_with_media(filename, status=message, in_reply_to_status_id=status.id)
                    except tweepy.error.TweepError as e:
                        print ("error response code: " + str(e.response.status))
                        print ("error message: " + str(e.response.reason))

# streamingを始めるための準備
auth = get_oauth()
api = tweepy.API(auth)
stream = tweepy.Stream(auth, StreamListener(), secure=True)
print ("Start Streaming!")
stream.userstream()