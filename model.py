import numpy as np
import pandas as pd
import tensorflow as tf
import data
import cv2
from sklearn.utils import shuffle
from tqdm import tqdm
import os

train_frames = 20400
test_frames = 10798

seed = [1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144]

def train(train_data, val_data):
    # Python optimisation variables
    learning_rate = 0.0001
    batch_size = 16
    steps = 400
    epochs = 85

    x = tf.placeholder(tf.float32, [None, 66, 220, 6])
    y = tf.placeholder(tf.float32, [None, 1])


    #create CNN Block

    conv_1 = Conv2D (pad(x,3), 6, 64, [7, 7],2, name='conv1')
    conv_2 = Conv2D( pad(conv_1, 2), 64, 128, [5, 5], 2, name='conv2')

    print(conv_1.shape)
    print(conv_2.shape)

    conv_3 = Conv2D(pad(conv_2,2), 128, 256, [5, 5], 2, name='conv3')
    conv3_1 = Conv2D(pad(conv_3), 256, 256, [3, 3], 1, name='conv3_1')

    print(conv_3.shape)
    print(conv3_1.shape)

    conv4 = Conv2D(pad(conv3_1), 256, 512, [3, 3], 2, name='conv4')

    print(conv4.shape)

    conv4_1 = Conv2D(pad(conv4), 512, 512, [3, 3], 1, name='conv4_1')


    print(conv4_1.shape)

    conv5 = Conv2D(pad(conv4_1), 512, 512, [3, 3], 2, name='conv5')
    conv5_1 = Conv2D(pad(conv5), 512, 512, [3, 3], 1, name='conv5_1')

    conv6 = Conv2D(pad(conv5), 512, 1024, [3, 3], 1, name='conv6')

    #create regression layers
    fc1 = tf.contrib.layers.flatten(conv6)
    fc1 = tf.contrib.layers.fully_connected(fc1, 1000)
    fc2 = tf.contrib.layers.fully_connected(fc1, 1000)

    out = tf.contrib.layers.fully_connected(fc2, 1, activation_fn=None)

    #add loss
    MSE = tf.losses.mean_squared_error(y, out)
    #add optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(MSE)


    #accuracy
    mse_sum = tf.reduce_sum(tf.square(tf.subtract(y, out)))
    accuracy = tf.divide(mse_sum, tf.cast(batch_size, tf.float32))

    init_op = tf.global_variables_initializer()

    tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter(data.DATA_PATH)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init_op)
        for epoch in range(epochs):
            avg_cost = 0
            for step in range(steps):
                for x_batch, y_batch in datagen(train_data, batch_size, 'train'):
                    #x_batch = tf.convert_to_tensor(x_batch, dtype=tf.float32)
                    #y_batch = tf.convert_to_tensor(y_batch, dtype=tf.float32)
                    _,c = sess.run([optimizer, MSE], feed_dict={x: x_batch, y: y_batch})
                avg_cost += c/steps
         #validation

            for x_batch, y_batch in datagen(val_data, batch_size):
                #x_batch = tf.convert_to_tensor(x_batch, dtype=tf.float32)
                #y_batch = tf.convert_to_tensor(y_batch, dtype=tf.float32)
                test_acc = sess.run(accuracy, feed_dict={x: x_batch, y: y_batch})

            print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost),
                      " test accuracy: {:.3f}".format(test_acc))
            summary =sess.run(merged, feed_dict={x: x_batch, y: y_batch})
            writer.add_summary(summary, epoch)
        print("\n Training complete!")
        writer.add_graph(sess.graph)
        save_path = saver.save(sess, os.path.join(data.DATA_PATH, 'model.ckpt'))
        print("Model saved in path: %s" % save_path)


def Conv2D(input_data, num_input_channels, num_filters, filter_shape, stride, name):
    conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels, num_filters]

    weights = tf.Variable(tf.truncated_normal(conv_filt_shape,stddev=0.03), name=name + '_W')

    bias = tf.Variable(tf.truncated_normal([num_filters]), name=name + '_b')

    out_layer = tf.nn.conv2d(input_data, weights, [1,stride,stride,1], padding='SAME')
    out_layer += bias

    out_layer = tf.nn.relu(out_layer)

    return out_layer


def pad(tensor, num=1):
    """
    Pads the given tensor along the height and width dimensions with `num` 0s on each side
    """
    return tf.pad(tensor, [[0, 0], [num, num], [num, num], [0, 0]], "CONSTANT")




def split_val(dataframe, seed):

    train_data = pd.DataFrame()
    valid_data = pd.DataFrame()
    data_size = len(dataframe.index) -1
    for i in tqdm(range(data_size)):
        idx1 = np.random.randint(len(dataframe) - 1)
        idx2 = idx1 + 1

        row1 = dataframe.iloc[[idx1]].reset_index()
        row2 = dataframe.iloc[[idx2]].reset_index()

        randInt = np.random.randint(9)
        if 0 <= randInt <= 1:
            valid_frames = [valid_data, row1, row2]
            valid_data = pd.concat(valid_frames, axis=0, join='outer', ignore_index=False)
        if randInt >= 2:
            train_frames = [train_data, row1, row2]
            train_data = pd.concat(train_frames, axis=0, join='outer', ignore_index=False)
    train_data.to_csv(os.path.join(data.DATA_PATH, 'trainsplit_meta.csv'), index=False)
    valid_data.to_csv(os.path.join(data.DATA_PATH, 'validsplit_meta.csv'), index=False)
    return train_data, valid_data

def brighness_reg(img,bval):

    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img_hsv[:,:,2] = img_hsv[:,:,2] * bval

    image_reg = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

    return  image_reg


def crop(img):
    crop = img[100:440, :-90, :]
    image = cv2.resize(crop, (220, 66), interpolation=cv2.INTER_AREA)
    return image

def preprocess(path, speed,split):

    if split == 'train':
        bval = 0.2 + np.random.uniform()
    else:
        bval = 1.0
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = brighness_reg(img, bval)
    img = crop(img)
    return img, speed


def datagen(data, batch_size = 32, type=''):
    image_batch = np.zeros((batch_size, 66, 220, 6))
    label_batch = np.zeros((batch_size))
    data_size = len(data)
    for i in range(batch_size):
        idx = np.random.randint(1, data_size - 1)

        row_t0 = data.iloc[[idx -1]].reset_index()
        row_t1 = data.iloc[[idx]].reset_index()
        row_t2 = data.iloc[[idx + 1]].reset_index()

        t0 = row_t0['image_index'].values[0]
        t1 = row_t1['image_index'].values[0]
        t2 = row_t2['image_index'].values[0]

        if abs(t1 - t0) == 1 and t1 > t0:
            row1 = row_t0
            row2 = row_t1

        elif abs(t2 - t1) == 1 and t2 > t1:
            row1 = row_t1
            row2 = row_t2
        else:
            print('Error generating row')

        x1, y1 = preprocess(row1['image_path'].values[0], row1['speed'].values[0], split=type)
        x2, y2 = preprocess(row2['image_path'].values[0], row2['speed'].values[0], split=type)

        image = np.concatenate((x1, x2), axis=2)

        image_batch[i] = image
        label_batch[i] = np.mean([y1,y2])

        yield shuffle(image_batch, label_batch)
