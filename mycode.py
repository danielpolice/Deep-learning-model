import tensorflow
import os
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

if tensorflow.__version__.startswith('1.'):
    import tensorflow as tf
    from tensorflow.python.platform import gfile
else:
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    import tensorflow.compat.v1.gfile as gfile

print("Tensorflow Version:", tf.__version__)

tf_input = tf.placeholder(shape=[None, 32, 32, 3], dtype=tf.float32, name='input')
tf_keep_prob = tf.placeholder(dtype=tf.float32, name="keep_prob")
tf_keep_prob_2 = tf.placeholder(dtype=tf.float32, name="keep_prob_2")
tf_label_batch = tf.placeholder(shape=[None], dtype=tf.int32, name="label_batch")

model_shape = tf_input.shape
print("model shape:", model_shape)
print(type(model_shape))

kernel_size = [3, 3]
# first conv
net = tf.layers.conv2d(
    inputs=tf_input,
    filters=8,
    kernel_size=kernel_size,
    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
    padding="same",
    activation=tf.nn.relu
)
# net = tf.layers.conv2d(
#     inputs=net,
#     filters=64,
#     kernel_size=kernel_size,
#     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
#     padding="same",
#     activation=tf.nn.relu
# )

# first maxpooling
net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=2, padding='SAME')
print("pool_1 shape:", net.shape)
# ---------------------------------------------------------------------------------

# second conv
net = tf.layers.conv2d(
    inputs=net,
    filters=16,
    kernel_size=kernel_size,
    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
    padding="same",
    activation=tf.nn.relu
)
# net = tf.layers.conv2d(
#     inputs=net,
#     filters=128,
#     kernel_size=kernel_size,
#     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
#     padding="same",
#     activation=tf.nn.relu
# )

# second maxpooling
net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=2, padding='SAME')
print("pool_2 shape:", net.shape)
# -------------------------------------------------------------------------------

# third conv
net = tf.layers.conv2d(
    inputs=net,
    filters=32,
    kernel_size=kernel_size,
    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
    padding="same",
    activation=tf.nn.relu
)
# net = tf.layers.conv2d(
#     inputs=net,
#     filters=256,
#     kernel_size=kernel_size,
#     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
#     padding="same",
#     activation=tf.nn.relu
# )
# net = tf.layers.conv2d(
#     inputs=net,
#     filters=256,
#     kernel_size=kernel_size,
#     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
#     padding="same",
#     activation=tf.nn.relu
# )
# third maxpooling
net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=2, padding='SAME')
print("pool_3 shape:", net.shape)
# ----------------------------------------------------------------------------

# fourth conv
net = tf.layers.conv2d(
    inputs=net,
    filters=64,
    kernel_size=kernel_size,
    kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
    padding="same",
    activation=tf.nn.relu
)
# net = tf.layers.conv2d(
#     inputs=net,
#     filters=512,
#     kernel_size=kernel_size,
#     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
#     padding="same",
#     activation=tf.nn.relu
# )
# net = tf.layers.conv2d(
#     inputs=net,
#     filters=512,
#     kernel_size=kernel_size,
#     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
#     padding="same",
#     activation=tf.nn.relu
# )
# fourth maxpooling
net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=2, padding='SAME')
print("pool_4 shape:", net.shape)
# ------------------------------------------------------------------------------

# fifth conv
# net = tf.layers.conv2d(
#     inputs=net,
#     filters=512,
#     kernel_size=kernel_size,
#     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
#     padding="same",
#     activation=tf.nn.relu
# )
# net = tf.layers.conv2d(
#     inputs=net,
#     filters=512,
#     kernel_size=kernel_size,
#     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
#     padding="same",
#     activation=tf.nn.relu
# )
# net = tf.layers.conv2d(
#     inputs=net,
#     filters=512,
#     kernel_size=kernel_size,
#     kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.1),
#     padding="same",
#     activation=tf.nn.relu
# )
# fifth maxpooling
# net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=2, padding='SAME')
# print("pool_5 shape:", net.shape)

# -----------------------------------------------------------------------------

# flatten
net = tf.layers.flatten(net)

# fully connection
# FC-4096
net = tf.nn.dropout(net, keep_prob=tf_keep_prob)
net = tf.layers.dense(inputs=net, units=128, activation=tf.nn.relu)
print("FC_1 shape:", net.shape)

# FC-4096
# net = tf.nn.dropout(net, keep_prob=tf_keep_prob_2)
# net = tf.layers.dense(inputs=net, units=4096, activation=tf.nn.relu)
# print("FC_2 shape:", net.shape)

# FC-1000
output = tf.layers.dense(inputs=net, units=10, activation=None)
print("output shape:", output.shape)

prediction = tf.nn.softmax(output, name="prediction")
# --------------------------------------------------------------------------------

# loss and optimizer
learning_rate = 1e-4

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=tf_label_batch,
    logits=output),
    name="loss")
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
# ----------------------------------------------------------------------------------

# read image paths and record classes from the train folder
root_dir = r"C:\xiaodan\project\mycode\fashion_mnist\train"

# var
img_format = {'png', 'jpg', 'bmp'}
paths_train = list()
class2label = dict()
count = 0
labels_train = list()

for dirname, sub_dirname, filenames in os.walk(root_dir):
    if len(filenames) > 0:
        for filename in filenames:
            if filename.split(".")[-1] in img_format:
                full_path = os.path.join(dirname, filename)
                paths_train.append(full_path)

                # collect classes
                classname = dirname.split("\\")[-1]
                if classname not in class2label.keys():
                    class2label[classname] = count
                    count += 1

                # read labels
                labels_train.append(class2label[classname])

paths_train = np.array(paths_train)
labels_train = np.array(labels_train)

print("Path_train shape:", paths_train.shape)
print("Label_train shape:", labels_train.shape)
# -----------------------------------------------------------------
# show the classname to label
print(class2label)

#
rdm_num = np.random.randint(0, paths_train.shape[0])
print("rdm_num:", rdm_num)
img = cv2.imread(paths_train[rdm_num])
plt.imshow(img)
plt.axis('on')
plt.title(labels_train[rdm_num])
plt.show()

# read image paths and record classes from the validation folder
root_dir = r"C:\xiaodan\project\mycode\fashion_mnist\val"

# var
paths_val = list()
count = 0
labels_val = list()

for dirname, sub_dirname, filenames in os.walk(root_dir):
    if len(filenames) > 0:
        for filename in filenames:
            if filename.split(".")[-1] in img_format:
                full_path = os.path.join(dirname, filename)
                paths_val.append(full_path)
                classname = dirname.split("\\")[-1]
                # read labels
                labels_val.append(class2label[classname])

paths_val = np.array(paths_val)
labels_val = np.array(labels_val)

print("Path_val shape:", paths_val.shape)
print("Label_val shape:", labels_val.shape)
# -----------------------------------------------------------------
# show the classname to label
print(class2label)

#
rdm_num = np.random.randint(0, len(paths_val))
print("rdm_num:", rdm_num)
img = cv2.imread(paths_val[rdm_num])
plt.imshow(img)
plt.axis('on')
plt.title(labels_val[rdm_num])
plt.show()


# --------------------------------------------------------
# accuracy calculation funtion
def evaluation(predictions, labels):
    count = 0
    for i in range(predictions.shape[0]):
        if np.argmax(predictions[i]) == labels[i]:
            count += 1
    return count

# ---------------------------------------------------------
# create the folder to save model weights(CKPT,PB)
save_dir = r"C:\xiaodan\project\mycode\modelweight\tf_train"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

out_dir_prefix = os.path.join(save_dir, "model2")
saver = tf.train.Saver(max_to_keep=2)  # the number of CKPT files in save_dir
# appoint PB node names that will be saved after training
pb_save_path = os.path.join(save_dir, "pb_model2.pb")
pb_save_list = ['prediction']
# ------------------------------------------------------------------------
# model training
# var
batch_size = 32
img_quantity_train = paths_train.shape[0]
img_quantity_val = paths_val.shape[0]
GPU_ratio = None
epochs = 50
img_size = (tf_input.shape[1].value, tf_input.shape[2].value)
print(img_size)
print(type(img_size))

# calculate iterations
ites = math.ceil(img_quantity_train / batch_size)
print("img_quantity_train:", img_quantity_train)
print("ites:", ites)

ites_val = math.ceil(img_quantity_val / batch_size)
print("img_quantity_train:", img_quantity_val)
print("ites_val:", ites_val)

# GPU setting
config = tf.ConfigProto(log_device_placement=True,
                        allow_soft_placement=True)
if GPU_ratio is None:
    config.gpu_options.allow_growth = True
else:
    config.gpu_options.per_process_gpu_memory_fraction = GPU_ratio

with tf.Session(config=config) as sess:
    latest_weights_path = tf.train.latest_checkpoint(save_dir)
    print(latest_weights_path)

    if latest_weights_path is not None:
        saver.restore(sess, latest_weights_path)
        print("use previous model param:{}".format(latest_weights_path))
    else:
        sess.run(tf.global_variables_initializer())
        print("no previous model param can be used!")

    for epoch in range(epochs):
        train_loss = 0
        train_loss_2 = 0
        train_accuracy = 0

        val_loss = 0
        val_loss_2 = 0
        val_accuracy = 0

        # shuffle
        indice = np.random.permutation(img_quantity_train)
        paths_train = paths_train[indice]
        labels_train = labels_train[indice]

        for idx in range(ites):
            num_start = idx * batch_size
            num_end = num_start + batch_size
            if num_end > img_quantity_train:
                num_end = img_quantity_train

            batch_dim = [num_end - num_start]
            batch_dim.extend(model_shape[1:])
            batch_data = np.zeros(batch_dim, dtype=np.float32)

            for i, path in enumerate(paths_train[num_start:num_end]):
                img = cv2.imread(path)
                if img is None:
                    print("read failed:", path)
                else:
                    img = cv2.resize(img, (img_size[1], img_size[0]))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    batch_data[i] = img

            batch_data /= 255
            batch_labels = labels_train[num_start:num_end]

            sess.run(optimizer, feed_dict={tf_input: batch_data,
                                           tf_label_batch: batch_labels,
                                           tf_keep_prob: 0.5,
                                           tf_keep_prob_2: 0.75})
            print("finish one iteration")
        print("finish one epoch")

        # calculate the loss and accuracy of training set
        for idx in range(ites):
            num_start = idx * batch_size
            num_end = num_start + batch_size
            if num_end > img_quantity_train:
                num_end = img_quantity_train

            batch_dim = [num_end - num_start]
            batch_dim.extend(model_shape[1:])
            batch_data = np.zeros(batch_dim, dtype=np.float32)

            for i, path in enumerate(paths_train[num_start:num_end]):
                img = cv2.imread(path)
                if img is None:
                    print("read failed:", path)
                else:
                    img = cv2.resize(img, (img_size[1], img_size[0]))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    batch_data[i] = img

            batch_data /= 255
            batch_labels = labels_train[num_start:num_end]

            # loss and accuracy
            loss_temp, prediction_temp = sess.run([loss, prediction], feed_dict={tf_input: batch_data,
                                                                                 tf_label_batch: batch_labels,
                                                                                 tf_keep_prob: 1.0,
                                                                                 tf_keep_prob_2: 1.0})
            train_loss += loss_temp
            train_loss_2 += loss_temp * (num_end - num_start)
            train_accuracy += evaluation(prediction_temp, batch_labels)

        train_loss /= ites
        train_loss_2 /= img_quantity_train

        train_accuracy /= img_quantity_train
        print("train_loss:", train_loss)
        print("train_loss_2:", train_loss_2)
        print("train_accuracy", train_accuracy)

        # calculate the loss and accuracy of validation set
        for idx in range(ites_val):
            num_start = idx * batch_size
            num_end = np.minimum(num_start + batch_size, img_quantity_val)

            batch_dim = [num_end - num_start]
            batch_dim.extend(model_shape[1:])
            batch_data = np.zeros(batch_dim, dtype=np.float32)

            for i, path in enumerate(paths_val[num_start:num_end]):
                img = cv2.imread(path)
                if img is None:
                    print("read failed:", path)
                else:
                    img = cv2.resize(img, (img_size[1], img_size[0]))
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    batch_data[i] = img

            batch_data /= 255
            batch_labels = labels_val[num_start:num_end]

            # loss and accuracy
            loss_temp, prediction_temp = sess.run([loss, prediction], feed_dict={tf_input: batch_data,
                                                                                 tf_label_batch: batch_labels,
                                                                                 tf_keep_prob: 1.0,
                                                                                 tf_keep_prob_2: 1.0})
            val_loss += loss_temp
            val_loss_2 += loss_temp * (num_end - num_start)
            val_accuracy += evaluation(prediction_temp, batch_labels)

        val_loss /= ites_val
        val_loss_2 /= img_quantity_val

        val_accuracy /= img_quantity_val
        print("val_loss:", val_loss)
        print("val_loss_2:", val_loss_2)
        print("val_accuracy", val_accuracy)

        # save CKPT weights
        model_save_path = saver.save(sess, out_dir_prefix, global_step=epoch)
        print("save model CKPT to:", model_save_path)

        # save PB files
        graph = tf.get_default_graph().as_graph_def()
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, graph, pb_save_list)
        with tf.gfile.GFile(pb_save_path, 'wb') as f:
            f.write(output_graph_def.SerialzeToString())
        print("save PB file to", pb_save_path)