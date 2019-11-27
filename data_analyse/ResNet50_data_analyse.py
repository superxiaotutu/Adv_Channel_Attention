import sys
sys.path.append('../')

import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
from data_analyse.val_data import *
import tqdm
import pickle
import numpy as np
from data_analyse.preprocess.inception import *
import os
# set GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=sess_config)

image_R = tf.placeholder(tf.float32, [None, 299, 299, 3])
image_A = tf.placeholder(tf.float32, [None, 299, 299, 3])
label = tf.placeholder(tf.float32, [None, 1000])

batchsize = 50
attack_step = int(sys.argv[1])
IMAGENET_VAL = ImageNet_datastream(sess, batchsize, 299)

def resnet_v2_50(image, reuse=tf.AUTO_REUSE):
    # preprocess = lambda x: preprocess_image(x, 224, 224, is_training=False)
    # preprocessed = tf.map_fn(preprocess, elems=image)
    preprocessed = ((image/255)-0.5)*2
    arg_scope = nets.resnet_v2.resnet_arg_scope(weight_decay=0.0)
    with slim.arg_scope(arg_scope):
        logits, end_point = nets.resnet_v2.resnet_v2_50(preprocessed, 1001, is_training=False, reuse=reuse)
        logits = tf.squeeze(logits)
        logits = logits[:, 1:]  # ignore background class
        probs = tf.nn.softmax(logits)  # probabilities
    return logits, probs, end_point


def FGSM(x, logits, eps=8):
    total_class_num = tf.shape(logits)[1]
    ori_class = tf.argmax(logits, 1)
    one_hot_class = tf.one_hot(ori_class, total_class_num)
    cross_entropy = tf.losses.softmax_cross_entropy(one_hot_class,
                                                    logits,
                                                    label_smoothing=0.1,
                                                    weights=1.0)
    x_adv = x + eps * tf.sign(tf.gradients(cross_entropy, x)[0])
    x_adv = tf.clip_by_value(x_adv, 0, 255)
    return tf.stop_gradient(x_adv)


logits_R, probs_R, end_point_R = resnet_v2_50(image_R)
logits_A, probs_A, end_point_A = resnet_v2_50(image_A)

correct_R = tf.equal(tf.argmax(logits_R, 1), (tf.argmax(label, 1)))
accuracy_R = tf.reduce_mean(tf.cast(correct_R, "float"))

correct_A = tf.equal(tf.argmax(logits_A, 1), (tf.argmax(label, 1)))
accuracy_A = tf.reduce_mean(tf.cast(correct_A, "float"))

# kernel op
# minus_op = [tf.cast(tf.subtract(end_point_R[i], end_point_A[i]), tf.float16) for i in end_point_R]
minus_op = [tf.reduce_mean(tf.abs(tf.subtract(end_point_R[i], end_point_A[i]))) for i in end_point_R]

FGSM_Uint8 = FGSM(image_R, logits_R, 2)

# load target var
restore_vars = [
    var for var in tf.global_variables()
    if var.name.startswith('resnet_v2_50/')
]
pre_train_saver = tf.train.Saver(restore_vars)
pre_train_saver.restore(sess, "/home/kirin/Python_Code/Adv_Channel_Attention/data_analyse/models/resnet_v2_50/resnet_v2_50.ckpt")


ans = []
T_RACC, TAACC = 0, 0
pbar = tqdm.trange(50000 // batchsize)
for i in pbar:
    batch_R, label_R = IMAGENET_VAL.get_test_batch()
    batch_A = batch_R
    for t in range(30):
        batch_A = sess.run(FGSM_Uint8, feed_dict={image_R: batch_A, label: label_R})
        batch_A = np.clip(batch_A, batch_R - attack_step, batch_R + attack_step)
    batch_A = np.clip(batch_A, 0, 255)
    R_ACC, A_ACC = sess.run([accuracy_R, accuracy_A], feed_dict={image_R: batch_R, image_A: batch_A, label: label_R})
    tmp = sess.run(minus_op, feed_dict={image_R: batch_R, image_A: batch_A})
    if i == 0:
        ans = tmp
    else:
        for l in range(len(ans)):
            ans[l] = np.vstack((ans[l], tmp[l]))
    T_RACC += R_ACC
    TAACC += A_ACC
    pbar.set_description("step:{}, R_ACC:{:.4f}, A_ACC:{:.4f}".format(attack_step, T_RACC / (i + 1), TAACC / (i + 1)))

f = open("resnet50.txt", 'a')
print("attack_step:{}".format(attack_step))
f.writelines("attack_step:{}\n".format(attack_step))
for i in range(len(ans)):
    shape = np.shape(ans[i])
    size = 1
    for channel in shape:
        size = size * channel
    avg = np.sum(np.abs(ans[i]) / size)
    print("{} {}".format(list(end_point_R)[i], avg))
    f.writelines("{} {}\n".format(list(end_point_R)[i], avg))

index = 50000 // batchsize
print("R_ACC:{:.4f}, A_ACC:{:.4f}".format(T_RACC / index, TAACC / index))
f.writelines("R_ACC {:.4f}\nA_ACC {:.4f}\n\n".format(T_RACC / index, TAACC / index))
f.close()
