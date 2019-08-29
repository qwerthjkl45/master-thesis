#form content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
from __future__ import division
import os, time, scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import rawpy
import glob
import models2 as models

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from load_data import *
import skimage
import loss
import vgg

input_dir = '../DPED/dped/new_Sony_train/short/'
gt_dir = '../DPED/dped/new_Sony_train/long/'
checkpoint_dir = './sid_w_sa/'#'./tmp4_1/5_2
result_dir = './sid_w_sa/'
d_checkpoint_dir = './disc/'



ps = 300  # patch size for training

train_data  = trainset('../DPED/dped/new_Sony_train/short/', '../DPED/dped/new_Sony_train/long/')
trainloader = DataLoader(train_data, batch_size=5,shuffle=True)
# get train IDs
def lrelu(x):
    return tf.maximum(x * 0.2, x)


def upsample_and_concat(x1, x2, output_channels, in_channels):
    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal([pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2), strides=[1, pool_size, pool_size, 1])

    deconv_output = tf.concat([deconv, x2], 3)
    deconv_output.set_shape([None, None, None, output_channels * 2])

    return deconv_output





def pack_raw(raw):
    # pack Bayer image to 4 channels
    im = raw.raw_image_visible.astype(np.float32)
    im = np.maximum(im - 512, 0) / (16383 - 512)  # subtract the black level

    im = np.expand_dims(im, axis=2)
    img_shape = im.shape
    H = img_shape[0]
    W = img_shape[1]

    out = np.concatenate((im[0:H:2, 0:W:2, :],
                          im[0:H:2, 1:W:2, :],
                          im[1:H:2, 1:W:2, :],
                          im[1:H:2, 0:W:2, :]), axis=2)
    return out


def parse_prob():
    f = open('./prob.txt', 'r')
    file_prob = {}
    f1 = f.readlines()
    for x in f1:
        [idx, prob_list] = x.split(" ")
        p1, p2, p3, _ = prob_list.split(",")
        file_list = ['0.1', '0.04', '0.033']
        prob_sum = float(int(p1) + int(p2) + int(p3))
        file_prob[int(idx)] = [int(p1)/prob_sum, int(p2)/prob_sum, int(p3)/prob_sum]
    return file_prob

ADV_PATCH_WIDTH = 600
ADV_PATCH_HEIGHT = 600
PATCH_WIDTH = 300
PATCH_HEIGHT = 300


sess = tf.Session()
in_image = tf.placeholder(tf.float32, [None, 300, 300, 4])
gt_image = tf.placeholder(tf.float32, [None, 600, 600, 3])
adv_ = tf.placeholder(tf.float32, [None, 1])
out_image = models.unet(in_image)

gt_and_out = tf.concat([gt_image, out_image], 3) #4 x h x w x 6
crop_gt_and_out = tf.random_crop(gt_and_out, [5, 300, 300, 6])

# 1. discriminator loss:
enhanced_gray = tf.reshape(tf.image.rgb_to_grayscale(crop_gt_and_out[:, :, :, 3:6]), [-1, PATCH_WIDTH * PATCH_HEIGHT])
dslr_gray = tf.reshape(tf.image.rgb_to_grayscale(crop_gt_and_out[:, :, :, 0:3]), [-1, PATCH_WIDTH * PATCH_HEIGHT])

# push randomly the enhanced or dslr image to an adversarial CNN-discriminator

adversarial_ = tf.multiply(enhanced_gray, 1 - adv_) + tf.multiply(dslr_gray, adv_)
adversarial_image = tf.reshape(adversarial_, [-1, PATCH_HEIGHT, PATCH_WIDTH, 1])
discrim_predictions = models.adversarial1(adversarial_image)

# texture (adversarial) loss

discrim_target = tf.concat([adv_, 1 - adv_], 1)

loss_discrim = -tf.reduce_sum(discrim_target * tf.log(tf.clip_by_value(discrim_predictions, 1e-10, 1.0)))

loss_texture = -loss_discrim /1000

correct_predictions = tf.equal(tf.argmax(discrim_predictions, 1), tf.argmax(discrim_target, 1))
discim_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

# 2. color loss:
loss_color = 1000 * tf.reduce_mean(tf.abs(out_image - gt_image))


# 3. content loss:
loss_content = loss.content_loss(crop_gt_and_out[:, :, :, 0:3], crop_gt_and_out[:, :, :, 3:6], 5)/1200



G_loss = loss_color + loss_texture + loss_content #+ loss_tv

#t_vars = tf.trainable_variables()
generator_vars = [v for v in tf.global_variables() if v.name.startswith("generator")]
discriminator_vars = [v for v in tf.global_variables() if v.name.startswith("discriminator")]
lr = tf.placeholder(tf.float32)

G_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(G_loss, var_list=generator_vars)
D_opt = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss_discrim, var_list=discriminator_vars)

d_saver = tf.train.Saver(var_list = discriminator_vars, max_to_keep=1)
saver = tf.train.Saver(max_to_keep=1)

sess.run(tf.global_variables_initializer())

#d_ckpt = tf.train.get_checkpoint_state(d_checkpoint_dir)
#if d_ckpt:
#    print('loaded discriminator:' + d_ckpt.model_checkpoint_path)
#    d_saver.restore(sess, d_ckpt.model_checkpoint_path)

ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
lastepoch = 0
if ckpt:
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)
    lastepoch = int(ckpt.model_checkpoint_path.split("/")[-1].split('-')[0]) + 1
#
# Raw data takes long time to load. Keep them in memory after loaded.
#gt_images = [None] * 6000
#input_images = {}
#input_images['300'] = [None] * len(train_ids)
#input_images['250'] = [None] * len(train_ids)
#input_images['100'] = [None] * len(train_ids)
#g_loss = np.zeros((5000, 1))

#allfolders = glob.glob('./result/*0')

learning_rate = 1e-4
batch_size = 5
train_acc_discrim = 0.0 
all_zeros = np.reshape(np.zeros((batch_size, 1)), [batch_size, 1])

for epoch in range(lastepoch, 4001):
    cnt = 0
    if epoch > (2000):
        learning_rate = 1e-5
    for i, img_pair in enumerate(trainloader):
        train_data = img_pair['lr'].numpy()      
        train_data = np.float32(train_data)
        train_answ = img_pair['hr'].numpy()
        train_answ = np.float32(train_answ)

        if train_data.shape[0] < batch_size:
            print('==finish one epoch===')
            continue
        st = time.time()
        cnt += 5

        # crop
        H = train_data.shape[1]
        W = train_data.shape[2]

        xx = np.random.randint(0, W - ps)
        yy = np.random.randint(0, H - ps)
        input_patch = train_data[:, yy:yy + ps, xx:xx + ps, :]
        gt_patch =train_answ[:, yy * 2:yy * 2 + ps * 2, xx * 2:xx * 2 + ps * 2, :]  

        if np.random.randint(2, size=1)[0] == 1:  # random flip
            input_patch = np.flip(input_patch, axis=1)
            gt_patch = np.flip(gt_patch, axis=1)
        if np.random.randint(2, size=1)[0] == 1:
            input_patch = np.flip(input_patch, axis=2)
            gt_patch = np.flip(gt_patch, axis=2)
        if np.random.randint(2, size=1)[0] == 1:  # random transpose
            input_patch = np.transpose(input_patch, (0, 2, 1, 3))
            gt_patch = np.transpose(gt_patch, (0, 2, 1, 3))
            
        input_patch = np.minimum(input_patch, 1.0)

        _, G_current, output, l_tex, l_c, l_con = sess.run([G_opt, G_loss, out_image, loss_texture, loss_color, loss_content],
                                       feed_dict={in_image: input_patch, gt_image: gt_patch, lr: learning_rate,  adv_: all_zeros})
        
        output = np.minimum(np.maximum(output, 0), 1)


        # train discriminator
        # generate image swaps (dslr or enhanced) for discriminator
        swaps = np.reshape(np.random.randint(0, 2, batch_size), [batch_size, 1])
        [_, D_accuracy, l_tex] = sess.run([D_opt, discim_accuracy, loss_texture],
                                            feed_dict={in_image: input_patch, gt_image: gt_patch, lr: learning_rate,  adv_: swaps})
        train_acc_discrim += D_accuracy / 100
        psnr = skimage.measure.compare_psnr(gt_patch, output)

      
        '''
        if epoch % 5 == 0 and cnt % 10 == 0:
            if not os.path.isdir(result_dir + '%04d' % epoch):
                os.makedirs(result_dir + '%04d' % epoch)

            temp = np.concatenate((gt_patch[0, :, :, :], output[0, :, :, :]), axis=1)
            scipy.misc.toimage(temp * 255, high=255, low=0, cmin=0, cmax=255).save(
                result_dir + '%04d/%05d_00_train.jpg' % (epoch, cnt))


        '''
        if cnt % 5 == 0:
            print("gan + ratio + model2/5_2 w : %d %d Loss=%.3f: texture, color, content = (%.3f,%.3f, %.3f), PSNR=%.3f | D_accuracy: %.4g" % (epoch, cnt, G_current, l_tex, l_c, l_con,\
                                                                                                                                      psnr, D_accuracy))
            train_acc_discrim = 0.0 
          
    saver.save(sess, checkpoint_dir + str(epoch)  + '-' +'model.ckpt')
    if epoch == 1015:
        break    
    
        
