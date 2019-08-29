from __future__ import division
import os, scipy.io
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import rawpy
import glob
from models2 import unet
import skimage

input_dir = './sony_test_data/short/'
gt_dir = './sony_test_data/long/'
checkpoint_dir =  './sid_w_sa/'#bak_batchsize4/'
result_dir = './sid_w_sa/'


# get test IDs
test_fns = glob.glob(gt_dir + '/1*.ARW')
test_ids = [int(os.path.basename(test_fn)[0:5]) for test_fn in test_fns]

DEBUG = 0
if DEBUG == 1:
    save_freq = 2
    test_ids = test_ids[0:5]


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




def rgb2gray(img):
    img = img * 255
    R = img[:, :, :, 0]
    G = img[:, :, :, 1]
    B = img[:, :, :, 2]
    Y = (0.2989 * R) + (0.5870 * G) + (0.1140 * B)
    Y = np.reshape(Y, (Y.shape[1], Y.shape[2]))
    return Y


sess = tf.Session()
in_image = tf.placeholder(tf.float32, [None, None, None, 4])
gt_image = tf.placeholder(tf.float32, [None, None, None, 3])

out_image = unet(in_image)

saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if True:
    #tmp = checkpoint_dir + "1015-model.ckpt"
    #print('loaded ' + tmp)
    #saver.restore(sess, tmp)
    
    print('loaded ' + ckpt.model_checkpoint_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

if not os.path.isdir(result_dir + 'final/'):
    os.makedirs(result_dir + 'final/')
if not os.path.isdir(result_dir + 'gt/'):
    os.makedirs(result_dir + 'gt/')

psnr_list = []

for test_id in test_ids:
    # test the first image in each sequence
    for f_idx in range(10):
        in_files = glob.glob(input_dir + '%05d_%02d_*.ARW' % (test_id, f_idx))
        for k in range(len(in_files)):
            in_path = in_files[k]
            in_fn = os.path.basename(in_path)
            print(in_fn)
            gt_files = glob.glob(gt_dir + '%05d_00*.ARW' % test_id)
            gt_path = gt_files[0]
            gt_fn = os.path.basename(gt_path)
            in_exposure = float(in_fn[9:-5])
            gt_exposure = float(gt_fn[9:-5])
            ratio = min(gt_exposure / in_exposure, 300)

            raw = rawpy.imread(in_path)
            input_full = np.expand_dims(pack_raw(raw), axis=0) * ratio

            im = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt_raw = rawpy.imread(gt_path)
            im = gt_raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
            gt_full = np.expand_dims(np.float32(im / 65535.0), axis=0)


            input_full = np.minimum(input_full, 1.0)

            output = sess.run(out_image, feed_dict={in_image: input_full})
            output = np.minimum(np.maximum(output, 0), 1)

            output = output[0, :, :, :]
            gt_full = gt_full[0, :, :, :]
  
            
            psnr_list += [skimage.measure.compare_psnr(gt_full, output)]

            print('Avg PSNR: %.3f'%(sum(psnr_list)/len(psnr_list)))
            print('\n')
        
            scipy.misc.toimage(output * 255, high=255, low=0, cmin=0, cmax=255).save(
            result_dir + 'final/%s.png' % (in_fn[:-4]))









       
      

