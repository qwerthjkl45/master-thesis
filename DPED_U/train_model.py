# python train_model.py model={iphone,sony,blackberry} dped_dir=dped vgg_dir=vgg_pretrained/imagenet-vgg-verydeep-19.mat

import tensorflow as tf
from scipy import misc
import numpy as np
import sys

from load_dataset import load_test_data, load_batch
from ssim import MultiScaleSSIM
import models
import utils as lutils
import vgg


from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import numpy
from load_data import *
import loss

# defining size of the training image patches

PATCH_WIDTH = 100
PATCH_HEIGHT = 100
PATCH_SIZE = PATCH_WIDTH * PATCH_HEIGHT * 3

# processing command arguments

phone, batch_size, train_size, learning_rate, num_train_iters, \
w_content, w_color, w_texture, w_tv, w_ssim,\
dped_dir, vgg_dir, eval_step = lutils.process_command_args(sys.argv)

#np.random.seed(0)

# loading training and test data

print("Loading test data...")
test_data, test_answ = load_test_data(phone, dped_dir, PATCH_SIZE)
print("Test data was loaded\n")

print("Loading training data...")
train_data  = trainset(dped_dir + '/iphone/', dped_dir + '/canon/')
#train_data  = trainset(dped_dir + '/train/raw/', dped_dir + '/train/target/')
trainloader = DataLoader(train_data, batch_size=batch_size,shuffle=True)
print("Training data was loaded\n")

TEST_SIZE = test_data.shape[0]
print(TEST_SIZE)
print('-----')
num_test_batches = int(test_data.shape[0]/batch_size)

# defining system architecture

PATCH_WIDTH = 100
PATCH_HEIGHT = 100
w_content = 0.001
w_color = 0.7
w_texture = 17.5
w_tv = 100
w_ssim = 0.5

with tf.Graph().as_default(), tf.Session() as sess:
    
    # placeholders for training data

    phone_ = tf.placeholder(tf.float32, [None, PATCH_SIZE])
    phone_image = tf.reshape(phone_, [-1, PATCH_HEIGHT, PATCH_WIDTH, 3])

    dslr_ = tf.placeholder(tf.float32, [None, PATCH_SIZE])
    dslr_image = tf.reshape(dslr_, [-1, PATCH_HEIGHT, PATCH_WIDTH, 3])

    adv_ = tf.placeholder(tf.float32, [None, 1])

    # get processed enhanced image

    enhanced = models.unet(phone_image)

    # transform both dslr and enhanced images to grayscale

    enhanced_gray = tf.reshape(tf.image.rgb_to_grayscale(enhanced), [-1, PATCH_WIDTH * PATCH_HEIGHT])
    dslr_gray = tf.reshape(tf.image.rgb_to_grayscale(dslr_image),[-1, PATCH_WIDTH * PATCH_HEIGHT])

    # push randomly the enhanced or dslr image to an adversarial CNN-discriminator

    adversarial_ = tf.multiply(enhanced_gray, 1 - adv_) + tf.multiply(dslr_gray, adv_)
    adversarial_image = tf.reshape(adversarial_, [-1, PATCH_HEIGHT, PATCH_WIDTH, 1])

    discrim_predictions = models.adversarial(adversarial_image)

    # losses
    # 1) texture (adversarial) loss

    discrim_target = tf.concat([adv_, 1 - adv_], 1)

    loss_discrim = -tf.reduce_sum(discrim_target * tf.log(tf.clip_by_value(discrim_predictions, 1e-10, 1.0)))
    loss_texture = -loss_discrim

    correct_predictions = tf.equal(tf.argmax(discrim_predictions, 1), tf.argmax(discrim_target, 1))
    discim_accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    # 2) content loss
    '''
    CONTENT_LAYER = 'relu5_4'

    enhanced_vgg = vgg.net(vgg_dir, vgg.preprocess(enhanced * 255))
    dslr_vgg = vgg.net(vgg_dir, vgg.preprocess(dslr_image * 255))

    content_size = utils._tensor_size(dslr_vgg[CONTENT_LAYER]) * batch_size
    loss_content = 2 * tf.nn.l2_loss(enhanced_vgg[CONTENT_LAYER] - dslr_vgg[CONTENT_LAYER]) / content_size
    '''    
    loss_content = loss.content_loss(dslr_image, enhanced, batch_size)
    
    # 3) color loss
    enhanced_blur = lutils.blur(enhanced)
    dslr_blur = lutils.blur(dslr_image)

    #loss_color = tf.reduce_sum(tf.pow(dslr_blur - enhanced_blur, 2))/(2 * batch_size)
    loss_color = tf.reduce_sum(tf.abs(dslr_image - enhanced))/(2 * batch_size)

    #loss_color = loss.color_loss(dslr_image, enhanced, batch_size)

    # 4) total variation loss

    batch_shape = (batch_size, PATCH_WIDTH, PATCH_HEIGHT, 3)
    tv_y_size = lutils._tensor_size(enhanced[:,1:,:,:])
    tv_x_size = lutils._tensor_size(enhanced[:,:,1:,:])
    y_tv = tf.nn.l2_loss(enhanced[:,1:,:,:] - enhanced[:,:batch_shape[1]-1,:,:])
    x_tv = tf.nn.l2_loss(enhanced[:,:,1:,:] - enhanced[:,:,:batch_shape[2]-1,:])
    loss_tv = 2 * (x_tv/tv_x_size + y_tv/tv_y_size) / batch_size

    # 5) ssim
    loss_ssim = loss.Mssim_loss(dslr_image, enhanced)

    # final loss

    loss_generator = w_content * loss_content + w_texture * loss_texture + w_color * loss_color + w_tv * loss_tv + w_ssim * loss_ssim

    # psnr loss

    enhanced_flat = tf.reshape(enhanced, [-1, PATCH_SIZE])

    loss_mse = tf.reduce_sum(tf.pow(dslr_ - enhanced_flat, 2))/(PATCH_SIZE * batch_size)
    loss_psnr = 20 * lutils.log10(1.0 / tf.sqrt(loss_mse))

    # optimize parameters of image enhancement (generator) and discriminator networks

    generator_vars = [v for v in tf.global_variables() if v.name.startswith("generator")]
    discriminator_vars = [v for v in tf.global_variables() if v.name.startswith("discriminator")]

    train_step_gen = tf.train.AdamOptimizer(learning_rate).minimize(loss_generator, var_list=generator_vars)
    train_step_disc = tf.train.AdamOptimizer(learning_rate).minimize(loss_discrim, var_list=discriminator_vars)

    saver = tf.train.Saver(var_list=generator_vars, max_to_keep=100)

    print('Initializing variables')
    sess.run(tf.global_variables_initializer())

    print('Training network')

    train_loss_gen = 0.0
    train_acc_discrim = 0.0
    lpsnr = 0.0

    all_zeros = np.reshape(np.zeros((batch_size, 1)), [batch_size, 1])
    test_crops = test_data[np.random.randint(0, TEST_SIZE, 5), :]

    logs = open('models/' + phone + '.txt', "w+")
    logs.close()

    for epoch in range(num_train_iters):
        print('======' + str(epoch) + '=====')
        for i, img_pair in enumerate(trainloader):
            #print(img_pair)
            # train generator

            #idx_train = np.random.randint(0, train_size, batch_size)

            #phone_images = train_data[idx_train]
            #dslr_images = train_answ[idx_train]

            train_data = img_pair['lr'].numpy()
            if train_data.shape[0] < batch_size:
                print('finish one EPOCH')
                continue


            train_data = np.float16(np.reshape(train_data, [batch_size, PATCH_SIZE]))/255
            phone_images = train_data        
            train_answ = img_pair['hr'].numpy()
            train_answ = np.float16(np.reshape(train_answ, [batch_size, PATCH_SIZE]))/255
            dslr_images = train_answ

            if train_answ.shape[0] < batch_size:
                print('finish one EPOCH')
                continue

            [loss_temp, temp, lc] = sess.run([loss_generator, train_step_gen, loss_color],
                                            feed_dict={phone_: phone_images, dslr_: dslr_images, adv_: all_zeros})
            train_loss_gen += loss_temp / eval_step

            # train discriminator

            idx_train = np.random.randint(0, train_size, batch_size)

            # generate image swaps (dslr or enhanced) for discriminator
            swaps = np.reshape(np.random.randint(0, 2, batch_size), [batch_size, 1])

            #phone_images = train_data[idx_train]
            #dslr_images = train_answ[idx_train]
            [accuracy_temp, temp] = sess.run([discim_accuracy, train_step_disc],
                                            feed_dict={phone_: phone_images, dslr_: dslr_images, adv_: swaps})
            train_acc_discrim += accuracy_temp / eval_step
            if i % eval_step == 0:

                # test generator and discriminator CNNs

                test_losses_gen = np.zeros((1, 6))
                test_accuracy_disc = 0.0
                loss_ssim = 0.0

                for j in range(num_test_batches):

                    be = j * batch_size
                    en = (j+1) * batch_size

                    swaps = np.reshape(np.random.randint(0, 2, batch_size), [batch_size, 1])

                    phone_images = test_data[be:en]
                    dslr_images = test_answ[be:en]

                    [enhanced_crops, accuracy_disc, lg, lcon, lcol, ltex, ltv, lpsnr] = sess.run([enhanced, discim_accuracy, \
                                    loss_generator, loss_content, loss_color, loss_texture, loss_tv, loss_psnr], \
                                    feed_dict={phone_: phone_images, dslr_: dslr_images, adv_: swaps})

                    losses = np.asarray([lg, lcon, lcol, ltex, ltv, lpsnr])
                    test_losses_gen += np.asarray(losses) / num_test_batches
                    test_accuracy_disc += accuracy_disc / num_test_batches

                    loss_ssim += MultiScaleSSIM(np.reshape(dslr_images * 255, [batch_size, PATCH_HEIGHT, PATCH_WIDTH, 3]),
                                                        enhanced_crops * 255) / num_test_batches

                logs_disc = "step %d, %s | discriminator accuracy | train: %.4g, test: %.4g " % \
                      (i, phone, train_acc_discrim, test_accuracy_disc)

                logs_gen = "generator losses | train: %.4g, test: %.4g | content: %.4g, color: %.4g, texture: %.4g, tv: %.4g | psnr: %.4g, ssim: %.4g\n" % \
                      (train_loss_gen, test_losses_gen[0][0], test_losses_gen[0][1], test_losses_gen[0][2],
                       test_losses_gen[0][3], test_losses_gen[0][4], test_losses_gen[0][5], loss_ssim)

                print(logs_disc)
                print(logs_gen)

                # save the results to log file

                logs = open('models/' + phone + '.txt', "a")
                logs.write(logs_disc)
                logs.write('\n')
                logs.write(logs_gen)
                logs.write('\n')
                logs.close()

                # save visual results for several test image crops

                enhanced_crops = sess.run(enhanced, feed_dict={phone_: test_crops, dslr_: dslr_images, adv_: all_zeros})

                idx = 0
                for crop in enhanced_crops:
                    before_after = np.hstack((np.reshape(test_crops[idx], [PATCH_HEIGHT, PATCH_WIDTH, 3]), crop))
                    misc.imsave('results/' + str(phone)+ "_" + str(idx) + '_iteration_' + str(i) + '.jpg', before_after)
                    idx += 1

                train_loss_gen = 0.0
                train_acc_discrim = 0.0
                lpsnr = 0.0
                # save the model that corresponds to the current iteration

                saver.save(sess, 'models/' + str(phone) + '_iteration_' + str(i) + '.ckpt', write_meta_graph=False)

                # reload a different batch of training data

                #del train_data
                #del train_answ
            #train_data, train_answ = load_batch(phone, dped_dir, train_size, PATCH_SIZE)
