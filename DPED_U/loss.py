import tensorflow as tf
import utils
import vgg as vgg
from ssim import _MultiScaleSSIM

PATCH_WIDTH = 100
PATCH_HEIGHT = 100
w_content = 0.001
w_color = 0.5
w_texture = 17.5
w_tv = 100
w_ssim = 0.5


# content loss
def content_loss(target, prediction,batch_size):
  CONTENT_LAYER = 'relu5_4'
  CONTENT_LAYER1 = 'relu3_4'
  CONTENT_LAYER2 = 'relu1_2'
  vgg_dir = './vgg_pretrained/imagenet-vgg-verydeep-19.mat'
  enhanced_vgg = vgg.net(vgg_dir, vgg.preprocess(prediction * 255))
  dslr_vgg = vgg.net(vgg_dir, vgg.preprocess(target * 255))

  content_size = utils._tensor_size(dslr_vgg[CONTENT_LAYER]) * batch_size
  content_size1 = utils._tensor_size(dslr_vgg[CONTENT_LAYER1]) * batch_size
  content_size2 = utils._tensor_size(dslr_vgg[CONTENT_LAYER2]) * batch_size

  loss_content = 2 * tf.nn.l2_loss(enhanced_vgg[CONTENT_LAYER] - dslr_vgg[CONTENT_LAYER]) / content_size
  loss_content1 = 2 * tf.nn.l2_loss(enhanced_vgg[CONTENT_LAYER1] - dslr_vgg[CONTENT_LAYER1]) / content_size1
  loss_content2 = 2 * tf.nn.l2_loss(enhanced_vgg[CONTENT_LAYER2] - dslr_vgg[CONTENT_LAYER2]) / content_size2

  return (tf.reduce_mean(loss_content)+tf.reduce_mean(loss_content1)+tf.reduce_mean(loss_content2))/3
  
def color_loss(target,prediction,batch_size):
    loss_color = tf.reduce_sum(tf.abs(target - prediction)) / (2 * batch_size)
    return loss_color*0.1
    
def Mssim_loss(target,prediction):
    loss_Mssim = 1-_MultiScaleSSIM(target,prediction)
    return loss_Mssim*1000

