import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import nn
'''
generator with mt.pheonix architecture
'''

def lrelu(x):
    return tf.maximum(x*0.2,x)

def upsample_and_concat(x1, x2, output_channels, in_channels):

    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal( [pool_size, pool_size, output_channels, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2) , strides=[1, pool_size, pool_size, 1] )

    deconv_output =  tf.concat([deconv, x2],3)
    deconv_output.set_shape([None, None, None, output_channels*2])

    return deconv_output

def unet(input):
    with tf.variable_scope("generator"):
        input1 = slim.conv2d(input, 16, [3, 3], stride=2, rate=1, activation_fn=lrelu, scope='input_conv')
        conv1=slim.conv2d(input1,16,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_1')
        conv1=slim.conv2d(conv1,16,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_2')
        pool1=slim.conv2d(conv1,16,[3,3], stride=2, rate=1, activation_fn=lrelu, scope='pooling1' )

        conv2=slim.conv2d(pool1,32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_1')
        conv2=slim.conv2d(conv2,32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_2')
        pool2=slim.conv2d(conv2,32,[3,3], stride=2, rate=1, activation_fn=lrelu, scope='pooling2' )

        conv3=slim.conv2d(pool2,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv3_1')
        conv3=slim.conv2d(conv3,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv3_2')
        pool3=slim.conv2d(conv3,64,[3,3], stride=2, rate=1, activation_fn=lrelu, scope='pooling3' )


        conv4=slim.conv2d(pool3,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv4_1')
        conv4=slim.conv2d(conv4,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv4_2')
        pool4=slim.conv2d(conv4,128,[3,3], stride=2, rate=1, activation_fn=lrelu, scope='pooling4' )


        conv5=slim.conv2d(pool4,256,[3,3], rate=1, activation_fn=lrelu,scope='g_conv5_1')
        conv_global = tf.reduce_mean(conv5,axis=[1,2])
        conv_dense = tf.layers.dense(conv_global,units=128,activation=tf.nn.relu)
        feature = tf.expand_dims(conv_dense,axis=1)
        feature = tf.expand_dims(feature,axis=2)
        ones = tf.zeros(shape=tf.shape(conv4))
        global_feature = feature + ones


        up6 =  tf.concat([conv4, global_feature], axis=3)
        conv6=slim.conv2d(up6,  128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv6_1')
        conv6=slim.conv2d(conv6,128,[3,3], rate=1, activation_fn=lrelu,scope='g_conv6_2')

        up7 =  upsample_and_concat( conv6, conv3, 64, 128  )
        conv7=slim.conv2d(up7,  64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_1')
        conv7=slim.conv2d(conv7,64,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_2')

        up8 =  upsample_and_concat( conv7, conv2, 32, 64 )
        conv8=slim.conv2d(up8,  32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv8_1')
        conv8=slim.conv2d(conv8,32,[3,3], rate=1, activation_fn=lrelu,scope='g_conv8_2')

        up9 =  upsample_and_concat( conv8, conv1, 16, 32 )
        conv9=slim.conv2d(up9,  16,[3,3], rate=1, activation_fn=lrelu,scope='g_conv9_1')
        conv9=slim.conv2d(conv9,16,[3,3], rate=1, activation_fn=lrelu,scope='g_conv9_2')

        conv9 = input1 * conv9
        deconv_filter = tf.Variable(tf.truncated_normal([2, 2, 3, 16], stddev=0.02))
        conv10 = tf.nn.conv2d_transpose(conv9, deconv_filter, tf.shape(input), strides=[1, 2, 2, 1])
        out = slim.conv2d(conv10, 3, [3, 3],rate=1,activation_fn=nn.tanh,scope='out') * 0.58 + 0.52

    return out

def adversarial(image_):

    with tf.variable_scope("discriminator"):

        conv1 = _conv_layer(image_, 48, 11, 4, batch_nn = False)
        conv2 = _conv_layer(conv1, 128, 5, 2)
        conv3 = _conv_layer(conv2, 192, 3, 1)
        conv4 = _conv_layer(conv3, 192, 3, 1)
        conv5 = _conv_layer(conv4, 128, 3, 2)
        
        flat_size = 128 * 7 * 7
        conv5_flat = tf.reshape(conv5, [-1, flat_size])

        W_fc = tf.Variable(tf.truncated_normal([flat_size, 1024], stddev=0.01))
        bias_fc = tf.Variable(tf.constant(0.01, shape=[1024]))

        fc = leaky_relu(tf.matmul(conv5_flat, W_fc) + bias_fc)

        W_out = tf.Variable(tf.truncated_normal([1024, 2], stddev=0.01))
        bias_out = tf.Variable(tf.constant(0.01, shape=[2]))

        adv_out = tf.nn.softmax(tf.matmul(fc, W_out) + bias_out)
    
    return adv_out

def weight_variable(shape, name):

    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name=name)

def bias_variable(shape, name):

    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, name=name)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def leaky_relu(x, alpha = 0.2):
    return tf.maximum(alpha * x, x)

def _conv_layer(net, num_filters, filter_size, strides, batch_nn=True):
    
    weights_init = _conv_init_vars(net, num_filters, filter_size)
    strides_shape = [1, strides, strides, 1]
    bias = tf.Variable(tf.constant(0.03, shape=[num_filters]))

    net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME') + bias   
    net = leaky_relu(net)

    if batch_nn:
        net = _instance_norm(net)

    return net

def _instance_norm(net):

    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]

    mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))

    epsilon = 1e-3
    normalized = (net-mu)/(sigma_sq + epsilon)**(.5)

    return scale * normalized + shift

def _conv_init_vars(net, out_channels, filter_size, transpose=False):

    _, rows, cols, in_channels = [i.value for i in net.get_shape()]

    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]

    weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=0.01, seed=1), dtype=tf.float32)
    return weights_init
