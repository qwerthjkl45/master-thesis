import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import nn
'''
generator with mt.pheonix architecture
add self attention
'''

def lrelu(x):
    return tf.maximum(x*0.2,x)

def upsample_and_concat(x1, x2, output_channels, in_channels):

    pool_size = 2
    deconv_filter = tf.Variable(tf.truncated_normal( [pool_size, pool_size, output_channels, in_channels], stddev=0.01))
    deconv = tf.nn.conv2d_transpose(x1, deconv_filter, tf.shape(x2) , strides=[1, pool_size, pool_size, 1] )

    deconv_output =  tf.concat([deconv, x2],3)
    deconv_output.set_shape([None, None, None, output_channels*2])

    return deconv_output
    
def hw_flatten(x):
    x_shape = x.get_shape().as_list()
    return tf.reshape(x, shape=[-1, tf.shape(x)[1]*tf.shape(x)[2], tf.shape(x)[-1]])


def attention(x, out_dim, scope='generator', reuse=False):
    
    with tf.variable_scope(scope, reuse=reuse):
        f = _conv_layer(x, out_dim // 8, 1, 1)
        g = _conv_layer(x, out_dim // 8, 1, 1)
        h = _conv_layer(x, out_dim, 1, 1)
        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # [bs, N, N]
        beta = tf.nn.softmax(s)  # attention map
        o = tf.matmul(beta, hw_flatten(h))  # [bs, N, C]
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
        x_shape = x.get_shape().as_list()
        o = tf.reshape(o, shape=[-1, tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]])  # [bs, h, w, C]
        x = gamma * o + x
    return x


def unet(input):
    with tf.variable_scope("generator"):
        #input1 = slim.conv2d(input, 16, [3, 3], rate=1, activation_fn=lrelu, scope='input_conv')
        conv1=slim.conv2d(input,16*2,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_1')
        conv1=slim.conv2d(conv1,16*2,[3,3], rate=1, activation_fn=lrelu,scope='g_conv1_2')
        #pool1=slim.conv2d(conv1,16*2,[3,3], stride=2, rate=1, activation_fn=lrelu, scope='pooling1' )
        pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')
        

        conv2=slim.conv2d(pool1,32*2,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_1')
        conv2=slim.conv2d(conv2,32*2,[3,3], rate=1, activation_fn=lrelu,scope='g_conv2_2')
        #conv2 = attention(conv2, 64, 'a1')
        #pool2=slim.conv2d(conv2,32*2,[3,3], stride=2, rate=1, activation_fn=lrelu, scope='pooling2' )
        pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

        conv3=slim.conv2d(pool2,64*2,[3,3], rate=1, activation_fn=lrelu,scope='g_conv3_1')
        conv3=slim.conv2d(conv3,64*2,[3,3], rate=1, activation_fn=lrelu,scope='g_conv3_2')
        pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')        
        #conv3 = attention(conv3, 64*2, 'a1')
        #pool3=slim.conv2d(conv3,64*2,[3,3], stride=2, rate=1, activation_fn=lrelu, scope='pooling3' )


        conv4=slim.conv2d(pool3,128*2,[3,3], rate=1, activation_fn=lrelu,scope='g_conv4_1')
        conv4=slim.conv2d(conv4,128*2,[3,3], rate=1, activation_fn=lrelu,scope='g_conv4_2')
        pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')        
        #conv4 = attention(conv4, 128*2, 'a2')
        #pool4=slim.conv2d(conv4,128*2,[3,3], stride=2, rate=1, activation_fn=lrelu, scope='pooling4' )
        print('pool4: ', pool4.shape)
        
        conv5=slim.conv2d(pool4,256*2,[3,3], rate=1, activation_fn=lrelu,scope='g_conv5_1')
        conv_global = tf.reduce_mean(conv5,axis=[1,2])
        conv_dense = tf.layers.dense(conv_global,units=128 * 2,activation=tf.nn.relu)
        feature = tf.expand_dims(conv_dense,axis=1)
        feature = tf.expand_dims(feature,axis=2)
        ones = tf.zeros(shape=tf.shape(conv4))
        global_feature = feature + ones


        up6 =  tf.concat([conv4, global_feature], axis=3)
        conv6=slim.conv2d(up6,  128*2,[3,3], rate=1, activation_fn=lrelu,scope='g_conv6_1')
        conv6=slim.conv2d(conv6,128*2,[3,3], rate=1, activation_fn=lrelu,scope='g_conv6_2')
        print('conv6: ', conv6.shape)
        #conv6 = attention(conv6, 128*2, 'a3')

        up7 =  upsample_and_concat( conv6, conv3, 64*2, 128 *2 )
        conv7=slim.conv2d(up7,  64*2,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_1')
        conv7=slim.conv2d(conv7,64*2,[3,3], rate=1, activation_fn=lrelu,scope='g_conv7_2')
        #conv7 = attention(conv7, 64*2, 'a4')
       


        up8 =  upsample_and_concat( conv7, conv2, 32*2, 64*2 )
        conv8=slim.conv2d(up8,  32*2,[3,3], rate=1, activation_fn=lrelu,scope='g_conv8_1')
        conv8=slim.conv2d(conv8,32*2,[3,3], rate=1, activation_fn=lrelu,scope='g_conv8_2')
        #conv8 = attention(conv8, 64, 'a2')

        up9 =  upsample_and_concat( conv8, conv1, 16*2, 32*2 )
        conv9=slim.conv2d(up9,  16*2,[3,3], rate=1, activation_fn=lrelu,scope='g_conv9_1')
        conv9=slim.conv2d(conv9,16*2,[3,3], rate=1, activation_fn=lrelu,scope='g_conv9_2')
        print('con9: ', conv9.shape) 

   
            
        #deconv_filter = tf.Variable(tf.truncated_normal([2, 2, 12, 32], stddev=0.02))
        #conv10 = tf.nn.conv2d_transpose(conv9, deconv_filter, tf.convert_to_tensor([tf.shape(input)[0], 300, 300, 12]), strides=[1, 2, 2, 1])
        #print('conv10: ', conv10.shape)
        
        out = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn = None,scope='out') 
        out = tf.depth_to_space(out, 2)
        print('outt: ', out.shape)
        '''
        conv10 = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
        out = tf.depth_to_space(conv10, 2)
        ''' 
        
    return out
    
def network(input):
    with tf.variable_scope("generator"):
        conv1 = slim.conv2d(input, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_1')
        conv1 = slim.conv2d(conv1, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv1_2')
        pool1 = slim.max_pool2d(conv1, [2, 2], padding='SAME')

        conv2 = slim.conv2d(pool1, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_1')
        conv2 = slim.conv2d(conv2, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv2_2')
        conv2 = attention(conv2, 64, 'a0')
        pool2 = slim.max_pool2d(conv2, [2, 2], padding='SAME')

        conv3 = slim.conv2d(pool2, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_1')
        conv3 = slim.conv2d(conv3, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv3_2')
        conv3 = attention(conv3, 128, 'a1')
        pool3 = slim.max_pool2d(conv3, [2, 2], padding='SAME')

        conv4 = slim.conv2d(pool3, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_1')
        conv4 = slim.conv2d(conv4, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv4_2')
        pool4 = slim.max_pool2d(conv4, [2, 2], padding='SAME')

        conv5 = slim.conv2d(pool4, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_1')
        conv5 = slim.conv2d(conv5, 512, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv5_2')

        up6 = upsample_and_concat(conv5, conv4, 256, 512)
        conv6 = slim.conv2d(up6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_1')
        conv6 = slim.conv2d(conv6, 256, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv6_2')

        up7 = upsample_and_concat(conv6, conv3, 128, 256)
        conv7 = slim.conv2d(up7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_1')
        conv7 = slim.conv2d(conv7, 128, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv7_2')
        conv7 = attention(conv7, 128, 'a3')

        up8 = upsample_and_concat(conv7, conv2, 64, 128)
        conv8 = slim.conv2d(up8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_1')
        conv8 = slim.conv2d(conv8, 64, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv8_2')
        conv8 = attention(conv8, 64, 'a4')


        up9 = upsample_and_concat(conv8, conv1, 32, 64)
        conv9 = slim.conv2d(up9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_1')
        conv9 = slim.conv2d(conv9, 32, [3, 3], rate=1, activation_fn=lrelu, scope='g_conv9_2')

        conv10 = slim.conv2d(conv9, 12, [1, 1], rate=1, activation_fn=None, scope='g_conv10')
        out = tf.depth_to_space(conv10, 2)
    return out
    
def adversarial1(image_):

    with tf.variable_scope("discriminator"):
        print(' image_:', image_.shape)
        conv1 = _conv_layer(image_, 48, 11, 4, batch_nn = False)
        conv2 = _conv_layer(conv1, 128, 5, 2)
        print('adv: ', conv2.shape)
        conv2 = attention(conv2, 128, 'd1')
        conv3 = _conv_layer(conv2, 192, 3, 1)
        conv4 = _conv_layer(conv3, 192, 3, 1)
        conv5 = _conv_layer(conv4, 128, 3, 2)
        
        #conv5 = attention(conv5, 128, 'd2')
       
       
        flat_size = 128 * 19 * 19
        conv5_flat = tf.reshape(conv5, [-1, flat_size])

        W_fc = tf.Variable(tf.truncated_normal([flat_size, 1024], stddev=0.01))
        bias_fc = tf.Variable(tf.constant(0.01, shape=[1024]))

        fc = leaky_relu(tf.matmul(conv5_flat, W_fc) + bias_fc)

        W_out = tf.Variable(tf.truncated_normal([1024, 2], stddev=0.01))
        bias_out = tf.Variable(tf.constant(0.01, shape=[2]))

        adv_out = tf.nn.softmax(tf.matmul(fc, W_out) + bias_out)
    
    return adv_out

def adversarial(image_):

    with tf.variable_scope("discriminator"):
        print(' image_:', image_.shape)
        conv1 = _conv_layer(image_, 48, 11, 4, batch_nn = False)
        conv2 = _conv_layer(conv1, 128, 5, 2)
        print('adv: ', conv2.shape)
        conv2 = attention(conv2, 128, 'd1')
        conv3 = _conv_layer(conv2, 192, 3, 1)
        conv4 = _conv_layer(conv3, 192, 3, 1)
        conv5 = _conv_layer(conv4, 128, 3, 2)
        conv6 = _conv_layer(conv5, 128, 3, 2)
        #conv5 = attention(conv5, 128, 'd2')
       
        print(conv6.shape) 
        flat_size = 128 * 10 * 10
        conv5_flat = tf.reshape(conv6, [-1, flat_size])

        W_fc = tf.Variable(tf.truncated_normal([flat_size, 1024], stddev=0.01))
        bias_fc = tf.Variable(tf.constant(0.01, shape=[1024]))

        fc = leaky_relu(tf.matmul(conv5_flat, W_fc) + bias_fc)

        W_out = tf.Variable(tf.truncated_normal([1024, 2], stddev=0.01))
        bias_out = tf.Variable(tf.constant(0.01, shape=[2]))

        adv_out = tf.nn.softmax(tf.matmul(fc, W_out) + bias_out)
    
    return adv_out

def resnet(input_image):

    with tf.variable_scope("generator"):

        W1 = weight_variable([9, 9, 3, 64], name="W1"); b1 = bias_variable([64], name="b1");
        c1 = tf.nn.relu(conv2d(input_image, W1) + b1)

        # residual 1

        W2 = weight_variable([3, 3, 64, 64], name="W2"); b2 = bias_variable([64], name="b2");
        c2 = tf.nn.relu(_instance_norm(conv2d(c1, W2) + b2))

        W3 = weight_variable([3, 3, 64, 64], name="W3"); b3 = bias_variable([64], name="b3");
        c3 = tf.nn.relu(_instance_norm(conv2d(c2, W3) + b3)) + c1

        # residual 2

        W4 = weight_variable([3, 3, 64, 64], name="W4"); b4 = bias_variable([64], name="b4");
        c4 = tf.nn.relu(_instance_norm(conv2d(c3, W4) + b4))

        W5 = weight_variable([3, 3, 64, 64], name="W5"); b5 = bias_variable([64], name="b5");
        c5 = tf.nn.relu(_instance_norm(conv2d(c4, W5) + b5)) + c3

        # residual 3

        W6 = weight_variable([3, 3, 64, 64], name="W6"); b6 = bias_variable([64], name="b6");
        c6 = tf.nn.relu(_instance_norm(conv2d(c5, W6) + b6))

        W7 = weight_variable([3, 3, 64, 64], name="W7"); b7 = bias_variable([64], name="b7");
        c7 = tf.nn.relu(_instance_norm(conv2d(c6, W7) + b7)) + c5

        # residual 4

        W8 = weight_variable([3, 3, 64, 64], name="W8"); b8 = bias_variable([64], name="b8");
        c8 = tf.nn.relu(_instance_norm(conv2d(c7, W8) + b8))

        W9 = weight_variable([3, 3, 64, 64], name="W9"); b9 = bias_variable([64], name="b9");
        c9 = tf.nn.relu(_instance_norm(conv2d(c8, W9) + b9)) + c7

        # Convolutional

        W10 = weight_variable([3, 3, 64, 64], name="W10"); b10 = bias_variable([64], name="b10");
        c10 = tf.nn.relu(conv2d(c9, W10) + b10)

        W11 = weight_variable([3, 3, 64, 64], name="W11"); b11 = bias_variable([64], name="b11");
        c11 = tf.nn.relu(conv2d(c10, W11) + b11)

        # Final

        W12 = weight_variable([9, 9, 64, 3], name="W12"); b12 = bias_variable([3], name="b12");
        enhanced = tf.nn.tanh(conv2d(c11, W12) + b12) * 0.58 + 0.5

    return enhanced


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
    bias = tf.Variable(tf.constant(0.01, shape=[num_filters]))

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
