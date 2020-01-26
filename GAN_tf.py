from __future__ import print_function, division
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from scipy.io import savemat, loadmat

from utils import load_images_and_flow_1clip
from plot_entire_one_frame import flow_to_color

from ProgressBar import ProgressBar

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

p_keep = 0.7


def sample_images(dataset_name, in_flows, in_frames, out_flows, out_frames, epoch, batch_i):
    def scale_range(img):
        for i in range(img.shape[-1]):
            img[..., i] = (img[..., i] - np.min(img[..., i]))/(np.max(img[..., i]) - np.min(img[..., i]))
        return img

    assert len(np.unique([len(in_flows), len(in_frames), len(out_flows), len(out_frames)])) == 1
    os.makedirs('generated/%s' % dataset_name, exist_ok=True)
    r, c = 4, len(in_flows)

    gen_imgs = np.concatenate([0.5*in_frames+0.5, 0.5*out_frames+0.5, in_flows, out_flows])

    titles = ['in_frame', 'out_frame', 'in_flow', 'out_flow']
    assert len(titles) == r
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            if i < 2:
                axs[i, j].imshow(np.clip(gen_imgs[cnt], 0., 1.))
            else:
                axs[i, j].imshow(scale_range(gen_imgs[cnt]))
            axs[i, j].set_title(titles[i])
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig("generated/%s/%d_%d.png" % (dataset_name, epoch, batch_i))
    plt.close()


def conv2d(x, out_channel, filter_size=3, stride=1, scope=None, return_filters=False):
    if isinstance(filter_size, int):
        filter_size = (filter_size, filter_size)
    assert len(filter_size) == 2
    with tf.variable_scope(scope):
        in_channel = x.get_shape()[-1]
        w = tf.get_variable('w', [filter_size[0], filter_size[1], in_channel, out_channel], initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable('b', [out_channel], initializer=tf.constant_initializer(0.0))
        result = tf.nn.conv2d(x, w, [1, stride, stride, 1], 'SAME') + b
        if return_filters:
            return result, w, b
        return result


def conv_transpose(x, output_shape, filter_size=3, scope=None, return_filters=False):
    if isinstance(filter_size, int):
        filter_size = (filter_size, filter_size)
    assert len(filter_size) == 2
    with tf.variable_scope(scope):
        w = tf.get_variable('w', [filter_size[0], filter_size[1], output_shape[-1], x.get_shape()[-1]],
                            initializer=tf.truncated_normal_initializer(stddev=0.02))
        b = tf.get_variable('b', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        convt = tf.nn.bias_add(tf.nn.conv2d_transpose(x, w, output_shape=output_shape, strides=[1, 2, 2, 1]), b)
        if return_filters:
            return convt, w, b
        return convt


def conv2d_Inception(x, out_channel, max_filter_size=7, scope=None):
    assert max_filter_size % 2 == 1 and max_filter_size < 8
    n_branch = (max_filter_size+1) // 2
    assert out_channel % n_branch == 0
    nf_branch = out_channel // n_branch
    with tf.variable_scope(scope):
        # 1x1
        s1_11 = conv2d(x, nf_branch, filter_size=(1, 1), scope='s1_11')
        if n_branch == 1:
            return s1_11
        # 3x3
        s3_11 = conv2d(x, nf_branch, filter_size=(1, 1), scope='s3_11')
        s3_1n = conv2d(s3_11, nf_branch, filter_size=(1, 3), scope='s3_1n')
        s3_n1 = conv2d(s3_1n, nf_branch, filter_size=(3, 1), scope='s3_n1')
        if n_branch == 2:
            return tf.concat([s1_11, s3_n1], -1)
        # 5x5
        s5_11 = conv2d(x, nf_branch, filter_size=(1, 1), scope='s5_11')
        s5_1n = conv2d(s5_11, nf_branch, filter_size=(1, 3), scope='s5_1n_1')
        s5_n1 = conv2d(s5_1n, nf_branch, filter_size=(3, 1), scope='s5_n1_1')
        s5_1n = conv2d(s5_n1, nf_branch, filter_size=(1, 3), scope='s5_1n_2')
        s5_n1 = conv2d(s5_1n, nf_branch, filter_size=(3, 1), scope='s5_n1_2')
        if n_branch == 3:
            return tf.concat([s1_11, s3_n1, s5_n1], -1)
        # 7x7
        s7_11 = conv2d(x, nf_branch, filter_size=(1, 1), scope='s7_11')
        s7_1n = conv2d(s7_11, nf_branch, filter_size=(1, 3), scope='s7_1n_1')
        s7_n1 = conv2d(s7_1n, nf_branch, filter_size=(3, 1), scope='s7_n1_1')
        s7_1n = conv2d(s7_n1, nf_branch, filter_size=(1, 3), scope='s7_1n_2')
        s7_n1 = conv2d(s7_1n, nf_branch, filter_size=(3, 1), scope='s7_n1_2')
        s7_1n = conv2d(s7_n1, nf_branch, filter_size=(1, 3), scope='s7_1n_3')
        s7_n1 = conv2d(s7_1n, nf_branch, filter_size=(3, 1), scope='s7_n1_3')
        return tf.concat([s1_11, s3_n1, s5_n1, s7_n1], -1)


# 128 * 192
def Generator(input_data, is_training, keep_prob, return_layers=False):

    def G_conv_bn_relu(x, out_channel, filter_size, stride=2, training=False, bn=True, scope=None):
        with tf.variable_scope(scope):
            d = conv2d(x, out_channel, filter_size=filter_size, stride=stride, scope='conv')
            if bn:
                d = tf.layers.batch_normalization(d, training=training)
            d = tf.nn.leaky_relu(d)
            return d

    def G_deconv_bn_dr_relu_concat(layer_input, skip_input, out_shape, filter_size, p_keep_drop, training=False, scope=None):
        with tf.variable_scope(scope):
            """Layers used during upsampling"""
            u = conv_transpose(layer_input, out_shape, filter_size=filter_size, scope='deconv')
            u = tf.layers.batch_normalization(u, training=training)
            u = tf.nn.dropout(u, p_keep_drop)
            u = tf.nn.relu(u)
            if skip_input is not None:
                u = tf.concat([u, skip_input], -1)
            return u

    with tf.variable_scope('generator'):
        b_size = tf.shape(input_data)[0]
        h = tf.shape(input_data)[1]
        w = tf.shape(input_data)[2]

        h0 = input_data
        filters = 64
        filter_size = (4, 4)
        '''COMMON ENCODER'''
        h0 = conv2d_Inception(h0, filters, max_filter_size=7, scope='gen_h0')
        h1 = G_conv_bn_relu(h0, filters, filter_size, stride=1, training=is_training, bn=False, scope='gen_h1')
        h2 = G_conv_bn_relu(h1, filters*2, filter_size, stride=2, training=is_training, bn=True, scope='gen_h2')
        h3 = G_conv_bn_relu(h2, filters*4, filter_size, stride=2, training=is_training, bn=True, scope='gen_h3')
        h4 = G_conv_bn_relu(h3, filters*8, filter_size, stride=2, training=is_training, bn=True, scope='gen_h4')
        h5 = G_conv_bn_relu(h4, filters*8, filter_size, stride=2, training=is_training, bn=True, scope='gen_h5')

        '''Unet DECODER for OPTICAL FLOW'''
        h4fl = G_deconv_bn_dr_relu_concat(h5, h4, [b_size, h//8, w//8, filters*4], filter_size, keep_prob, training=is_training, scope='gen_h4fl')
        h3fl = G_deconv_bn_dr_relu_concat(h4fl, h3, [b_size, h//4, w//4, filters*4], filter_size, keep_prob, training=is_training, scope='gen_h3fl')
        h2fl = G_deconv_bn_dr_relu_concat(h3fl, h2, [b_size, h//2, w//2, filters*2], filter_size, keep_prob, training=is_training, scope='gen_h2fl')
        h1fl = G_deconv_bn_dr_relu_concat(h2fl, h1, [b_size, h, w, filters], filter_size, keep_prob, training=is_training, scope='gen_h1fl')
        out_flow = conv2d(h1fl, 3, filter_size=3, stride=1, scope='gen_flow')

        '''Unet DECODER for FRAME'''
        h4fr = G_deconv_bn_dr_relu_concat(h5, None, [b_size, h//8, w//8, filters*4], filter_size, keep_prob, training=is_training, scope='gen_h4fr')
        h3fr = G_deconv_bn_dr_relu_concat(h4fr, None, [b_size, h//4, w//4, filters*4], filter_size, keep_prob, training=is_training, scope='gen_h3fr')
        h2fr = G_deconv_bn_dr_relu_concat(h3fr, None, [b_size, h//2, w//2, filters*2], filter_size, keep_prob, training=is_training, scope='gen_h2fr')
        h1fr = G_deconv_bn_dr_relu_concat(h2fr, None, [b_size, h, w, filters], filter_size, keep_prob, training=is_training, scope='gen_h1fr')
        out_frame = conv2d(h1fr, input_data.get_shape()[-1], filter_size=3, stride=1, scope='gen_frame')
        #
        if return_layers:
            return out_flow, out_frame, [h0, h1, h2, h3, h4, h5, h4fl, h3fl, h2fl, h1fl, h4fr, h3fr, h2fr, h1fr]
        return out_flow, out_frame


# 128*192
def Discriminator(frame_true, flow_hat, is_training, reuse=False, return_middle_layers=False):

    def D_conv_bn_active(x, out_channel, filter_size, stride=2, training=False, bn=True, active=tf.nn.leaky_relu, scope=None):
        with tf.variable_scope(scope):
            d = conv2d(x, out_channel, filter_size=filter_size, stride=stride, scope='conv')
            if bn:
                d = tf.layers.batch_normalization(d, training=training)
            if active is not None:
                d = active(d)
            return d

    with tf.variable_scope('discriminator') as var_scope:
        if reuse:
            var_scope.reuse_variables()

        filters = 64
        filter_size = (4, 4)

        h0 = tf.concat([frame_true, flow_hat], -1)
        h1 = D_conv_bn_active(h0, filters, filter_size, stride=2, training=is_training, bn=False, scope='dis_h1')
        h2 = D_conv_bn_active(h1, filters*2, filter_size, stride=2, training=is_training, bn=True, scope='dis_h2')
        h3 = D_conv_bn_active(h2, filters*4, filter_size, stride=2, training=is_training, bn=True, scope='dis_h3')
        h4 = D_conv_bn_active(h3, filters*8, filter_size, stride=2, training=is_training, bn=True, active=None, scope='dis_h4')

        if return_middle_layers:
            return tf.nn.sigmoid(h4), h4, [h1, h2, h3]
        return tf.nn.sigmoid(h4), h4


def train_Unet_naive_with_batch_norm(training_images, training_flows, max_epoch, dataset_name='', start_model_idx=0, batch_size=16):

    print('no. of images = %s' % len(training_images))
    assert len(training_images) == len(training_flows)
    h, w = training_images.shape[1:3]
    assert h < w
    training_images /= 0.5
    training_images -= 1.

    plh_frame_true = tf.placeholder(tf.float32, shape=[None, h, w, 3])
    plh_flow_true = tf.placeholder(tf.float32, shape=[None, h, w, 3])
    plh_is_training = tf.placeholder(tf.bool)

    # generator
    plh_dropout_prob = tf.placeholder_with_default(1.0, shape=())
    output_opt, output_appe = Generator(plh_frame_true, plh_is_training, plh_dropout_prob)

    # discriminator
    D_real, D_real_logits = Discriminator(plh_frame_true, plh_flow_true, plh_is_training, reuse=False)
    D_fake, D_fake_logits = Discriminator(plh_frame_true, output_opt, plh_is_training, reuse=True)

    # appearance loss
    dy1, dx1 = tf.image.image_gradients(output_appe)
    dy0, dx0 = tf.image.image_gradients(plh_frame_true)
    loss_inten = tf.reduce_mean((output_appe - plh_frame_true)**2)
    loss_gradi = tf.reduce_mean(tf.abs(tf.abs(dy1)-tf.abs(dy0)) + tf.abs(tf.abs(dx1)-tf.abs(dx0)))
    loss_appe = loss_inten + loss_gradi

    # optical loss
    loss_opt = tf.reduce_mean(tf.abs(output_opt - plh_flow_true))

    # GAN loss
    D_loss = 0.5*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones_like(D_real))) + \
             0.5*tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros_like(D_fake)))
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones_like(D_fake)))
    G_loss_total = 0.25*G_loss + loss_appe + 2*loss_opt

    # optimizers
    t_vars = tf.trainable_variables()
    g_vars = [var for var in t_vars if 'gen_' in var.name]
    d_vars = [var for var in t_vars if 'dis_' in var.name]

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        D_optimizer = tf.train.AdamOptimizer(learning_rate=0.00002, beta1=0.5, beta2=0.9, name='AdamD').minimize(D_loss, var_list=d_vars)
        G_optimizer = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5, beta2=0.9, name='AdamG').minimize(G_loss_total, var_list=g_vars)
    init_op = tf.global_variables_initializer()

    # tensorboard
    tf.summary.scalar('D_loss', D_loss)
    tf.summary.scalar('G_loss', G_loss)
    tf.summary.scalar('appe_loss', loss_appe)
    tf.summary.scalar('opt_loss', loss_opt)
    merge = tf.summary.merge_all()

    #
    saver = tf.train.Saver(max_to_keep=30)
    with tf.Session() as sess:
        losses = np.array([], dtype=np.float32).reshape((0, 4))
        sess.run(init_op)
        if start_model_idx > 0:
            saver.restore(sess, './training_saver/%s/model_ckpt_%d.ckpt' % (dataset_name, start_model_idx))
            losses = np.loadtxt('./training_saver/%s/train_loss_%d.txt' % (dataset_name, start_model_idx), delimiter=',')
        # define log path for tensorboard
        tensorboard_path = './training_saver/%s/logs/2/train' % (dataset_name)
        if not os.path.exists(tensorboard_path):
            pathlib.Path(tensorboard_path).mkdir(parents=True, exist_ok=True)
        train_writer = tf.summary.FileWriter(tensorboard_path, sess.graph)
        print('Run: tensorboard --logdir logs/2')
        # executive training stage

        for i in range(start_model_idx, max_epoch):
            tf.set_random_seed(i)
            np.random.seed(i)
            batch_idx = np.array_split(np.random.permutation(len(training_images)), np.ceil(len(training_images)/batch_size))
            for j in range(len(batch_idx)):
                # discriminator
                _, curr_D_loss, summary = sess.run([D_optimizer, D_loss, merge],
                                                   feed_dict={plh_frame_true: training_images[batch_idx[j]],
                                                              plh_flow_true: training_flows[batch_idx[j]],
                                                              plh_is_training: True})
                if j % 10 == 0:
                    _, curr_G_loss, curr_loss_appe, curr_loss_opt, curr_gen_frames, curr_gen_flows, summary = \
                                    sess.run([G_optimizer, G_loss, loss_appe, loss_opt, output_appe[:4], output_opt[:4], merge],
                                             feed_dict={plh_frame_true: training_images[batch_idx[j]],
                                                        plh_flow_true: training_flows[batch_idx[j]],
                                                        plh_is_training: True,
                                                        plh_dropout_prob: p_keep})
                    sample_images(dataset_name, training_flows[batch_idx[j][:4]], training_images[batch_idx[j][:4]],
                                  curr_gen_flows, curr_gen_frames, i, j)

                else:
                    _, curr_G_loss, curr_loss_appe, curr_loss_opt, summary = sess.run([G_optimizer, G_loss, loss_appe, loss_opt, merge],
                                                                                      feed_dict={plh_frame_true: training_images[batch_idx[j]],
                                                                                                 plh_flow_true: training_flows[batch_idx[j]],
                                                                                                 plh_is_training: True,
                                                                                                 plh_dropout_prob: p_keep})
                # write log for tensorboard
                train_writer.add_summary(summary, i*len(batch_idx)+j)
                train_writer.flush()
                print('epoch %d/%d, iter %3d/%d: D_loss = %.4f, G_loss = %.4f, loss_appe = %.4f, loss_flow = %.4f'
                      % (i+1, max_epoch, j+1, len(batch_idx), curr_D_loss, curr_G_loss, curr_loss_appe, curr_loss_opt))
                if np.isnan(curr_D_loss) or np.isnan(curr_G_loss) or np.isnan(curr_loss_appe) or np.isnan(curr_loss_opt):
                    return
                losses = np.concatenate((losses, [[curr_D_loss, curr_G_loss, curr_loss_appe, curr_loss_opt]]), axis=0)
        saver.save(sess, './training_saver/%s/model_ckpt_%d.ckpt' % (dataset_name, i+1))
        np.savetxt('./training_saver/%s/train_loss_%d.txt' % (dataset_name, i+1), losses, delimiter=',')


def test_Unet_naive_with_batch_norm(test_images, test_flows, h, w, dataset, sequence_n_frame,
                                    clip_idx, batch_size=32, model_idx=20, using_test_data=True):
    print(test_images.shape, test_flows.shape, np.sum(sequence_n_frame))
    assert len(test_images) == len(test_flows)
    assert len(test_images) == sequence_n_frame[clip_idx]

    test_images /= 0.5
    test_images -= 1.

    plh_frame_true = tf.placeholder(tf.float32, shape=[None, h, w, 3])
    plh_is_training = tf.placeholder(tf.bool)

    # generator
    plh_dropout_prob = tf.placeholder_with_default(1.0, shape=())
    output_opt, output_appe = Generator(plh_frame_true, plh_is_training, plh_dropout_prob)

    saver = tf.train.Saver(max_to_keep=20)

    saved_out_appes = np.zeros(test_images.shape)
    saved_out_flows = np.zeros(test_flows.shape)

    with tf.Session() as sess:
        saved_model_file = './training_saver/%s/model_ckpt_%d.ckpt' % (dataset['name'], model_idx)
        saver.restore(sess, saved_model_file)
        #
        saved_data_path = './training_saver/%s/output_%s/%d_epoch' % (dataset['name'], 'test' if using_test_data else 'train', model_idx)
        if not os.path.exists(saved_data_path):
            pathlib.Path(saved_data_path).mkdir(parents=True, exist_ok=True)

        saved_data_file = '%s/output_%d.npz' % (saved_data_path, clip_idx)
        if os.path.isfile(saved_data_file):
            print('File existed! Return!')
            return

        batch_idx = np.array_split(np.arange(len(test_images)), np.ceil(len(test_images)/batch_size))
        #
        progress = ProgressBar(len(batch_idx), fmt=ProgressBar.FULL)
        for j in range(len(batch_idx)):
            progress.current += 1
            progress()
            saved_out_appes[batch_idx[j]], saved_out_flows[batch_idx[j]] = \
                sess.run([output_appe, output_opt],
                         feed_dict={plh_frame_true: test_images[batch_idx[j]],
                                    plh_is_training: False,
                                    plh_dropout_prob: 1.0})
            saved_out_appes[batch_idx[j]] = 0.5*(saved_out_appes[batch_idx[j]] + 1)
        progress.done()

    np.savez_compressed(saved_data_file, image=saved_out_appes, flow=saved_out_flows)


def visualize_layers_filters(img_paths, test_images, h, w, dataset, layer_idx, model_idx=20):
    def convert_to_visualize(img, only_clip=False, gamma=None):
        if only_clip:
            return np.clip(img, 0.0, 1.0)
        if len(img.shape) == 2:
            img = (img-np.min(img)) / (np.max(img) - np.min(img))
        else:
            img = np.dstack([(img[..., i]-np.min(img[..., i])) / (np.max(img[..., i]) - np.min(img[..., i])) for i in range(img.shape[-1])])
        if gamma is not None:
            img = img**(1./gamma)
        return img

    assert len(img_paths) == len(test_images)
    print(test_images.shape)

    test_images /= 0.5
    test_images -= 1.

    plh_frame_true = tf.placeholder(tf.float32, shape=[None, h, w, 3])
    plh_is_training = tf.placeholder(tf.bool)

    # generator
    plh_dropout_prob = tf.placeholder_with_default(1.0, shape=())
    output_opt, output_appe, layers = Generator(plh_frame_true, plh_is_training, plh_dropout_prob, return_layers=True)
    if layer_idx is not None:
        layers = layers[layer_idx]

    feature_maps = [2, 4, 6]
    n_feature_map = len(feature_maps)

    saver = tf.train.Saver(max_to_keep=20)
    with tf.Session() as sess:
        saved_model_file = './training_saver/%s/model_ckpt_%d.ckpt' % (dataset['name'], model_idx)
        saver.restore(sess, saved_model_file)
        #
        output_frames, output_optics, output_layers = sess.run([output_appe, output_opt, layers],
                                                               feed_dict={plh_frame_true: test_images,
                                                                          plh_is_training: False,
                                                                          plh_dropout_prob: 1.0})
        test_images = test_images * 0.5 + 0.5
        output_frames = output_frames * 0.5 + 0.5
        print('output_layers:', len(output_layers), [x.shape for x in output_layers])
        for k in range(len(test_images)):
            out_dict = dict()
            r, c = len(output_layers), n_feature_map
            fig, axs = plt.subplots(r, c)
            for i in range(r):
                out_dict['layer_%d' % i] = output_layers[i][k]
                for j in range(c):
                    print(i, k, j, output_layers[i].shape)
                    axs[i, j].imshow(convert_to_visualize(output_layers[i][k, :, :, feature_maps[j]]), cmap='autumn')
                    axs[i, j].axis('off')
            savemat('%s_%s.mat' % (img_paths[k][:-4], dataset['name']), out_dict)
            plt.show()
            continue

            plt.figure()
            plt.subplot(221), plt.imshow(test_images[0]), plt.axis('off')
            plt.subplot(222), plt.imshow(output_frames[0]), plt.axis('off')
            plt.subplot(223), plt.imshow(np.mean(abs(test_images[0] - output_frames[0]), axis=-1), 'jet'), plt.axis('off')
            plt.subplot(224)
            plt.imshow(test_images[0])
            plt.imshow(np.mean(abs(test_images[0] - output_frames[0]), axis=-1), cmap='jet', alpha=0.45)
            plt.axis('off')
            plt.show()


def visualize_epoch_output(h, w, dataset, frame_idx, clip_idx, model_idx, show_output=False):
    #
    image_data, flow_data = load_images_and_flow_1clip(dataset, clip_idx, train=False)
    assert frame_idx in np.arange(len(flow_data))
    test_image = image_data[frame_idx]
    test_flow = flow_data[frame_idx]
    #
    saved_data_file = 'img_samples/out_each_epoch/%s_clip_%d_frame_%d.mat' % (dataset['name'], clip_idx, frame_idx)
    if os.path.isfile(saved_data_file):
        data = loadmat(saved_data_file)
    else:
        data = dict()
        data['appe'] = test_image
        data['flow'] = test_flow
    #
    test_image /= 0.5
    test_image -= 1.

    plh_frame_true = tf.placeholder(tf.float32, shape=[None, h, w, 3])
    plh_is_training = tf.placeholder(tf.bool)

    # generator
    plh_dropout_prob = tf.placeholder_with_default(1.0, shape=())
    output_opt, output_appe = Generator(plh_frame_true, plh_is_training, plh_dropout_prob)

    saver = tf.train.Saver(max_to_keep=20)
    with tf.Session() as sess:
        saved_model_file = './training_saver/%s/model_ckpt_%d.ckpt' % (dataset['name'], model_idx)
        saver.restore(sess, saved_model_file)
        #
        output_frame, output_optic = sess.run([output_appe, output_opt],
                                              feed_dict={plh_frame_true: [test_image],
                                                         plh_is_training: False,
                                                         plh_dropout_prob: 1.0})
        test_image = test_image * 0.5 + 0.5
        output_frame = output_frame[0] * 0.5 + 0.5
        output_optic = output_optic[0]
        print(model_idx, output_frame.shape)
        if show_output:
            plt.figure()
            plt.subplot(221), plt.imshow(test_image)
            plt.subplot(222), plt.imshow(output_frame)
            plt.subplot(223), plt.imshow(flow_to_color(test_flow))
            plt.subplot(224), plt.imshow(flow_to_color(output_optic))
            plt.show()
        data['appe_%d' % model_idx] = output_frame
        data['flow_%d' % model_idx] = output_optic
        savemat(saved_data_file, data)
