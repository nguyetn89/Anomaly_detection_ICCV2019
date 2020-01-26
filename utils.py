import os
import glob
import pathlib
import copy
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from sklearn.utils.fixes import signature

from scipy.misc import imread
from scipy.io import loadmat
from skimage.measure import compare_ssim as ssim


def resize(datum, size):
    if len(datum.shape) == 2:
        return cv.resize(datum.astype(float), tuple(size))
    elif len(datum.shape) == 3:
        return np.dstack([cv.resize(datum[:, :, i].astype(float), tuple(size)) for i in range(datum.shape[-1])])
    else:
        print('unexpected data size', datum.shape)
        return None


def extend_gray_channel(datum):
    if len(datum.shape) < 3 or datum.shape[2] == 1:
        return np.dstack((datum, datum, datum))
    return datum


def extend_mag_channel(datum):
    mag, _ = cv.cartToPolar(datum[:, :, 0], datum[:, :, 1])
    return np.concatenate((datum, np.expand_dims(mag, axis=2)), axis=-1)


def load_images_and_flows(dataset, new_size=[128, 192], train=True, force_recalc=False):
    img_dir = dataset['path_train' if train else 'path_test']
    flow_files = sorted(glob.glob(dataset['path_train' if train else 'path_test'] + ('/Train*.npz' if train else '/Test*.npz')))
    n_images = np.sum(count_sequence_n_frame(dataset, test=not train) - 1)  # check: e.g. 15312 for Avenue
    print('number of images: ', n_images)
    resized_image_data = np.empty((n_images, new_size[0], new_size[1], 3), dtype=np.float32)
    resized_flow_data = np.empty((n_images, new_size[0], new_size[1], 3), dtype=np.float32)
    n_clip = dataset['n_clip_train' if train else 'n_clip_test']
    #
    idx = 0
    assert len(flow_files) == n_clip
    for i in range(n_clip):
        clip_path = '%s/%s%s/' % (img_dir, 'Train' if train else 'Test', str(i+1).zfill(3))
        print(clip_path)
        # image
        img_files = sorted(glob.glob(clip_path + '*.tif'))[:-1]
        saved_image_file = '%s/%s_image_clip_%d.npz' % (img_dir, 'training' if train else 'test', i+1)
        if os.path.isfile(saved_image_file) and not force_recalc:
            image_data = np.load(saved_image_file)['image']
        else:
            image_data = np.array([extend_gray_channel(resize(imread(img_file)/255., (new_size[1], new_size[0])))
                                   for img_file in img_files]).astype(np.float32)
            np.savez_compressed(saved_image_file, image=image_data)
        resized_image_data[idx:idx+len(image_data)] = image_data
        # flow
        flow_data = np.load(flow_files[i])['data']
        saved_flow_file = '%s/%s_flow_clip_%d.npz' % (img_dir, 'training' if train else 'test', i+1)
        if os.path.isfile(saved_flow_file) and not force_recalc:
            flow_data = np.load(saved_flow_file)['flow']
        else:
            flow_data = np.array([extend_mag_channel(resize(flow_datum, (new_size[1], new_size[0]))) for flow_datum in flow_data]).astype(np.float32)
            np.savez_compressed(saved_flow_file, flow=flow_data)
        resized_flow_data[idx:idx+len(flow_data)] = flow_data
        #
        assert len(image_data) == len(flow_data)
        idx += len(image_data)
        print('clip', i+1, image_data.shape, flow_data.shape)

    return resized_image_data, resized_flow_data


def load_images_and_flow_1clip(dataset, clip_idx, train=False):
    img_dir = dataset['path_train' if train else 'path_test']
    saved_image_file = '%s/%s_image_clip_%d.npz' % (img_dir, 'training' if train else 'test', clip_idx + 1)
    saved_flow_file = '%s/%s_flow_clip_%d.npz' % (img_dir, 'training' if train else 'test', clip_idx + 1)
    print(saved_image_file)
    assert os.path.isfile(saved_image_file)
    assert os.path.isfile(saved_flow_file)
    image_data = np.load(saved_image_file)['image']
    flow_data = np.load(saved_flow_file)['flow']
    return image_data, flow_data


# get sequence of number of clip's frames
def count_sequence_n_frame(dataset, test=True):
    sequence_n_frame = np.zeros(dataset['n_clip_test' if test else 'n_clip_train'], dtype=int)
    for i in range(len(sequence_n_frame)):
        clip_path = '%s/%s%s/' % (dataset['path_test' if test else 'path_train'], 'Test' if test else 'Train', str(i+1).zfill(3))
        sequence_n_frame[i] = len(sorted(glob.glob(clip_path + '*.tif')))
    return sequence_n_frame


# 1: abnormal, 0: normal
def get_test_frame_labels(ground_truth, sequence_n_frame):
    assert len(ground_truth) == len(sequence_n_frame)
    labels_exclude_last = np.zeros(0, dtype=int)
    labels_exclude_first = np.zeros(0, dtype=int)
    labels_full = np.zeros(0, dtype=int)
    for i in range(len(sequence_n_frame)):
        seg = ground_truth[i]
        tmp_labels = np.zeros(sequence_n_frame[i])
        for j in range(0, len(seg), 2):
            tmp_labels[(seg[j]-1):seg[j+1]] = 1
        labels_exclude_last = np.append(labels_exclude_last, tmp_labels[:-1])
        labels_exclude_first = np.append(labels_exclude_first, tmp_labels[1:])
        labels_full = np.append(labels_full, tmp_labels)
    return labels_exclude_last, labels_exclude_first, labels_full


# POST-PROCESSING
def plot_error_map(dataset, clip_idx, frame_idx, model_idx, using_test_data=True, figure_idx=1):
    images, flows = load_images_and_flow_1clip(dataset, clip_idx, train=not using_test_data)
    print(images.shape, flows.shape)

    #
    if isinstance(frame_idx, int):
        frame_idx = [frame_idx]

    #
    images, flows = images[frame_idx], flows[frame_idx]

    # load outputted results
    saved_data_path = './training_saver/%s/output_%s/%d_epoch' % (dataset['name'], 'test' if using_test_data else 'train', model_idx)
    saved_data_file = '%s/output_%d.npz' % (saved_data_path, clip_idx)
    if not os.path.isfile(saved_data_file):
        print('File not existed! Return!')
        return
    loader = np.load(saved_data_file)
    images_hat, flows_hat = loader['image'][frame_idx], loader['flow'][frame_idx]

    # plot
    def scale_range(img):
        for i in range(img.shape[-1]):
            img[..., i] = (img[..., i] - np.min(img[..., i]))/(np.max(img[..., i]) - np.min(img[..., i]))
        return img

    r, c = 6, len(frame_idx)

    gen_imgs = np.concatenate([images, images_hat, np.absolute(images-images_hat)**2, flows, flows_hat, np.absolute(flows-flows_hat)**2])

    plt.figure(figure_idx)
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            if i < 2:
                axs[i, j].imshow(np.clip(gen_imgs[cnt], 0., 1.))
            elif i == 2:
                axs[i, j].imshow(np.mean(gen_imgs[cnt], axis=2), cmap='gray')
            elif i < 5:
                axs[i, j].imshow(scale_range(gen_imgs[cnt]))
            else:
                axs[i, j].imshow(np.mean(gen_imgs[cnt], axis=2))
                # axs[i, j].imshow(gen_imgs[cnt][..., -1], cmap='gray')
            # axs[i, j].set_title(titles[i])
            axs[i, j].axis('off')
            cnt += 1
    plt.draw()


# ASSESSMENT
def calc_anomaly_score_one_frame(frame_true, frame_hat, flow_true, flow_hat, thresh_cut_off=[0, 0, 0], return_as_map=False, operation=np.mean):
    assert frame_true.shape == frame_hat.shape
    assert flow_true.shape == flow_hat.shape

    loss_appe = (frame_true-frame_hat)**2
    loss_flow = (flow_true-flow_hat)**2

    # calc angle of optical flow
    _, angle_true = cv.cartToPolar(flow_true[:, :, 0], flow_true[:, :, 1])
    _, angle_hat = cv.cartToPolar(flow_hat[:, :, 0], flow_hat[:, :, 1])
    angle_diff = np.absolute(angle_true - angle_hat)
    loss_angle_flow = np.min(np.array([angle_diff, 2*np.pi - angle_diff]), axis=0)**2
    assert loss_angle_flow.shape == flow_true.shape[:2]

    # cut-off low scores to check only high scores
    if thresh_cut_off is not None:
        assert len(thresh_cut_off) == 3
        loss_appe = np.clip(loss_appe, thresh_cut_off[0], None)
        loss_flow = np.clip(loss_flow, thresh_cut_off[1], None)
        loss_angle_flow = np.clip(loss_angle_flow, thresh_cut_off[2], None)

    # return score map for pixel-wise assessment
    if return_as_map:
        return operation(loss_appe, axis=-1), operation(loss_flow, axis=-1), loss_angle_flow

    def calc_measures_single_item(item_true, item_hat, squared_error, max_val_hat):
        PSNR_X = 10*np.log10(np.max(item_hat)**2/np.mean(squared_error))
        PSNR_inv = np.max(item_hat)**2 * np.mean(squared_error)
        PSNR = 10*np.log10(max_val_hat**2/np.mean(squared_error))
        SSIM = ssim(item_true, item_hat, data_range=np.max([item_true, item_hat])-np.min([item_true, item_hat]),
                    multichannel=len(item_true.shape) == 3 and item_true.shape[-1] > 1)
        stat_MSE = np.mean(squared_error)
        stat_maxSE = np.max(squared_error)
        stat_std = np.std(squared_error)
        stat_MSE_1channel = np.mean(np.sum(squared_error, axis=-1)) if len(squared_error.shape) == 3 else -1
        stat_maxSE_1channel = np.max(np.sum(squared_error, axis=-1)) if len(squared_error.shape) == 3 else -1
        stat_std_1channel = np.std(np.sum(squared_error, axis=-1)) if len(squared_error.shape) == 3 else -1
        return np.array([PSNR_X, PSNR_inv, PSNR, SSIM, stat_MSE, stat_maxSE, stat_std, stat_MSE_1channel, stat_maxSE_1channel, stat_std_1channel])

    scores_appe = calc_measures_single_item(frame_true, frame_hat, loss_appe, 1.0)
    scores_flow = calc_measures_single_item(flow_true, flow_hat, loss_flow, np.pi)
    scores_angle = calc_measures_single_item(angle_true, angle_hat, loss_angle_flow, np.pi)
    scores_mag = calc_measures_single_item(flow_true[..., -1], flow_hat[..., -1], (flow_true[..., -1] - flow_hat[..., -1])**2, 20.)

    return np.array([scores_appe, scores_flow, scores_angle, scores_mag])


def calc_anomaly_score(frames_true, frames_hat, flows_true, flows_hat):
    assert frames_true.shape == frames_hat.shape
    assert flows_true.shape == flows_hat.shape
    return np.array([calc_anomaly_score_one_frame(frames_true[i], frames_hat[i], flows_true[i], flows_hat[i])
                     for i in range(len(frames_true))])


# suitable for Avenue
def calc_score_one_clip(dataset, epoch, clip_idx, train=False, force_calc=False):
    saved_data_path = './training_saver/%s/output_%s/%d_epoch' % (dataset['name'], 'train' if train else 'test', epoch)
    saved_score_file = '%s/score_epoch_%d_clip_%d.npz' % (saved_data_path, epoch, clip_idx + 1)
    if not force_calc and os.path.isfile(saved_score_file):
        return np.load(saved_score_file)['data']
    # load true data and outputted data
    in_appe, in_flow = load_images_and_flow_1clip(dataset, clip_idx, train=train)
    saved_data_file = '%s/output_%d.npz' % (saved_data_path, clip_idx)
    out_loader = np.load(saved_data_file)
    out_appe, out_flow = out_loader['image'].astype(np.float32), out_loader['flow'].astype(np.float32)

    print(in_appe.shape, out_appe.shape, in_flow.shape, out_flow.shape)
    assert in_appe.shape == out_appe.shape
    assert in_flow.shape == out_flow.shape
    # calc score and save to file
    score_frame = calc_anomaly_score(in_appe, out_appe, in_flow, out_flow)
    np.savez_compressed(saved_score_file, data=score_frame)
    return score_frame


def calc_score_full_clips(dataset, epoch, train=False, force_calc=True):
    def flip_scores(scores):
        norm_scores = np.zeros_like(scores)
        for i in range(len(norm_scores)):
            norm_scores[i] = scores[i]
            norm_scores[i, :, 0] = 1./norm_scores[i, :, 0]  # PSNR_X
            norm_scores[i, :, 2] = 1./norm_scores[i, :, 2]  # PSNR
            norm_scores[i, :, 3] = 1./norm_scores[i, :, 3]  # SSIM
            # norm_scores[i,:,6] *= norm_scores[i,:,4]**1
            # norm_scores[i,:,9] *= norm_scores[i,:,7]**1
        return norm_scores

    saved_data_path = './training_saver/%s/output_%s/%d_epoch' % (dataset['name'], 'train' if train else 'test', epoch)
    saved_score_file = '%s_scores.npz' % saved_data_path
    if not force_calc and os.path.isfile(saved_score_file):
        return flip_scores(np.load(saved_score_file)['data'])
    # calc scores for all clips and save to file
    n_clip = dataset['n_clip_train' if train else 'n_clip_test']
    for i in range(n_clip):
        if i == 0:
            score_frame = calc_score_one_clip(dataset, epoch, i, train=train, force_calc=False)
        else:
            score_frame = np.concatenate((score_frame, calc_score_one_clip(dataset, epoch, i, train=train, force_calc=False)), axis=0)
    np.savez_compressed(saved_score_file, data=score_frame)
    return flip_scores(score_frame)


def get_weights(dataset, epoch, force_calc=True):
    if dataset is None:
        return None
    saved_data_path = './training_saver/%s/output_train/%d_epoch' % (dataset['name'], epoch)
    saved_score_file = '%s_scores.npz' % saved_data_path
    if os.path.isfile(saved_score_file) and not force_calc:
        training_scores = np.load(saved_score_file)['data']
    else:
        training_scores = calc_score_full_clips(dataset, epoch, train=True)
    return np.mean(training_scores, axis=0)


def basic_assess_AUC(scores, labels, plot_pr_idx=None):
    assert len(scores) == len(labels)
    if plot_pr_idx is not None:
        precision, recall, _ = precision_recall_curve(labels, scores[:, plot_pr_idx])
        print(len(np.where(labels == 0)[0]), len(np.where(labels == 1)[0]), len(np.unique(precision)), len(np.unique(recall)))
        step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.show()
    return np.array([roc_auc_score(labels, scores[:, i]) for i in range(scores.shape[1])]),
           np.array([average_precision_score(labels, scores[:, i]) for i in range(scores.shape[1])])


def save_SSIM_scores_subway(dataset, model_test, train=False):
    score_frame = calc_score_full_clips(dataset, model_test, train=train)
    scores_appe_SSIM = score_frame[:, 0, 3]
    print(scores_appe_SSIM.shape)
    saved_data_path = './training_saver/%s/output_test/%d_epoch' % (dataset['name'], model_test)
    saved_score_file = '%s/appe_SSIM_scores.npz' % saved_data_path
    np.savez(saved_score_file, data=scores_appe_SSIM)


def full_assess_AUC(dataset, score_frame, frame_labels, w_img=0.5, w_flow=0.5, sequence_n_frame=None,
                    clip_normalize=True, use_pr=False, selected_score_estimation_ways=None, save_pr_appe_SSIM_epoch=None):
    def normalize_clip_scores(scores, ver=1):
        assert ver in [1, 2]
        if ver == 1:
            return [item/np.max(item, axis=0) for item in scores]
        else:
            return [(item-np.min(item, axis=0))/(np.max(item, axis=0)-np.min(item, axis=0)) for item in scores]

    scores_appe = score_frame[:, 0, :]
    scores_flow = score_frame[:, 1, :]
    scores_angle = score_frame[:, 2, :]
    scores_mag = score_frame[:, 3, :]

    need_append = len(score_frame) < len(frame_labels)

    idx = selected_score_estimation_ways
    if idx is None:
        idx = np.arange(scores_appe.shape[1])
    scores_appe = scores_appe[:, idx]
    scores_flow = scores_flow[:, idx]
    scores_angle = scores_angle[:, idx]
    scores_mag = scores_mag[:, idx]

    if not isinstance(w_img, float):
        w_img = w_img[idx]
    if not isinstance(w_flow, float):
        w_flow = w_flow[idx]
    # w_img, w_flow = abs(w_img), abs(w_flow)
    print('shape:', w_img.shape, w_flow.shape, scores_appe.shape, scores_flow.shape)
    scores_comb = np.log(1./w_img**1*scores_appe) + 2*np.log(1./w_flow**1*scores_flow)
    # scores_comb *= -1
    if sequence_n_frame is not None:
        accumulated_n_frame = np.cumsum(sequence_n_frame-1)[:-1]

        scores_appe = np.split(scores_appe, accumulated_n_frame, axis=0)
        scores_flow = np.split(scores_flow, accumulated_n_frame, axis=0)
        scores_comb = np.split(scores_comb, accumulated_n_frame, axis=0)
        scores_angle = np.split(scores_angle, accumulated_n_frame, axis=0)
        scores_mag = np.split(scores_mag, accumulated_n_frame, axis=0)

        if clip_normalize:
            ver = 1
            np.seterr(divide='ignore', invalid='ignore')
            scores_appe = normalize_clip_scores(scores_appe, ver=ver)
            scores_flow = normalize_clip_scores(scores_flow, ver=ver)
            scores_comb = normalize_clip_scores(scores_comb, ver=ver)
            scores_angle = normalize_clip_scores(scores_angle, ver=ver)
            scores_mag = normalize_clip_scores(scores_mag, ver=ver)

        if need_append:
            scores_appe = [np.concatenate((item, [item[0]]), axis=0) for item in scores_appe]
            scores_flow = [np.concatenate((item, [item[0]]), axis=0) for item in scores_flow]
            scores_comb = [np.concatenate((item, [item[0]]), axis=0) for item in scores_comb]
            scores_angle = [np.concatenate((item, [item[0]]), axis=0) for item in scores_angle]
            scores_mag = [np.concatenate((item, [item[0]]), axis=0) for item in scores_mag]

        scores_appe = np.concatenate(scores_appe, axis=0)
        scores_flow = np.concatenate(scores_flow, axis=0)
        scores_comb = np.concatenate(scores_comb, axis=0)
        scores_angle = np.concatenate(scores_angle, axis=0)
        scores_mag = np.concatenate(scores_mag, axis=0)

    print(scores_appe.shape, scores_flow.shape, scores_comb.shape, scores_angle.shape, scores_mag.shape)
    print('              PSNR_X,PSNR_inv,PSNR,SSIM,MSE,maxSE,std,MSE_1c,maxSE_1c,std_1c')
    auc, prc = basic_assess_AUC(scores_appe, frame_labels) if len(np.unique(frame_labels)) > 1 else [-1, -1]
    if use_pr:
        print('appearance PRscore:', ', '.join(('%.3f' % val) for val in prc))
    else:
        print('appearance AUCs:', ', '.join(('%.3f' % val) for val in auc))

    auc, prc = basic_assess_AUC(scores_flow, frame_labels) if len(np.unique(frame_labels)) > 1 else [-1, -1]
    if use_pr:
        print('optic flow PRscore:', ', '.join(('%.3f' % val) for val in prc))
    else:
        print('optic flow AUCs:', ', '.join(('%.3f' % val) for val in auc))

    auc, prc = basic_assess_AUC(scores_comb, frame_labels) if len(np.unique(frame_labels)) > 1 else [-1, -1]
    if use_pr:
        print('combinatio PRscore:', ', '.join(('%.3f' % val) for val in prc))
    else:
        print('combinatio AUCs:', ', '.join(('%.3f' % val) for val in auc))

    auc, prc = basic_assess_AUC(scores_angle, frame_labels) if len(np.unique(frame_labels)) > 1 else [-1, -1]
    if use_pr:
        print('direction  PRscore:', ', '.join(('%.3f' % val) for val in prc))
    else:
        print('direction  AUCs:', ', '.join(('%.3f' % val) for val in auc))

    auc, prc = basic_assess_AUC(scores_mag, frame_labels) if len(np.unique(frame_labels)) > 1 else [-1, -1]
    if use_pr:
        print('magnitude  PRscore:', ', '.join(('%.3f' % val) for val in prc))
    else:
        print('magnitude  AUCs:', ', '.join(('%.3f' % val) for val in auc))

    if save_pr_appe_SSIM_epoch is not None:
        p, r, _ = precision_recall_curve(frame_labels, scores_appe[:, 3])
        pr = [p, r]
        print('mAP of appearance SSIM:', average_precision_score(frame_labels, scores_appe[:, 3]))
        out_file = './training_saver/%s/output_test/%d_epoch/eval_appe_SSIM.npz' % (dataset['name'], save_pr_appe_SSIM_epoch)
        np.savez_compressed(out_file, pr=pr)


def find_max_patch(diff_map_flow, diff_map_appe, size=16, step=4, plot=False):
    assert len(diff_map_flow.shape) == 2 and diff_map_flow.shape[0] % size == 0 and diff_map_flow.shape[1] % size == 0
    assert size % step == 0
    max_val_mean, std_1, pos_1, std_appe_1, mean_appe_1 = 0, 0, None, 0, 0
    max_val_std, mean_2, pos_2, std_appe_2, mean_appe_2 = 0, 0, None, 0, 0
    for i in range(0, diff_map_flow.shape[0]-size, step):
        for j in range(0, diff_map_flow.shape[1]-size, step):
            curr_std = np.std(diff_map_flow[i:i+size, j:j+size])
            curr_mean = np.mean(diff_map_flow[i:i+size, j:j+size])
            curr_std_appe = np.std(diff_map_appe[i:i+size, j:j+size])
            curr_mean_appe = np.mean(diff_map_appe[i:i+size, j:j+size])
            if curr_mean > max_val_mean:
                max_val_mean = curr_mean
                std_1 = curr_std
                pos_1 = [i, j]
                std_appe_1 = curr_std_appe
                mean_appe_1 = curr_mean_appe
            if curr_std > max_val_std:
                max_val_std = curr_std
                mean_2 = curr_mean
                pos_2 = [i, j]
                std_appe_2 = curr_std_appe
                mean_appe_2 = curr_mean_appe

    if plot:
        print(pos_1, max_val_mean, std_1, std_appe_1, mean_appe_1)
        print(pos_2, mean_2, max_val_std, std_appe_2, mean_appe_2)
        rect_mean = Rectangle((pos_1[1], pos_1[0]), size, size, linewidth=2, edgecolor='g', facecolor='none')
        rect_std = Rectangle((pos_2[1], pos_2[0]), size, size, linewidth=2, edgecolor='r', facecolor='none')
        fig1, ax1 = plt.subplots(1)
        ax1.imshow(diff_map_flow)
        ax1.add_patch(rect_mean)
        ax1.add_patch(rect_std)
        rect_mean = Rectangle((pos_1[1], pos_1[0]), size, size, linewidth=2, edgecolor='g', facecolor='none')
        rect_std = Rectangle((pos_2[1], pos_2[0]), size, size, linewidth=2, edgecolor='r', facecolor='none')
        fig2, ax2 = plt.subplots(1)
        ax2.imshow(diff_map_appe)
        ax2.add_patch(rect_mean)
        ax2.add_patch(rect_std)
        plt.show()
    return max_val_mean, std_1, std_appe_1, mean_appe_1, mean_2, max_val_std, std_appe_2, mean_appe_2


def calc_score_max_patch_one_clip(dataset, epoch, clip_idx, step=4, train=False, force_calc=False):
    saved_data_path = './training_saver/%s/output_%s/%d_epoch' % (dataset['name'], 'train' if train else 'test', epoch)
    saved_score_file = '%s/patch_score_epoch_%d_clip_%d_step_%d.npz' % (saved_data_path, epoch, clip_idx + 1, step)
    if not force_calc and os.path.isfile(saved_score_file):
        return np.load(saved_score_file)['data']
    # load true data and outputted data
    in_appe, in_flow = load_images_and_flow_1clip(dataset, clip_idx, train=train)
    saved_data_file = '%s/output_%d.npz' % (saved_data_path, clip_idx)
    out_loader = np.load(saved_data_file)
    out_appe, out_flow = out_loader['image'].astype(np.float32), out_loader['flow'].astype(np.float32)

    assert in_flow.shape == out_flow.shape
    # calc score and save to file
    diff_map_flow = (in_flow-out_flow)**2
    diff_map_appe = (in_appe-out_appe)**2

    score_seq = np.array([find_max_patch(np.sum(diff_map_flow[i], axis=-1), np.sum(diff_map_appe[i], axis=-1), step=step)
                          for i in range(len(in_flow))])
    np.savez_compressed(saved_score_file, data=score_seq)
    return score_seq


def full_assess_AUC_by_max_patch(dataset, epoch, frame_labels, step=4, sequence_n_frame=None,
                                 clip_normalize=True, force_calc_score=True, save_roc_pr_idx=8):
    if dataset['name'] == 'Entrance':
        frame_labels = np.load('../dataset/Amit-Subway/subway_labels/entrance_clip_labels.npz')['label']
        frame_labels = [labels[:-1] for labels in frame_labels]
        frame_labels = np.concatenate(frame_labels)

    def normalize_clip_scores(scores, ver=1):
        assert ver in [1, 2]
        if ver == 1:
            return [item/np.max(item, axis=0) for item in scores]
        else:
            return [(item-np.min(item, axis=0))/(np.max(item, axis=0)-np.min(item, axis=0)) for item in scores]

    def load_patch_scores(dataset, epoch, step, train=False, force_calc_score=False):
        saved_data_path = './training_saver/%s/output_%s/%d_epoch' % (dataset['name'], 'train' if train else 'test', epoch)
        saved_score_file = '%s_max_patch_scores_step_%d.npz' % (saved_data_path, step)
        if not force_calc_score and os.path.isfile(saved_score_file):
            loaded_scores = np.load(saved_score_file)['data']
        else:
            loaded_scores = np.concatenate([calc_score_max_patch_one_clip(dataset, epoch, clip_idx, step=step, train=train)
                                            for clip_idx in range(dataset['n_clip_train' if train else 'n_clip_test'])], axis=0)
            np.savez_compressed(saved_score_file, data=loaded_scores)
        return loaded_scores

    training_scores = load_patch_scores(dataset, epoch, step, train=True, force_calc_score=force_calc_score)
    test_scores = load_patch_scores(dataset, epoch, step, train=False, force_calc_score=force_calc_score)

    print('training_scores:', training_scores.shape)
    print('test_scores:', test_scores.shape)
    mean_training_scores = np.mean(training_scores, axis=0)
    new_test_scores = np.array([test_score / mean_training_scores for test_score in test_scores])
    comb_test_scores = np.array([5*np.log(new_test_scores[i, [0, 1, 4, 5]]) + np.log(new_test_scores[i, [3, 2, 7, 6]])
                                 for i in range(len(new_test_scores))])
    full_scores = np.concatenate([new_test_scores, comb_test_scores], axis=1)

    norm_full_scores = copy.copy(full_scores)
    if sequence_n_frame is not None:
        accumulated_n_frame = np.cumsum(sequence_n_frame-1)[:-1]
        for i in range(full_scores.shape[1]):
            scores = np.split(full_scores[:, i], accumulated_n_frame, axis=0)
            if clip_normalize:
                ver = 1
                np.seterr(divide='ignore', invalid='ignore')
                scores = normalize_clip_scores(scores, ver=ver)

            norm_full_scores[:, i] = np.concatenate(scores, axis=0)

    print('AUC:', ['%.3f' % roc_auc_score(frame_labels, full_scores[:, i]) for i in range(full_scores.shape[1])])
    print('aPR:', ['%.3f' % average_precision_score(frame_labels, full_scores[:, i]) for i in range(full_scores.shape[1])])
    print('AUC:', ['%.3f' % roc_auc_score(frame_labels, norm_full_scores[:, i]) for i in range(norm_full_scores.shape[1])])
    print('aPR:', ['%.3f' % average_precision_score(frame_labels, norm_full_scores[:, i]) for i in range(norm_full_scores.shape[1])])
    if save_roc_pr_idx is not None:
        print('Results of selected index:')
        # ROC
        fpr_1, tpr_1, _ = roc_curve(frame_labels, full_scores[:, save_roc_pr_idx])
        roc_raw = [fpr_1, tpr_1]
        fpr_2, tpr_2, _ = roc_curve(frame_labels, norm_full_scores[:, save_roc_pr_idx])
        roc_norm = [fpr_2, tpr_2]
        print('AUC:', roc_auc_score(frame_labels, full_scores[:, save_roc_pr_idx]),
              roc_auc_score(frame_labels, norm_full_scores[:, save_roc_pr_idx]))
        # PR
        p_1, r_1, _ = precision_recall_curve(frame_labels, full_scores[:, save_roc_pr_idx])
        pr_raw = [p_1, r_1]
        p_2, r_2, _ = precision_recall_curve(frame_labels, norm_full_scores[:, save_roc_pr_idx])
        pr_norm = [p_2, r_2]
        print('mAP:', average_precision_score(frame_labels, full_scores[:, save_roc_pr_idx]),
              average_precision_score(frame_labels, norm_full_scores[:, save_roc_pr_idx]))
        #
        out_file = './training_saver/%s/output_test/%d_epoch/eval_step_%d.npz' % (dataset['name'], epoch, step)
        np.savez_compressed(out_file, roc_raw=roc_raw, roc_norm=roc_norm, pr_raw=pr_raw, pr_norm=pr_norm)


def convert_flows_to_grids_one_clip(dataset, epoch, clip_idx, train=False, force_calc=False):
    saved_data_path = './training_saver/%s/output_%s/%d_epoch' % (dataset['name'], 'train' if train else 'test', epoch)
    if not os.path.exists(saved_data_path):
        pathlib.Path(saved_data_path).mkdir(parents=True, exist_ok=True)

    saved_grid_file = '%s/output_grid_%d.npz' % (saved_data_path, clip_idx)
    if os.path.isfile(saved_grid_file) and not force_calc:
        return np.load(saved_grid_file)['data']

    saved_data_file = '%s/output_%d.npz' % (saved_data_path, clip_idx)
    if not os.path.isfile(saved_data_file):
        print('File not found!')
        return

    out_loader = np.load(saved_data_file)
    out_flow = out_loader['flow'].astype(np.float32)

    grids = np.array([datum_to_grid(datum) for datum in out_flow])
    #
    idx = -1
    if idx >= 0:
        plt.subplot(121)
        plt.imshow(out_flow[idx][..., -1])
        plt.subplot(122)
        plt.imshow(datum_to_grid(out_flow[idx][..., -1]).reshape((8, 12)))
        plt.show()
    #
    np.savez_compressed(saved_grid_file, data=grids)
    return grids


def full_assess_AUC_by_grid(dataset, epoch, frame_labels, sequence_n_frame=None, clip_normalize=True, force_calc_score=True):
    def normalize_clip_scores(scores, ver=1):
        assert ver in [1, 2]
        if ver == 1:
            return [item/np.max(item, axis=0) for item in scores]
        else:
            return [(item-np.min(item, axis=0))/(np.max(item, axis=0)-np.min(item, axis=0)) for item in scores]

    saved_data_path = './training_saver/%s/output_test/%d_epoch' % (dataset['name'], epoch)
    saved_score_file = '%s_grid_scores.npz' % saved_data_path
    if not force_calc_score and os.path.isfile(saved_score_file):
        full_scores = np.load(saved_score_file)['data']
    else:
        print('loading training grids...')
        training_flow_grids = np.concatenate([convert_flows_to_grids_one_clip(dataset, epoch, clip_idx, train=True)
                                              for clip_idx in range(dataset['n_clip_train'])], axis=0)
        print('loading test grids...')
        test_flow_grids = np.concatenate([convert_flows_to_grids_one_clip(dataset, epoch, clip_idx, train=False)
                                          for clip_idx in range(dataset['n_clip_test'])], axis=0)
        print('training and testing scoreNet...')
        full_scores = train_and_test_scoreNet(dataset['name'], epoch, training_flow_grids, test_flow_grids)
        np.savez_compressed(saved_score_file, data=full_scores)

    norm_full_scores = copy.copy(full_scores)
    if sequence_n_frame is not None:
        accumulated_n_frame = np.cumsum(sequence_n_frame-1)[:-1]
        for i in range(len(full_scores)):
            scores = np.split(full_scores[i], accumulated_n_frame, axis=0)

            if clip_normalize:
                ver = 1
                np.seterr(divide='ignore', invalid='ignore')
                scores = normalize_clip_scores(scores, ver=ver)

            norm_full_scores[i] = np.concatenate(scores, axis=0)

    print('AUC:', ['%.3f' % roc_auc_score(frame_labels, scores) for scores in full_scores])
    print('aPR:', ['%.3f' % average_precision_score(frame_labels, scores) for scores in full_scores])
    print('AUC:', ['%.3f' % roc_auc_score(frame_labels, scores) for scores in norm_full_scores])
    print('aPR:', ['%.3f' % average_precision_score(frame_labels, scores) for scores in norm_full_scores])


def assess_Boat_Belleview_Train(dataset, epoch, plot_pr=True):
    def load_grid_ground_truth_Boat(path_test):
        filenames = sorted(glob.glob('%s/GT%s/*.png' % (path_test, str(1).zfill(3))))[:-1]
        print(len(filenames))
        data = np.array([imread(filename, 'L')/255. for filename in filenames])
        data = np.where(data > 0.5, 1, 0)
        return data

    def score_map_to_grid(score_map, grid_size):
        grid_map_val = np.zeros(grid_size)
        grid_map_cnt = np.zeros(grid_size)
        for i in range(score_map.shape[0]):
            for j in range(score_map.shape[1]):
                u, v = np.array([i, j])/score_map.shape * np.array(grid_size)
                u, v = min(int(np.round(u)), grid_size[0]-1), min(int(np.round(v)), grid_size[1]-1)
                grid_map_cnt[u, v] += 1.
                grid_map_val[u, v] += score_map[i, j]
        return grid_map_val/grid_map_cnt

    def calc_score_map(dataset, clip_idx, train=False, force_calc=True):
        saved_data_path = './training_saver/%s/output_%s/%d_epoch' % (dataset['name'], 'train' if train else 'test', epoch)
        saved_score_file = '%s/score_map_epoch_%d_clip_%d.npz' % (saved_data_path, epoch, clip_idx + 1)
        if not force_calc and os.path.isfile(saved_score_file):
            loader = np.load(saved_score_file)
            return loader['flow'], loader['image']
        # load true data and outputted data
        in_appe, in_flow = load_images_and_flow_1clip(dataset, clip_idx, train=train)
        saved_data_file = '%s/output_%d.npz' % (saved_data_path, clip_idx)
        out_loader = np.load(saved_data_file)
        out_appe, out_flow = out_loader['image'].astype(np.float32), out_loader['flow'].astype(np.float32)

        assert in_flow.shape == out_flow.shape
        # calc score and save to file
        diff_map_flow = np.sum((in_flow-out_flow)**2, axis=-1)
        diff_map_appe = np.sum((in_appe-out_appe)**2, axis=-1)
        # return
        np.savez_compressed(saved_score_file, flow=diff_map_flow, image=diff_map_appe)
        return diff_map_flow, diff_map_appe

    assert dataset['name'] in ['Boat', 'Belleview', 'Train']
    assert dataset['n_clip_test'] == 1
    # load groundtruth
    print('loading ground truth...')
    dataset['ground_truth'] = load_grid_ground_truth_Boat(dataset['path_test'])
    # load scores
    print('loading scores...')
    train_diff_map_flow, train_diff_map_appe = calc_score_map(dataset, 0, train=True)
    test_diff_map_flow, test_diff_map_appe = calc_score_map(dataset, 0, train=False)

    plt.imshow((test_diff_map_appe[300]-np.min(test_diff_map_appe[300]))/(np.max(test_diff_map_appe[300])-np.min(test_diff_map_appe[300])))
    plt.show()
    # convert score maps to grids
    print('converting maps to grids...')
    test_grid_map_flow = np.array([score_map_to_grid(single_test_diff_map_flow, dataset['ground_truth'].shape[1:])
                                   for single_test_diff_map_flow in test_diff_map_flow])
    test_grid_map_appe = np.array([score_map_to_grid(single_test_diff_map_appe, dataset['ground_truth'].shape[1:])
                                   for single_test_diff_map_appe in test_diff_map_appe])

    plt.imshow(score_map_to_grid(test_grid_map_appe[300], dataset['ground_truth'].shape[1:]))
    plt.show()
    # AUC estimation
    print('assessing...')
    print(test_grid_map_flow.shape, dataset['ground_truth'].shape)
    assert test_grid_map_flow.shape == test_grid_map_appe.shape
    assert test_grid_map_flow.shape == dataset['ground_truth'].shape

    #
    from array import array
    data_to_write = array('f', np.array([[v1, v2, v3]
                                        for v1, v2, v3 in zip(test_grid_map_appe.flatten(),
                                                              test_grid_map_flow.flatten(),
                                                              dataset['ground_truth'].flatten())]).flatten())
    writer = open('scores_%s_epoch_%d.bin' % (dataset['name'], epoch), 'wb')
    data_to_write.tofile(writer)
    writer.close()
    return

    labels, scores = dataset['ground_truth'].flatten(), test_grid_map_appe.flatten()
    # precision, recall, _ = precision_recall_curve(labels, scores, pos_label=1)
    score_appe_vals = np.unique(test_grid_map_appe.flatten())
    score_appe_vals = np.random.choice(score_appe_vals, 1500)
    score_appe_vals = np.insert(score_appe_vals, 0, np.min(score_appe_vals) - 0.001)
    score_appe_vals = np.append(score_appe_vals, np.max(score_appe_vals) + 0.001)
    print(len(score_appe_vals))
    pr_matrix_appe = np.array([calc_pre_rec_pixel_wise_with_neighbor(test_grid_map_appe, dataset['ground_truth'], thresh)
                               for thresh in score_appe_vals])
    precision, recall = pr_matrix_appe[:, 0], pr_matrix_appe[:, 1]
    idx = np.argsort(recall)
    precision, recall = precision[idx], recall[idx]
    average_precision = np.sum((recall[1:]-recall[:-1])*precision[1:])
    assert average_precision >= 0.0 and average_precision <= 1.0

    print(len(np.where(labels == 0)[0]), len(np.where(labels == 1)[0]), len(np.unique(precision)), len(np.unique(recall)))
    # print('average PR:', average_precision_score(labels, scores))
    print('average PR:', average_precision)
    print('AUC:', roc_auc_score(labels, scores))
    if plot_pr:
        step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})
        plt.step(recall, precision, color='b', alpha=0.2, where='post')
        plt.fill_between(recall, precision, alpha=0.2, color='b', **step_kwargs)
        plt.xlabel('recall')
        plt.ylabel('precision')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.show()


###########################################
def get_segments(seq):
    def find_ends(seq):
        tmp = np.insert(seq, 0, -10)
        diff = tmp[1:] - tmp[:-1]
        peaks = np.where(diff != 1)[0]
        #
        ret = np.empty((len(peaks), 2), dtype=int)
        for i in range(len(ret)):
            ret[i] = [peaks[i], (peaks[i+1]-1) if i < len(ret)-1 else (len(seq)-1)]
        return ret
    #
    ends = find_ends(seq)
    return np.array([[seq[curr_end[0]], seq[curr_end[1]]] for curr_end in ends]).reshape(-1) + 1  # +1 for 1-based index (same as UCSD data)


def load_ground_truth_Avenue(folder, n_clip):
    ret = []
    for i in range(n_clip):
        filename = '%s/%d_label.mat' % (folder, i+1)
        # print(filename)
        data = loadmat(filename)['volLabel']
        n_bin = np.array([np.sum(data[0, i]) for i in range(len(data[0]))])
        abnormal_frames = np.where(n_bin > 0)[0]
        ret.append(get_segments(abnormal_frames))
    return ret


def load_ground_truth_Boat(path_test, n_clip=1):
    ret = []
    for i in range(n_clip):
        filenames = sorted(glob.glob('%s/GT%s/*.png' % (path_test, str(i+1).zfill(3))))
        print(len(filenames))
        data = np.array([imread(filename, 'L')/255. for filename in filenames])
        n_bin = np.array([np.sum(data[i]) for i in range(len(data))])
        abnormal_frames = np.where(n_bin > 0)[0]
        ret.append(get_segments(abnormal_frames))
    return ret
