import sys
import os
import argparse
import datetime
import numpy as np

from utils import load_images_and_flows, load_images_and_flow_1clip, count_sequence_n_frame, calc_score_one_clip, calc_score_full_clips
from utils import load_ground_truth_Avenue, load_ground_truth_Boat, get_test_frame_labels, get_weights, full_assess_AUC, save_SSIM_scores_subway
from utils import convert_flows_to_grids_one_clip, full_assess_AUC_by_grid, calc_score_max_patch_one_clip, full_assess_AUC_by_max_patch
from GAN_tf import train_Unet_naive_with_batch_norm, test_Unet_naive_with_batch_norm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# dataset
UCSDped2 = {'name': 'UCSDped2',
            'path': '../dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped2',
            'n_clip_train': 16,
            'n_clip_test': 12,
            'ground_truth': [[61, 180], [95, 180], [1, 146], [31, 180], [1, 129], [1, 159],
                             [46, 180], [1, 180], [1, 120], [1, 150], [1, 180], [88, 180]],
            'ground_truth_mask': np.arange(12)+1}

Avenue = {'name': 'Avenue',
          'path': '../dataset/Avenue/Avenue',
          'test_mask_path': '../dataset/Avenue/ground_truth_demo/testing_label_mask',
          'n_clip_train': 16,
          'n_clip_test': 21,
          'ground_truth': None,
          'ground_truth_mask': np.arange(21)+1}

Belleview = {'name': 'Belleview',
             'path': '../dataset/Traffic-Belleview',
             'n_clip_train': 1,
             'n_clip_test': 1,
             'ground_truth': None,
             'ground_truth_mask': [1]}

Train = {'name': 'Train',
         'path': '../dataset/Traffic-Train',
         'n_clip_train': 1,
         'n_clip_test': 1,
         'ground_truth': None,
         'ground_truth_mask': [1]}

Exit = {'name': 'Exit',
        'path': '../dataset/Amit-Subway/Exit-gate',
        'n_clip_train': 38,
        'n_clip_test': 35,
        'ground_truth_clip': [0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 3, 3, 2, 0, 0,
                              3, 0, 0, 3, 0, 0, [2, 2, 2], 0, 3, 0, 2, 2, 0, 1,
                              0, 3, 1, 0, [1, 2, 2, 2]]}  # 0 = normal, 1 = LT, 2 = WD, 3 = misc

Entrance = {'name': 'Entrance',
            'path': '../dataset/Amit-Subway/Entrance-gate',
            'n_clip_train': 12,
            'n_clip_test': 76,
            'ground_truth_clip': [3, [2, 5], 1, 1, 1, 5, 1, 2, [2, 2, 2, 5], 0, 0, 0,
                                  0, 3, 0, 0, 1, 0, 0, 2, 0, 0, 0, 3, 0, 3, 0, 0, 1, 1,
                                  0, 0, 3, 0, [4, 4], [2, 3], [1, 2, 2, 2], 0, [1, 4], 0,
                                  4, [2, 2], 0, 0, 1, 0, 1, 3, [2, 5], [1, 4, 4, 2, 2, 2, 2],
                                  2, 3, [1, 4, 4, 2, 2, 2], 0, 0, 1, 0, [4, 2], 3, [4, 2], 0,
                                  [4, 2], 4, 0, 0, 0, 2, 0, 4, 2, 0, 0, 0, 0, 0, 0]}  # 0 = normal, 1 = LT, 2 = WD, 3 = etc, 4 = NP, 5 = IN

dataset_dict = {'UCSDped2': UCSDped2, 'Avenue': Avenue, 'Belleview': Belleview, 'Train': Train, 'Exit': Exit, 'Entrance': Entrance}

h, w, d = 128, 192, 3


def main(argv):
    parser = argparse.ArgumentParser(description='task number')
    parser.add_argument('-d', '--dataset', help='dataset', default='')
    parser.add_argument('-t', '--task', help='task to perform', default=-1)
    parser.add_argument('-c', '--clip', help='clip index (zero-based)', default=-1)
    parser.add_argument('-s', '--set', help='test set', default=1)
    parser.add_argument('-e', '--epoch', help='number of epoch', default=40)
    parser.add_argument('-m', '--model', help='start model idx', default=0)
    parser.add_argument('-p', '--step', help='step for max_patch assessment', default=4)
    args = vars(parser.parse_args())
    #
    dataset = dataset_dict[args['dataset']]
    dataset['path_train'] = '%s/Train' % dataset['path']
    dataset['path_test'] = '%s/Test' % dataset['path']
    #
    task = int(args['task'])
    clip_idx = int(args['clip'])
    test_set = bool(int(args['set']))
    n_epoch_destination = int(args['epoch'])
    model_idx_to_start = int(args['model'])
    step = int(args['step'])
    model_test = model_idx_to_start
    print('Selected task = %d' % task)
    print('started time: %s' % datetime.datetime.now())

    '''======================'''
    ''' Task 1: Prepare data '''
    '''======================'''
    if task == 1:
        load_images_and_flows(dataset, new_size=[h, w], train=not test_set)

    '''========================='''
    ''' Task 2: Train GAN model '''
    '''========================='''
    if task == 2:
        # load data
        image_data, flow_data = load_images_and_flows(dataset, new_size=[h, w], train=True)
        print(image_data.shape, flow_data.shape)
        assert image_data.shape[-1] == d and flow_data.shape[-1] == d
        # set batch size
        batch_size = 16
        if dataset in [Avenue, Belleview, Exit]:
            batch_size = 8
        # train model
        train_Unet_naive_with_batch_norm(image_data, flow_data, n_epoch_destination, dataset_name=dataset['name'],
                                         start_model_idx=model_idx_to_start, batch_size=batch_size)

    '''========================'''
    ''' Task 3: Test GAN model '''
    '''========================'''
    if task == 3:
        print('test set: ', test_set)
        # load data
        image_data, flow_data = load_images_and_flow_1clip(dataset, clip_idx, train=not test_set)
        print(image_data.shape, flow_data.shape)
        assert image_data.shape[-1] == d and flow_data.shape[-1] == d
        # set batch size
        batch_size = 16
        if dataset in [Avenue, Belleview, Exit]:
            batch_size = 8
        # test model on 1 clip
        sequence_n_frame = count_sequence_n_frame(dataset, test=test_set) - 1  # minus 1 because of the removal of first frame
        print(sequence_n_frame)
        test_Unet_naive_with_batch_norm(image_data, flow_data, h, w, dataset, sequence_n_frame, clip_idx,
                                        batch_size=batch_size, model_idx=model_test, using_test_data=test_set)

    '''======================'''
    ''' Task -4: Calc scores '''
    '''======================'''
    if task == -4:
        score_frame = calc_score_one_clip(dataset, model_test, clip_idx, train=not test_set, force_calc=True)
        print('score_frame shape:', score_frame.shape)

    '''======================='''
    ''' Task 4: Estimate AUCs '''
    '''======================='''
    if task == 4:
        score_frame = calc_score_full_clips(dataset, model_test, train=not test_set)
        #
        sequence_n_frame = count_sequence_n_frame(dataset, test=True)
        if dataset not in [Exit, Entrance]:
            if dataset == Avenue:
                dataset['ground_truth'] = load_ground_truth_Avenue(dataset['test_mask_path'], dataset['n_clip_test'])
            else:
                if dataset in [Belleview, Train]:
                    dataset['ground_truth'] = load_ground_truth_Boat(dataset['path_test'], n_clip=dataset['n_clip_test'])
            #
            labels_exclude_last, labels_exclude_first, labels_full = get_test_frame_labels(dataset['ground_truth'], sequence_n_frame)
            print(np.unique(labels_exclude_last), np.unique(labels_exclude_first), np.unique(labels_full))
            #
            training_errors = get_weights(dataset, model_test)
            w_img, w_flow = training_errors[0].flatten(), training_errors[1].flatten()
            print('weights: w_img =', str(w_img), '- vw_flow =', str(w_flow))
            #
            score_estimation_ways = None
            print('labels_exclude_last: normalized clips')
            full_assess_AUC(dataset, score_frame, labels_exclude_last, w_img, w_flow, sequence_n_frame, True,
                            dataset in [Belleview, Train], score_estimation_ways,
                            save_pr_appe_SSIM_epoch=model_test if dataset in [Belleview, Train] else None)
            return

            print('labels_exclude_first: non-normalized clips')
            full_assess_AUC(dataset, score_frame, labels_exclude_first, w_img, w_flow, None, False,
                            dataset in [Belleview, Train], score_estimation_ways)
            print('labels_exclude_first: normalized clips')
            full_assess_AUC(dataset, score_frame, labels_exclude_first, w_img, w_flow, sequence_n_frame, True,
                            dataset in [Belleview, Train], score_estimation_ways)

            print('labels_full: non-normalized clips')
            full_assess_AUC(dataset, score_frame, labels_full, w_img, w_flow, sequence_n_frame, False,
                            dataset in [Belleview, Train], score_estimation_ways)
            print('labels_full: normalized clips')
            full_assess_AUC(dataset, score_frame, labels_full, w_img, w_flow, sequence_n_frame, True,
                            dataset in [Belleview, Train], score_estimation_ways)

    '''=========================='''
    ''' Task -7: Calc flow grids '''
    '''=========================='''
    if task == -7:
        grids = convert_flows_to_grids_one_clip(dataset, model_test, clip_idx, train=not test_set, force_calc=True)
        print('grids shape:', grids.shape)

    '''==========================='''
    ''' Task 7: Assess flow grids '''
    '''==========================='''
    if task == 7:
        sequence_n_frame = count_sequence_n_frame(dataset, test=True)
        if dataset not in [Exit, Entrance]:
            if dataset == Avenue:
                dataset['ground_truth'] = load_ground_truth_Avenue(dataset['test_mask_path'], dataset['n_clip_test'])
            else:
                if dataset in [Belleview, Train]:
                    dataset['ground_truth'] = load_ground_truth_Boat(dataset['path_test'], n_clip=dataset['n_clip_test'])
            #
            labels_exclude_last, labels_exclude_first, labels_full = get_test_frame_labels(dataset['ground_truth'], sequence_n_frame)
            full_assess_AUC_by_grid(dataset, model_test, labels_exclude_last, sequence_n_frame=sequence_n_frame, clip_normalize=True)

    '''===================================='''
    ''' Task -8: Calc max patch flow score '''
    '''===================================='''
    if task == -8:
        patch_scores = calc_score_max_patch_one_clip(dataset, model_test, clip_idx, step=step, train=not test_set, force_calc=True)
        print('patch_scores shape:', patch_scores.shape)

    '''======================================'''
    ''' Task 8: Assess max patch flow scores '''
    '''======================================'''
    if task == 8:
        sequence_n_frame = count_sequence_n_frame(dataset, test=True)
        if dataset not in [Exit, Entrance]:
            if dataset == Avenue:
                dataset['ground_truth'] = load_ground_truth_Avenue(dataset['test_mask_path'], dataset['n_clip_test'])
            else:
                if dataset in [Belleview, Train]:
                    dataset['ground_truth'] = load_ground_truth_Boat(dataset['path_test'], n_clip=dataset['n_clip_test'])
            #
            labels_exclude_last, labels_exclude_first, labels_full = get_test_frame_labels(dataset['ground_truth'], sequence_n_frame)
            full_assess_AUC_by_max_patch(dataset, model_test, labels_exclude_last, step=step,
                                         sequence_n_frame=sequence_n_frame, clip_normalize=True)

    '''================================='''
    ''' Task 11: Save SSIM score Subway '''
    '''================================='''
    if task == 11:
        save_SSIM_scores_subway(dataset, model_test, train=not test_set)


if __name__ == '__main__':
    main(sys.argv)
