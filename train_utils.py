import os
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import random
import pdb



def filtering_mask_selection(args, pred, noisy_label, threshold):
    # case1: predictions should be confident (base)
    # case2: predictions should be confident and predictions should be equal to the noisy label (less but correct)
    # case3: predictions should be confident or predictions should be equal to the noisy label (more but less correct)
    mode = args.filtering_case

    if mode == 'case1':
        pred_confidence, pred_category = pred.softmax(dim=1).max(dim=1)

        confident_pred = pred_confidence > threshold

        return confident_pred        # B, H, W
    
    elif mode == 'case2':
        pred_confidence, pred_category = pred.softmax(dim=1).max(dim=1)

        confident_pred = pred_confidence > threshold
        same_pred = pred_category == noisy_label

        filtered_clean_mask = confident_pred & same_pred

        return filtered_clean_mask        # B, H, W
    
    elif mode == 'case3':
        pred_confidence, pred_category = pred.softmax(dim=1).max(dim=1)

        confident_pred = pred_confidence > threshold
        same_pred = pred_category == noisy_label

        filtered_clean_mask = confident_pred | same_pred

        return filtered_clean_mask        # B, H, W
    
    else:
        print('unknown filtering case')


def filtering_mask_selection_v2(args, pred, noisy_label, threshold):
    # v2: add class balance operations
    # ----------------------------------------------------------------------------- #
    # case1: predictions should be confident (base)
    # case2: predictions should be confident and predictions should be equal to the noisy label (less but correct)
    # case3: predictions should be confident or predictions should be equal to the noisy label (more but less correct)
    mode = args.filtering_case
    device = pred.device

    batch_size, _, img_H, img_W = pred.shape

    if mode == 'case1':
        pred_confidence, pred_category = pred.softmax(dim=1).max(dim=1)

        selected_mask = pred_confidence > threshold
    
    elif mode == 'case2':
        pred_confidence, pred_category = pred.softmax(dim=1).max(dim=1)

        confident_pred = pred_confidence > threshold
        same_pred = pred_category == noisy_label

        selected_mask = confident_pred & same_pred
    
    elif mode == 'case3':
        pred_confidence, pred_category = pred.softmax(dim=1).max(dim=1)

        confident_pred = pred_confidence > threshold
        same_pred = pred_category == noisy_label

        selected_mask = confident_pred | same_pred

    else:
        selected_mask = torch.ones([batch_size, img_H, img_W]).bool().to(device)
        print('unknown filtering case')
    
    if args.use_balance_filtering:
        # first determine the number of the background pixels selected
        selected_background = (pred_category == 0) * selected_mask     # B, H, W
        selected_ODOC = (pred_category > 0) * selected_mask

        num_background_selected = selected_background.sum(dim=(-1, -2))     # B
        num_ODOC_selected = selected_ODOC.sum(dim=(-1, -2))

        max_background_number = num_ODOC_selected * args.balance_times  # max number of background pixels
        max_background_number[max_background_number == 0] = 5000        # in case no ODOC selected
        exceed_ref = (num_background_selected > max_background_number)  # according to batch

        # if exceed, then reduce
        num_background_selected_after_sample = num_background_selected
        num_background_selected_after_sample[exceed_ref] = max_background_number[exceed_ref]

        # then determine which background pixels can be filtered out
        if args.mode_balance == 'random':
            # random select some pixels belonging to the background
            for ii in range(batch_size):
                idx = np.arange(img_H * img_W)
                np.random.shuffle(idx)
                idx = idx.reshape(img_H, img_W)
                idx = torch.Tensor(idx).to(device)
                idx[selected_ODOC[ii]] += 1e20       # filter out foreground pixels

                if (idx <= num_background_selected_after_sample[ii]).sum() < num_background_selected_after_sample[ii]:
                    num_background_selected_after_sample[ii] += (num_background_selected_after_sample[ii] - (idx <= num_background_selected_after_sample[ii]).sum())

                selected_background[ii] = selected_background[ii] * (idx <= num_background_selected_after_sample[ii])

        elif args.mode_balance == 'cluster':
            # only select pixels within a square box
            for ii in range(batch_size):
                length = int(torch.sqrt(num_background_selected_after_sample[ii].float()))

                # idx = np.arange((img_H - (length // 2) - 1) * (img_W - (length // 2) - 1))
                # np.randum.shuffle(idx)
                # idx = idx.reshape((img_H - (length // 2) - 1), (img_W - (length // 2) - 1))
                idx = np.arange(img_H * img_W)
                np.random.shuffle(idx)
                idx = idx.reshape(img_H, img_W)
                idx = torch.Tensor(idx).to(device)
                idx[selected_ODOC[ii]] += 1e20       # filter out foreground pixels

                # prevent exceeding the boundary
                idx[0: ((length // 2) + 1), :] += 1e20
                idx[:, 0: ((length // 2) + 1)] += 1e20
                idx[(img_H - ((length // 2) + 1)):, :] += 1e20
                idx[:, (img_W - ((length // 2) + 1)):] += 1e20

                # select the box center
                count = 0
                while count < (img_H * img_W):
                    count += 1
                    center_loc_x, center_loc_y = torch.where(idx == count)
                    if len(center_loc_x) > 0:
                        sampled_mask = torch.zeros([img_H, img_W]).to(device)
                        sampled_mask[(center_loc_x - (length // 2) + 1): (center_loc_x + (length // 2) - 1), (center_loc_y - (length // 2) + 1): (center_loc_y + (length // 2) - 1)] = 1
                        sampled_mask = sampled_mask.bool()
                        break
                
                selected_background[ii] = selected_background[ii] * sampled_mask

        elif args.mode_balance == 'random_cluster':
            # select pixels within several square boxes
            for ii in range(batch_size):
                length = int(torch.sqrt(num_background_selected_after_sample[ii].float() / args.num_balance_cluster))

                # idx = np.arange((img_H - (length // 2) - 1) * (img_W - (length // 2) - 1))
                # np.randum.shuffle(idx)
                # idx = idx.reshape((img_H - (length // 2) - 1), (img_W - (length // 2) - 1))
                idx = np.arange(img_H * img_W)
                np.random.shuffle(idx)
                idx = idx.reshape(img_H, img_W)
                idx = torch.Tensor(idx).to(device)
                idx[selected_ODOC[ii]] += 1e20       # filter out foreground pixels

                # prevent exceeding the boundary
                idx[0: ((length // 2) + 1), :] += 1e20
                idx[:, 0: ((length // 2) + 1)] += 1e20
                idx[(img_H - ((length // 2) + 1)):, :] += 1e20
                idx[:, (img_W - ((length // 2) + 1)):] += 1e20

                # select the box center
                count = 0
                count_cluster = 0
                sampled_mask_pool = []
                while count_cluster < args.num_balance_cluster:
                    count += 1

                    center_loc_x, center_loc_y = torch.where(idx == count)
                    if len(center_loc_x) > 0:

                        count_cluster += 1

                        sampled_mask = torch.zeros([img_H, img_W]).to(device)
                        sampled_mask[(center_loc_x - (length // 2) + 1): (center_loc_x + (length // 2) - 1), (center_loc_y - (length // 2) + 1): (center_loc_y + (length // 2) - 1)] = 1
                        sampled_mask = sampled_mask.bool()
                        sampled_mask_pool.append(sampled_mask)
                    
                    if count > img_H * img_W:
                        break
                
                sampled_mask = torch.zeros([img_H, img_W]).bool().to(device)
                for jj in range(args.num_balance_cluster):
                    sampled_mask = sampled_mask | sampled_mask_pool[jj]

                selected_background[ii] = selected_background[ii] * sampled_mask
            
        selected_mask = selected_ODOC | selected_background

    return selected_mask


def filtering_mask_selection_general(args, pred, noisy_label, threshold):
    # max class no more than X times min class
    # ----------------------------------------------------------------------------- #
    # case1: predictions should be confident (base)
    # case2: predictions should be confident and predictions should be equal to the noisy label (less but correct)
    # case3: predictions should be confident or predictions should be equal to the noisy label (more but less correct)
    mode = args.filtering_case
    device = pred.device

    batch_size, _, img_H, img_W = pred.shape

    pred_confidence, pred_category = pred.softmax(dim=1).max(dim=1)

    if args.use_confident_filtering:

        if mode == 'case1':

            selected_mask = pred_confidence > threshold
        
        elif mode == 'case2':

            confident_pred = pred_confidence > threshold
            same_pred = pred_category == noisy_label

            selected_mask = confident_pred & same_pred
        
        elif mode == 'case3':

            confident_pred = pred_confidence > threshold
            same_pred = pred_category == noisy_label

            selected_mask = confident_pred | same_pred

        else:
            selected_mask = torch.ones([batch_size, img_H, img_W]).bool().to(device)
            print('unknown filtering case')
    else:

        selected_mask = torch.ones([batch_size, img_H, img_W]).bool().to(device)
        # print('do not select confident predictions')
    
    if args.use_balance_filtering:
        # use pred or noisy label to perform balance sampling
        if args.mode_balance_filtering == 'pred':
            use_mode_balance_filtering = pred_category
        else:
            use_mode_balance_filtering = noisy_label
        # first determine the number of the background pixels selected
        # select_class_mask: B, num_class, H, W
        for ii in range(args.num_class):
            if ii == 0:
                select_cat_wise_mask = (use_mode_balance_filtering == ii).unsqueeze(1)
            else:
                select_cat_wise_mask = torch.cat((select_cat_wise_mask, (use_mode_balance_filtering == ii).unsqueeze(1)), dim=1)

        select_cat_wise_num = select_cat_wise_mask.sum(dim=(-1, -2))     # B, num_class

        # find the min class
        min_select_cat_wise_num = select_cat_wise_num.min(dim=-1)[0]

        # calculate the max number
        # instance-wise ==> [B]
        max_select_cat_wise_num = min_select_cat_wise_num * args.balance_times
        max_select_cat_wise_num[max_select_cat_wise_num == 0] = args.min_max_cat_number
        
        # # number of pixels after balancing, can be moved into comments
        # balance_cat_wise_num = select_cat_wise_num.clone()
        # for ii in range(batch_size):
        #     balance_cat_wise_num[ii][balance_cat_wise_num[ii] > max_select_cat_wise_num[ii]] = max_select_cat_wise_num[ii]

        # perform balanced sampling
        balanced_mask = torch.zeros(batch_size, img_H, img_W).long().to(device)

        for ii in range(batch_size):
            for jj in range(args.num_class):
                if select_cat_wise_num[ii, jj] > max_select_cat_wise_num[ii]:
                    # find the predictions equal to the corresponding category
                    flattened_pred = use_mode_balance_filtering[ii].view(-1)
                    flattened_pred_equal_class = torch.where(flattened_pred == jj)[0]

                    # randomly select several pixels
                    index = [i for i in range(len(flattened_pred_equal_class))]
                    random.shuffle(index)
                    flattened_pred_equal_class = flattened_pred_equal_class[index]
                    balanced_sample_idx = flattened_pred_equal_class[:max_select_cat_wise_num[ii]]
                    
                    # locate the selected pixels
                    balanced_sample_mask = torch.zeros(img_H * img_W).to(device)
                    balanced_sample_mask[balanced_sample_idx] = 1
                    balanced_sample_mask = balanced_sample_mask.reshape(img_H, img_W)

                    balanced_mask[ii][balanced_sample_mask == 1] = 1 

                else:
                    # find the predictions equal to the corresponding category
                    flattened_pred = use_mode_balance_filtering[ii].view(-1)
                    flattened_pred_equal_class = torch.where(flattened_pred == jj)[0]

                    # randomly select several pixels
                    balanced_sample_idx = flattened_pred_equal_class
                    
                    # locate the selected pixels
                    balanced_sample_mask = torch.zeros(img_H * img_W).to(device)
                    balanced_sample_mask[balanced_sample_idx] = 1
                    balanced_sample_mask = balanced_sample_mask.reshape(img_H, img_W)

                    balanced_mask[ii][balanced_sample_mask == 1] = 1    

        balanced_mask = balanced_mask.bool()

    else:
        balanced_mask = torch.ones(batch_size, img_H, img_W).bool().to(device)

    return_mask = selected_mask & balanced_mask

    return return_mask


def filtering_selection_general(args, noisy_label):
    # max class no more than X times min class
    # ----------------------------------------------------------------------------- #
    # case1: predictions should be confident (base)
    # case2: predictions should be confident and predictions should be equal to the noisy label (less but correct)
    # case3: predictions should be confident or predictions should be equal to the noisy label (more but less correct)
    device = noisy_label.device

    batch_size, img_H, img_W = noisy_label.shape

    if args.use_filtering:
        
        use_mode_balance_filtering = noisy_label

        # first determine the number of the background pixels selected
        # select_class_mask: B, num_class, H, W
        for ii in range(args.num_class):
            if ii == 0:
                select_cat_wise_mask = (use_mode_balance_filtering == ii).unsqueeze(1)
            else:
                select_cat_wise_mask = torch.cat((select_cat_wise_mask, (use_mode_balance_filtering == ii).unsqueeze(1)), dim=1)

        select_cat_wise_num = select_cat_wise_mask.sum(dim=(-1, -2))     # B, num_class

        # find the min class
        min_select_cat_wise_num = select_cat_wise_num.min(dim=-1)[0]

        # calculate the max number
        # instance-wise ==> [B]
        max_select_cat_wise_num = min_select_cat_wise_num * args.balance_times
        max_select_cat_wise_num[max_select_cat_wise_num == 0] = args.min_max_cat_number
        
        # # number of pixels after balancing, can be moved into comments
        # balance_cat_wise_num = select_cat_wise_num.clone()
        # for ii in range(batch_size):
        #     balance_cat_wise_num[ii][balance_cat_wise_num[ii] > max_select_cat_wise_num[ii]] = max_select_cat_wise_num[ii]

        # perform balanced sampling
        balanced_mask = torch.zeros(batch_size, img_H, img_W).long().to(device)

        for ii in range(batch_size):
            for jj in range(args.num_class):
                if select_cat_wise_num[ii, jj] > max_select_cat_wise_num[ii]:
                    # find the predictions equal to the corresponding category
                    flattened_pred = use_mode_balance_filtering[ii].view(-1)
                    flattened_pred_equal_class = torch.where(flattened_pred == jj)[0]

                    # randomly select several pixels
                    index = [i for i in range(len(flattened_pred_equal_class))]
                    random.shuffle(index)
                    flattened_pred_equal_class = flattened_pred_equal_class[index]
                    balanced_sample_idx = flattened_pred_equal_class[:max_select_cat_wise_num[ii]]
                    
                    # locate the selected pixels
                    balanced_sample_mask = torch.zeros(img_H * img_W).to(device)
                    balanced_sample_mask[balanced_sample_idx] = 1
                    balanced_sample_mask = balanced_sample_mask.reshape(img_H, img_W)

                    balanced_mask[ii][balanced_sample_mask == 1] = 1 

                else:
                    # find the predictions equal to the corresponding category
                    flattened_pred = use_mode_balance_filtering[ii].view(-1)
                    flattened_pred_equal_class = torch.where(flattened_pred == jj)[0]

                    # randomly select several pixels
                    balanced_sample_idx = flattened_pred_equal_class
                    
                    # locate the selected pixels
                    balanced_sample_mask = torch.zeros(img_H * img_W).to(device)
                    balanced_sample_mask[balanced_sample_idx] = 1
                    balanced_sample_mask = balanced_sample_mask.reshape(img_H, img_W)

                    balanced_mask[ii][balanced_sample_mask == 1] = 1    

        balanced_mask = balanced_mask.bool()

    else:
        balanced_mask = torch.ones(batch_size, img_H, img_W).bool().to(device)

    return balanced_mask


def filtering_mask_selection_v4(args, pred, noisy_label, threshold):
    # v2: add class balance operations
    # v4: return both balanced mask and all mask
    # ----------------------------------------------------------------------------- #
    # case1: predictions should be confident (base)
    # case2: predictions should be confident and predictions should be equal to the noisy label (less but correct)
    # case3: predictions should be confident or predictions should be equal to the noisy label (more but less correct)
    mode = args.filtering_case
    device = pred.device

    batch_size, _, img_H, img_W = pred.shape

    if mode == 'case1':
        pred_confidence, pred_category = pred.softmax(dim=1).max(dim=1)

        selected_mask = pred_confidence > threshold
    
    elif mode == 'case2':
        pred_confidence, pred_category = pred.softmax(dim=1).max(dim=1)

        confident_pred = pred_confidence > threshold
        same_pred = pred_category == noisy_label

        selected_mask = confident_pred & same_pred
    
    elif mode == 'case3':
        pred_confidence, pred_category = pred.softmax(dim=1).max(dim=1)

        confident_pred = pred_confidence > threshold
        same_pred = pred_category == noisy_label

        selected_mask = confident_pred | same_pred

    else:
        selected_mask = torch.ones([batch_size, img_H, img_W]).bool().to(device)
        print('unknown filtering case')
    
    all_selected_mask = selected_mask.clone()

    if args.use_balance_filtering:
        # first determine the number of the background pixels selected
        selected_background = (pred_category == 0) * selected_mask     # B, H, W
        selected_ODOC = (pred_category > 0) * selected_mask

        num_background_selected = selected_background.sum(dim=(-1, -2))     # B
        num_ODOC_selected = selected_ODOC.sum(dim=(-1, -2))

        max_background_number = num_ODOC_selected * args.balance_times                     # max number of background pixels
        max_background_number[max_background_number == 0] = args.min_max_cat_number        # in case no ODOC selected
        exceed_ref = (num_background_selected > max_background_number)                     # according to batch

        # if exceed, then reduce
        num_background_selected_after_sample = num_background_selected
        num_background_selected_after_sample[exceed_ref] = max_background_number[exceed_ref]

        # then determine which background pixels can be filtered out
        if args.mode_balance == 'random':
            # random select some pixels belonging to the background
            for ii in range(batch_size):
                idx = np.arange(img_H * img_W)
                np.random.shuffle(idx)
                idx = idx.reshape(img_H, img_W)
                idx = torch.Tensor(idx).to(device)
                idx[selected_ODOC[ii]] += 1e20       # filter out foreground pixels

                if (idx <= num_background_selected_after_sample[ii]).sum() < num_background_selected_after_sample[ii]:
                    num_background_selected_after_sample[ii] += (num_background_selected_after_sample[ii] - (idx <= num_background_selected_after_sample[ii]).sum())

                selected_background[ii] = selected_background[ii] * (idx <= num_background_selected_after_sample[ii])

        elif args.mode_balance == 'cluster':
            # only select pixels within a square box
            for ii in range(batch_size):
                length = int(torch.sqrt(num_background_selected_after_sample[ii].float()))

                # idx = np.arange((img_H - (length // 2) - 1) * (img_W - (length // 2) - 1))
                # np.randum.shuffle(idx)
                # idx = idx.reshape((img_H - (length // 2) - 1), (img_W - (length // 2) - 1))
                idx = np.arange(img_H * img_W)
                np.random.shuffle(idx)
                idx = idx.reshape(img_H, img_W)
                idx = torch.Tensor(idx).to(device)
                idx[selected_ODOC[ii]] += 1e20       # filter out foreground pixels

                # prevent exceeding the boundary
                idx[0: ((length // 2) + 1), :] += 1e20
                idx[:, 0: ((length // 2) + 1)] += 1e20
                idx[(img_H - ((length // 2) + 1)):, :] += 1e20
                idx[:, (img_W - ((length // 2) + 1)):] += 1e20

                # select the box center
                count = 0
                while count < (img_H * img_W):
                    count += 1
                    center_loc_x, center_loc_y = torch.where(idx == count)
                    if len(center_loc_x) > 0:
                        sampled_mask = torch.zeros([img_H, img_W]).to(device)
                        sampled_mask[(center_loc_x - (length // 2) + 1): (center_loc_x + (length // 2) - 1), (center_loc_y - (length // 2) + 1): (center_loc_y + (length // 2) - 1)] = 1
                        sampled_mask = sampled_mask.bool()
                        break
                
                selected_background[ii] = selected_background[ii] * sampled_mask

        elif args.mode_balance == 'random_cluster':
            # select pixels within several square boxes
            for ii in range(batch_size):
                length = int(torch.sqrt(num_background_selected_after_sample[ii].float() / args.num_balance_cluster))

                # idx = np.arange((img_H - (length // 2) - 1) * (img_W - (length // 2) - 1))
                # np.randum.shuffle(idx)
                # idx = idx.reshape((img_H - (length // 2) - 1), (img_W - (length // 2) - 1))
                idx = np.arange(img_H * img_W)
                np.random.shuffle(idx)
                idx = idx.reshape(img_H, img_W)
                idx = torch.Tensor(idx).to(device)
                idx[selected_ODOC[ii]] += 1e20       # filter out foreground pixels

                # prevent exceeding the boundary
                idx[0: ((length // 2) + 1), :] += 1e20
                idx[:, 0: ((length // 2) + 1)] += 1e20
                idx[(img_H - ((length // 2) + 1)):, :] += 1e20
                idx[:, (img_W - ((length // 2) + 1)):] += 1e20

                # select the box center
                count = 0
                count_cluster = 0
                sampled_mask_pool = []
                while count_cluster < args.num_balance_cluster:
                    count += 1

                    center_loc_x, center_loc_y = torch.where(idx == count)
                    if len(center_loc_x) > 0:

                        count_cluster += 1

                        sampled_mask = torch.zeros([img_H, img_W]).to(device)
                        sampled_mask[(center_loc_x - (length // 2) + 1): (center_loc_x + (length // 2) - 1), (center_loc_y - (length // 2) + 1): (center_loc_y + (length // 2) - 1)] = 1
                        sampled_mask = sampled_mask.bool()
                        sampled_mask_pool.append(sampled_mask)
                    
                    if count > img_H * img_W:
                        break
                
                sampled_mask = torch.zeros([img_H, img_W]).bool().to(device)
                for jj in range(args.num_balance_cluster):
                    sampled_mask = sampled_mask | sampled_mask_pool[jj]

                selected_background[ii] = selected_background[ii] * sampled_mask
            
        selected_mask = selected_ODOC | selected_background

    return selected_mask, all_selected_mask



def filtering_noisy_mask_selection(args, pred, noisy_label, threshold):
    # only select noisy mask
    # ----------------------------------------------------------------------------- #

    pred_confidence, pred_category = pred.softmax(dim=1).max(dim=1)

    confident_pred = pred_confidence > threshold
    same_pred = pred_category != noisy_label

    selected_mask = confident_pred & same_pred
    
    return selected_mask


def weighting_noisy_mask(args, pred, noisy_label):
    # case1: pred == noisy_mask --> higher p, lower weight: higher p contains lower information
    # case2: pred != noisy_mask --> higher p, lower weight: higher p, mask may be wrong, lower weight
    B, D, H, W = pred.shape

    pred_confidence, pred_category = pred.softmax(dim=1).max(dim=1)
    same_pred = pred_category == noisy_label

    weight_map = torch.ones([B, H, W]).to(args.device)

    if args.down_weight_same_pred:
        weight_map[same_pred] = (args.standard_weight_same_pred - pred_confidence)[same_pred]
    
    if args.down_weight_different_pred:
        weight_map[~same_pred] = (args.standard_weight_different_pred - pred_confidence)[~same_pred]

    return weight_map


def filtering_selection(args, noisy_label):
    # according to the given noisy label, only tackle the class imbalance problem
    # ----------------------------------------------------------------------------- #
    mode = args.filtering_case
    device = noisy_label.device

    batch_size, img_H, img_W = noisy_label.shape

    selected_mask = torch.ones([batch_size, img_H, img_W]).bool().to(device)
    
    if args.use_balance_filtering:
        # first determine the number of the background pixels selected
        selected_background = (noisy_label == 0) * selected_mask     # B, H, W
        selected_ODOC = (noisy_label > 0) * selected_mask

        num_background_selected = selected_background.sum(dim=(-1, -2))     # B
        num_ODOC_selected = selected_ODOC.sum(dim=(-1, -2))

        max_background_number = num_ODOC_selected * args.balance_times  # max number of background pixels
        max_background_number[max_background_number == 0] = 5000        # in case no ODOC selected
        exceed_ref = (num_background_selected > max_background_number)  # according to batch

        # if exceed, then reduce
        num_background_selected_after_sample = num_background_selected
        num_background_selected_after_sample[exceed_ref] = max_background_number[exceed_ref]

        # then determine which background pixels can be filtered out
        if args.mode_balance == 'random':
            # random select some pixels belonging to the background
            for ii in range(batch_size):
                idx = np.arange(img_H * img_W)
                np.random.shuffle(idx)
                idx = idx.reshape(img_H, img_W)
                idx = torch.Tensor(idx).to(device)
                idx[selected_ODOC[ii]] += 1e20       # filter out foreground pixels

                if (idx <= num_background_selected_after_sample[ii]).sum() < num_background_selected_after_sample[ii]:
                    num_background_selected_after_sample[ii] += (num_background_selected_after_sample[ii] - (idx <= num_background_selected_after_sample[ii]).sum())

                selected_background[ii] = selected_background[ii] * (idx <= num_background_selected_after_sample[ii])

        elif args.mode_balance == 'cluster':
            # only select pixels within a square box
            for ii in range(batch_size):
                length = int(torch.sqrt(num_background_selected_after_sample[ii].float()))

                # idx = np.arange((img_H - (length // 2) - 1) * (img_W - (length // 2) - 1))
                # np.randum.shuffle(idx)
                # idx = idx.reshape((img_H - (length // 2) - 1), (img_W - (length // 2) - 1))
                idx = np.arange(img_H * img_W)
                np.random.shuffle(idx)
                idx = idx.reshape(img_H, img_W)
                idx = torch.Tensor(idx).to(device)
                idx[selected_ODOC[ii]] += 1e20       # filter out foreground pixels

                # prevent exceeding the boundary
                idx[0: ((length // 2) + 1), :] += 1e20
                idx[:, 0: ((length // 2) + 1)] += 1e20
                idx[(img_H - ((length // 2) + 1)):, :] += 1e20
                idx[:, (img_W - ((length // 2) + 1)):] += 1e20

                # select the box center
                count = 0
                while count < (img_H * img_W):
                    count += 1
                    center_loc_x, center_loc_y = torch.where(idx == count)
                    if len(center_loc_x) > 0:
                        sampled_mask = torch.zeros([img_H, img_W]).to(device)
                        sampled_mask[(center_loc_x - (length // 2) + 1): (center_loc_x + (length // 2) - 1), (center_loc_y - (length // 2) + 1): (center_loc_y + (length // 2) - 1)] = 1
                        sampled_mask = sampled_mask.bool()
                        break
                
                selected_background[ii] = selected_background[ii] * sampled_mask

        elif args.mode_balance == 'random_cluster':
            # select pixels within several square boxes
            for ii in range(batch_size):
                length = int(torch.sqrt(num_background_selected_after_sample[ii].float() / args.num_balance_cluster))

                # idx = np.arange((img_H - (length // 2) - 1) * (img_W - (length // 2) - 1))
                # np.randum.shuffle(idx)
                # idx = idx.reshape((img_H - (length // 2) - 1), (img_W - (length // 2) - 1))
                idx = np.arange(img_H * img_W)
                np.random.shuffle(idx)
                idx = idx.reshape(img_H, img_W)
                idx = torch.Tensor(idx).to(device)
                idx[selected_ODOC[ii]] += 1e20       # filter out foreground pixels

                # prevent exceeding the boundary
                idx[0: ((length // 2) + 1), :] += 1e20
                idx[:, 0: ((length // 2) + 1)] += 1e20
                idx[(img_H - ((length // 2) + 1)):, :] += 1e20
                idx[:, (img_W - ((length // 2) + 1)):] += 1e20

                # select the box center
                count = 0
                count_cluster = 0
                sampled_mask_pool = []
                while count_cluster < args.num_balance_cluster:
                    count += 1

                    center_loc_x, center_loc_y = torch.where(idx == count)
                    if len(center_loc_x) > 0:

                        count_cluster += 1

                        sampled_mask = torch.zeros([img_H, img_W]).to(device)
                        sampled_mask[(center_loc_x - (length // 2) + 1): (center_loc_x + (length // 2) - 1), (center_loc_y - (length // 2) + 1): (center_loc_y + (length // 2) - 1)] = 1
                        sampled_mask = sampled_mask.bool()
                        sampled_mask_pool.append(sampled_mask)
                    
                    if count > img_H * img_W:
                        break
                
                sampled_mask = torch.zeros([img_H, img_W]).bool().to(device)
                for jj in range(args.num_balance_cluster):
                    sampled_mask = sampled_mask | sampled_mask_pool[jj]

                selected_background[ii] = selected_background[ii] * sampled_mask
            
        selected_mask = selected_ODOC | selected_background

    return selected_mask

