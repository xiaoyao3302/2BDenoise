import os
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import pdb


def cal_IoU(region_A, region_B):
    # region_A, region_B: binary
    assert region_A.shape == region_B.shape
    inter_region = region_A * region_B              # TP
    IoU = inter_region.sum() / (region_A.sum() + region_B.sum() - inter_region.sum() + 1e-6)

    return IoU


def cal_Dice(region_A, region_B):
    # region_A, region_B: binary
    assert region_A.shape == region_B.shape
    inter_region = region_A * region_B              # TP
    Dice = 2 * inter_region.sum() / (region_A.sum() + region_B.sum() + 1e-6)

    return Dice


def create_breats_colormap():
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[1] = [255, 255, 255]

    return colormap


def create_breats_colormap_confidence():
    colormap = np.zeros((256, 3), dtype=np.uint8)
    colormap[1] = [128, 0, 0]
    colormap[2] = [0, 128, 0]
    colormap[3] = [128, 128, 0]
    colormap[4] = [0, 0, 128]
    colormap[5] = [128, 0, 128]
    colormap[6] = [0, 128, 128]
    colormap[7] = [128, 128, 128]
    colormap[8] = [64, 0, 0]
    colormap[9] = [192, 0, 0]
    colormap[10] = [64, 128, 0]         # least
    colormap[11] = [192, 128, 0]
    colormap[12] = [64, 0, 128]
    colormap[13] = [192, 0, 128]
    colormap[14] = [64, 128, 128]
    colormap[15] = [192, 128, 128]
    colormap[16] = [0, 64, 0]
    colormap[17] = [128, 64, 0]
    colormap[18] = [0, 192, 0]
    colormap[19] = [128, 192, 0]

    return colormap




def colorize(mask, colormap):
    color_mask = np.zeros([mask.shape[0], mask.shape[1], 3])
    for i in np.unique(mask):
        color_mask[mask == i] = colormap[i]

    return Image.fromarray(np.uint8(color_mask))


def save_filter_prediction_vis(filter_mask, gt_mask, data_name, save_path, epoch):
    device = filter_mask.device
    colormap = create_breats_colormap()
    plot_filter_value = torch.ones_like(filter_mask).to(device)
    plot_filter_value = plot_filter_value * filter_mask

    for ii in range(filter_mask.shape[0]):
        gray = np.uint8(plot_filter_value[ii].cpu().numpy())
        color = colorize(gray, colormap)
        image_name = data_name[ii].split('.')[0]          # eg: 414.jpg --> 414
        color_folder = save_path + '/epoch_' + str(epoch) + '/filter_mask/'
        os.makedirs(color_folder, exist_ok=True)
        color_path = os.path.join(color_folder, image_name + "_filter_mask.png")
        gray = Image.fromarray(gray)
        color.save(color_path)

        # save gt vis
        gray = np.uint8(gt_mask[ii].cpu().numpy())
        color = colorize(gray, colormap)
        image_name = data_name[ii].split('.')[0]          # eg: 414.jpg --> 414
        color_folder = save_path + '/epoch_' + str(epoch) + '/filter_mask/'
        os.makedirs(color_folder, exist_ok=True)
        color_path = os.path.join(color_folder, image_name + "_gt.png")
        gray = Image.fromarray(gray)
        color.save(color_path)


def save_filter_prediction_vis_v2(filter_mask, rest_mask, gt_mask, data_name, save_path, epoch):
    # vis clean mask and noisy mask
    device = filter_mask.device
    colormap = create_breats_colormap()
    plot_clean_mask_value = torch.ones_like(filter_mask).to(device)
    plot_clean_mask_value = plot_clean_mask_value * filter_mask

    plot_noisy_mask_value = torch.ones_like(filter_mask).to(device)
    plot_noisy_mask_value = plot_noisy_mask_value * rest_mask

    for ii in range(filter_mask.shape[0]):
        # save clean mask
        gray = np.uint8(plot_clean_mask_value[ii].cpu().numpy())
        color = colorize(gray, colormap)
        image_name = data_name[ii].split('.')[0]          # eg: 414.jpg --> 414
        color_folder = save_path + '/epoch_' + str(epoch) + '/filter_mask/'
        os.makedirs(color_folder, exist_ok=True)
        color_path = os.path.join(color_folder, image_name + "_clean_mask.png")
        gray = Image.fromarray(gray)
        color.save(color_path)

        # save noisy mask
        gray = np.uint8(plot_noisy_mask_value[ii].cpu().numpy())
        color = colorize(gray, colormap)
        image_name = data_name[ii].split('.')[0]          # eg: 414.jpg --> 414
        color_folder = save_path + '/epoch_' + str(epoch) + '/filter_mask/'
        os.makedirs(color_folder, exist_ok=True)
        color_path = os.path.join(color_folder, image_name + "_noisy_mask.png")
        gray = Image.fromarray(gray)
        color.save(color_path)

        # save gt vis
        gray = np.uint8(gt_mask[ii].cpu().numpy())
        color = colorize(gray, colormap)
        image_name = data_name[ii].split('.')[0]          # eg: 414.jpg --> 414
        color_folder = save_path + '/epoch_' + str(epoch) + '/filter_mask/'
        os.makedirs(color_folder, exist_ok=True)
        color_path = os.path.join(color_folder, image_name + "_gt.png")
        gray = Image.fromarray(gray)
        color.save(color_path)


def evaluate_save_breast(mode, io, device, save_path, model, loader, idx_epoch=0, need_save=True, normalized=False):
    count = 0
    print_losses = {'total': 0.0, 'seg': 0.0}

    print_results = {'mean_IoU': 0.0, 'mean_Dice': 0.0}
    
    batch_idx = 0
    criterion_seg = torch.nn.CrossEntropyLoss(reduction='mean').to(device)

    colormap = create_breats_colormap()

    with torch.no_grad():
        model.eval()

        IoU = 0.0
        Dice = 0.0

        for data_all in loader:
            data, labels, data_name = data_all[0].to(device), data_all[1].long().to(device).squeeze(), data_all[2]

            batch_size, _, img_H, img_W = data.shape

            if len(labels.shape) < 3:
                labels = labels.unsqueeze(0)

            logits = model(data)

            loss = criterion_seg(logits["pred"], labels)

            print_losses['seg'] += loss.item() * batch_size
            print_losses['total'] += loss.item() * batch_size

            # evaluation metrics
            preds = logits["pred"].max(dim=1)[1]

            for ii in range(batch_size):
                
                # ------------------------------------------------------------------ #
                # calculate IoU and Dice

                IoU += cal_IoU(labels, preds)
                Dice += cal_Dice(labels, preds)

                # ------------------------------------------------------------------ #
                if need_save:
                    
                    # save pred
                    gray = np.uint8(preds[ii].cpu().numpy())
                    color = colorize(gray, colormap)
                    image_name = data_name[ii].split('.')[0]          # eg: 414.jpg --> 414
                    color_folder = save_path + '/epoch_' + str(idx_epoch) + '/color/'
                    os.makedirs(color_folder, exist_ok=True)
                    color_path = os.path.join(color_folder, image_name + "_pred.png")
                    gray = Image.fromarray(gray)
                    color.save(color_path)

                    # save gt vis
                    gray = np.uint8(labels[ii].cpu().numpy())
                    color = colorize(gray, colormap)
                    image_name = data_name[ii].split('.')[0]          # eg: 414.jpg --> 414
                    color_folder = save_path + '/epoch_' + str(idx_epoch) + '/color/'
                    os.makedirs(color_folder, exist_ok=True)
                    color_path = os.path.join(color_folder, image_name + "_gt.png")
                    gray = Image.fromarray(gray)
                    color.save(color_path)

                    # save raw vis
                    toPIL = transforms.ToPILImage()
                    if normalized:
                        std = torch.Tensor([0.229, 0.224, 0.225]).to(device)
                        mean = torch.Tensor([0.485, 0.456, 0.406]).to(device)
                        un_norm = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
                        save_data = un_norm(data[ii])
                        pic = toPIL(save_data)
                        raw_path = os.path.join(color_folder, image_name + "_raw.jpg")
                        pic.save(raw_path)
                    else:
                        pic = toPIL(data[ii])
                        raw_path = os.path.join(color_folder, image_name + "_raw.jpg")
                        pic.save(raw_path)

            count += batch_size
            batch_idx += 1

    print_losses = {k: v * 1.0 / count for (k, v) in print_losses.items()}
    
    IoU = IoU / count
    Dice = Dice / count

    print_results['mean_IoU'] += IoU
    print_results['mean_Dice'] += Dice

    io.print_loss(mode, idx_epoch, print_losses)
    io.cprint("-----------------------------------------------------------------")
    io.print_result(mode, idx_epoch, print_results)

    return print_losses, print_results

def save_filter_prediction_vis_all(filter_mask, gt_mask, noisy_mask, raw_img, data_name, save_path, epoch, normalized=True):
    # save raw, gt, noisy, mask
    device = filter_mask.device
    colormap = create_breats_colormap()
    plot_filter_value = torch.ones_like(filter_mask).to(device)
    plot_filter_value = plot_filter_value * filter_mask

    for ii in range(filter_mask.shape[0]):
        # save pred mask
        gray = np.uint8(plot_filter_value[ii].cpu().numpy())
        color = colorize(gray, colormap)
        image_name = data_name[ii].split('.')[0]          # eg: 414.jpg --> 414
        color_folder = save_path + '/epoch_' + str(epoch) + '/filter_mask/'
        os.makedirs(color_folder, exist_ok=True)
        color_path = os.path.join(color_folder, image_name + "_filter_mask.png")
        gray = Image.fromarray(gray)
        color.save(color_path)

        # save noisy vis
        gray = np.uint8(noisy_mask[ii].cpu().numpy())
        color = colorize(gray, colormap)
        image_name = data_name[ii].split('.')[0]          # eg: 414.jpg --> 414
        color_folder = save_path + '/epoch_' + str(epoch) + '/filter_mask/'
        os.makedirs(color_folder, exist_ok=True)
        color_path = os.path.join(color_folder, image_name + "_noisy.png")
        gray = Image.fromarray(gray)
        color.save(color_path)

        # save gt vis
        gray = np.uint8(gt_mask[ii].cpu().numpy())
        color = colorize(gray, colormap)
        image_name = data_name[ii].split('.')[0]          # eg: 414.jpg --> 414
        color_folder = save_path + '/epoch_' + str(epoch) + '/filter_mask/'
        os.makedirs(color_folder, exist_ok=True)
        color_path = os.path.join(color_folder, image_name + "_gt.png")
        gray = Image.fromarray(gray)
        color.save(color_path)

        # save raw vis
        toPIL = transforms.ToPILImage()
        pic = toPIL(raw_img[ii])
        raw_path = os.path.join(color_folder, image_name + "_raw.jpg")
        pic.save(raw_path)


def evaluate_save_breast_VolMin(mode, io, device, save_path, model, trans, loader, idx_epoch=0, need_save=True, normalized=True):
    # reproduce VolMin
    count = 0
    print_losses = {'total': 0.0, 'seg': 0.0}

    print_results = {'mean_IoU': 0.0, 'mean_Dice': 0.0}
    
    batch_idx = 0
    criterion_seg = torch.nn.CrossEntropyLoss(reduction='mean').to(device)

    colormap = create_breats_colormap()

    with torch.no_grad():
        model.eval()

        IoU = 0.0
        Dice = 0.0

        for data_all in loader:
            data, labels, data_name = data_all[0].to(device), data_all[1].long().to(device).squeeze(), data_all[2]

            batch_size, _, img_H, img_W = data.shape

            if len(labels.shape) < 3:
                labels = labels.unsqueeze(0)

            logits = model(data)

            pred = logits["pred"]
            transition_matrix = trans()     # C * C

            final_pred = torch.matmul(pred.permute(0, 2, 3, 1), transition_matrix).permute(0, 3, 1, 2)

            loss = criterion_seg(final_pred, labels)

            print_losses['seg'] += loss.item() * batch_size
            print_losses['total'] += loss.item() * batch_size

            # evaluation metrics
            preds = final_pred.max(dim=1)[1]

            for ii in range(batch_size):
                
                # ------------------------------------------------------------------ #
                # calculate OD IoU and Dice
                labels_OD = labels.clone()
                preds_OD = preds.clone()

                labels_OD[labels > 0] = 1
                labels_OD[labels == 0] = 0
                preds_OD[preds > 0] = 1
                preds_OD[preds == 0] = 0

                IoU += cal_IoU(labels, preds)
                Dice += cal_Dice(labels, preds)

                # ------------------------------------------------------------------ #
                if need_save:
                    
                    # save pred
                    gray = np.uint8(preds[ii].cpu().numpy())
                    color = colorize(gray, colormap)
                    image_name = data_name[ii].split('.')[0]          # eg: 414.jpg --> 414
                    color_folder = save_path + '/epoch_' + str(idx_epoch) + '/color/'
                    os.makedirs(color_folder, exist_ok=True)
                    color_path = os.path.join(color_folder, image_name + "_pred.png")
                    gray = Image.fromarray(gray)
                    color.save(color_path)

                    # save gt vis
                    gray = np.uint8(labels[ii].cpu().numpy())
                    color = colorize(gray, colormap)
                    image_name = data_name[ii].split('.')[0]          # eg: 414.jpg --> 414
                    color_folder = save_path + '/epoch_' + str(idx_epoch) + '/color/'
                    os.makedirs(color_folder, exist_ok=True)
                    color_path = os.path.join(color_folder, image_name + "_gt.png")
                    gray = Image.fromarray(gray)
                    color.save(color_path)

                    # save raw vis
                    toPIL = transforms.ToPILImage()
                    if normalized:
                        std = torch.Tensor([0.229, 0.224, 0.225]).to(device)
                        mean = torch.Tensor([0.485, 0.456, 0.406]).to(device)
                        un_norm = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
                        save_data = un_norm(data[ii])
                        pic = toPIL(save_data)
                        raw_path = os.path.join(color_folder, image_name + "_raw.jpg")
                        pic.save(raw_path)
                    else:
                        pic = toPIL(data[ii])
                        raw_path = os.path.join(color_folder, image_name + "_raw.jpg")
                        pic.save(raw_path)

            count += batch_size
            batch_idx += 1

    print_losses = {k: v * 1.0 / count for (k, v) in print_losses.items()}
    
    IoU = IoU / count
    Dice = Dice / count

    print_results['mean_IoU'] += IoU
    print_results['mean_Dice'] += Dice

    io.print_loss(mode, idx_epoch, print_losses)
    io.cprint("-----------------------------------------------------------------")
    io.print_result(mode, idx_epoch, print_results)

    return print_losses, print_results


def evaluate_save_optic_2B(mode, io, device, save_path, model, loader, idx_epoch=0, need_save=True, normalized=True):
    # treat both pred = 1 or 2 as disk
    # vis 2B prediction
    count = 0
    print_losses = {'total': 0.0, 'seg': 0.0}

    print_results = {'mean_IoU': 0.0, 'OD_IoU': 0.0, 'OC_IoU': 0.0, 'mean_Dice': 0.0, 'OD_Dice': 0.0, 'OC_Dice': 0.0, 'all_mean_IoU': 0.0, 'all_OD_IoU': 0.0, 'all_OC_IoU': 0.0, 'all_mean_Dice': 0.0, 'all_OD_Dice': 0.0, 'all_OC_Dice': 0.0, 'noisy_mean_IoU': 0.0, 'noisy_OD_IoU': 0.0, 'noisy_OC_IoU': 0.0, 'noisy_mean_Dice': 0.0, 'noisy_OD_Dice': 0.0, 'noisy_OC_Dice': 0.0, 'noisy_mean_IoU_noisy': 0.0, 'noisy_OD_IoU_noisy': 0.0, 'noisy_OC_IoU_noisy': 0.0, 'noisy_mean_Dice_noisy': 0.0, 'noisy_OD_Dice_noisy': 0.0, 'noisy_OC_Dice_noisy': 0.0}
    
    batch_idx = 0
    criterion_seg = torch.nn.CrossEntropyLoss(reduction='mean').to(device)

    colormap = create_breats_colormap()

    with torch.no_grad():
        model.eval()

        OD_IoU = 0.0
        OD_Dice = 0.0
        OC_IoU = 0.0
        OC_Dice = 0.0

        noisy_OD_IoU = 0.0
        noisy_OD_Dice = 0.0
        noisy_OC_IoU = 0.0
        noisy_OC_Dice = 0.0

        noisy_OD_IoU_noisy = 0.0
        noisy_OD_Dice_noisy = 0.0
        noisy_OC_IoU_noisy = 0.0
        noisy_OC_Dice_noisy = 0.0

        all_OD_IoU = 0.0
        all_OD_Dice = 0.0
        all_OC_IoU = 0.0
        all_OC_Dice = 0.0

        for data_all in loader:
            data, noisy_labels, labels, data_name = data_all[0].to(device), data_all[1].long().to(device).squeeze(), data_all[2].long().to(device).squeeze(), data_all[3]

            batch_size, _, img_H, img_W = data.shape

            if len(labels.shape) < 3:
                labels = labels.unsqueeze(0)

            logits = model(data)

            loss = criterion_seg(logits["pred"], labels)

            print_losses['seg'] += loss.item() * batch_size
            print_losses['total'] += loss.item() * batch_size

            # evaluation metrics
            preds = logits["pred"].max(dim=1)[1]
            preds_noisy = logits["noisy_pred"].max(dim=1)[1]
            preds_all = logits["all_pred"].max(dim=1)[1]

            for ii in range(batch_size):
                
                # ------------------------------------------------------------------ #
                # calculate OD IoU and Dice
                labels_OD = labels.clone()
                preds_OD = preds.clone()

                labels_OD[labels > 0] = 1
                labels_OD[labels == 0] = 0
                preds_OD[preds > 0] = 1
                preds_OD[preds == 0] = 0

                OD_IoU += cal_IoU(labels_OD, preds_OD)
                OD_Dice += cal_Dice(labels_OD, preds_OD)

                # calculate OC IoU and Dice
                labels_OC = labels.clone()
                preds_OC = preds.clone()

                labels_OC[labels == 2] = 1
                labels_OC[labels != 2] = 0
                preds_OC[preds == 2] = 1
                preds_OC[preds != 2] = 0

                OC_IoU += cal_IoU(labels_OC, preds_OC)
                OC_Dice += cal_Dice(labels_OC, preds_OC)

                # noisy
                labels_OD = labels.clone()
                preds_OD = preds_noisy.clone()

                labels_OD[labels > 0] = 1
                labels_OD[labels == 0] = 0
                preds_OD[preds_noisy > 0] = 1
                preds_OD[preds_noisy == 0] = 0

                noisy_OD_IoU += cal_IoU(labels_OD, preds_OD)
                noisy_OD_Dice += cal_Dice(labels_OD, preds_OD)

                # calculate OC IoU and Dice
                labels_OC = labels.clone()
                preds_OC = preds_noisy.clone()

                labels_OC[labels == 2] = 1
                labels_OC[labels != 2] = 0
                preds_OC[preds_noisy == 2] = 1
                preds_OC[preds_noisy != 2] = 0

                noisy_OC_IoU += cal_IoU(labels_OC, preds_OC)
                noisy_OC_Dice += cal_Dice(labels_OC, preds_OC)

                # noisy compared with noisy label
                labels_OD = noisy_labels.clone()
                preds_OD = preds_noisy.clone()

                labels_OD[noisy_labels > 0] = 1
                labels_OD[noisy_labels == 0] = 0
                preds_OD[preds_noisy > 0] = 1
                preds_OD[preds_noisy == 0] = 0

                noisy_OD_IoU_noisy += cal_IoU(labels_OD, preds_OD)
                noisy_OD_Dice_noisy += cal_Dice(labels_OD, preds_OD)

                # calculate OC IoU and Dice
                labels_OC = noisy_labels.clone()
                preds_OC = preds_noisy.clone()

                labels_OC[noisy_labels == 2] = 1
                labels_OC[noisy_labels != 2] = 0
                preds_OC[preds_noisy == 2] = 1
                preds_OC[preds_noisy != 2] = 0

                noisy_OC_IoU_noisy += cal_IoU(labels_OC, preds_OC)
                noisy_OC_Dice_noisy += cal_Dice(labels_OC, preds_OC)

                # all
                labels_OD = labels.clone()
                preds_OD = preds_all.clone()

                labels_OD[labels > 0] = 1
                labels_OD[labels == 0] = 0
                preds_OD[preds_all > 0] = 1
                preds_OD[preds_all == 0] = 0

                all_OD_IoU += cal_IoU(labels_OD, preds_OD)
                all_OD_Dice += cal_Dice(labels_OD, preds_OD)

                # calculate OC IoU and Dice
                labels_OC = labels.clone()
                preds_OC = preds_all.clone()

                labels_OC[labels == 2] = 1
                labels_OC[labels != 2] = 0
                preds_OC[preds_all == 2] = 1
                preds_OC[preds_all != 2] = 0

                all_OC_IoU += cal_IoU(labels_OC, preds_OC)
                all_OC_Dice += cal_Dice(labels_OC, preds_OC)

                # ------------------------------------------------------------------ #
                if need_save:
                    
                    # save pred
                    gray = np.uint8(preds[ii].cpu().numpy())
                    color = colorize(gray, colormap)
                    image_name = data_name[ii].split('.')[0]          # eg: 414.jpg --> 414
                    color_folder = save_path + '/epoch_' + str(idx_epoch) + '/color/'
                    os.makedirs(color_folder, exist_ok=True)
                    color_path = os.path.join(color_folder, image_name + "_pred.png")
                    gray = Image.fromarray(gray)
                    color.save(color_path)

                    # save noisy pred
                    gray = np.uint8(preds_noisy[ii].cpu().numpy())
                    color = colorize(gray, colormap)
                    image_name = data_name[ii].split('.')[0]          # eg: 414.jpg --> 414
                    color_folder = save_path + '/epoch_' + str(idx_epoch) + '/color/'
                    os.makedirs(color_folder, exist_ok=True)
                    color_path = os.path.join(color_folder, image_name + "_noisy_pred.png")
                    gray = Image.fromarray(gray)
                    color.save(color_path)

                    # save all pred
                    gray = np.uint8(preds_all[ii].cpu().numpy())
                    color = colorize(gray, colormap)
                    image_name = data_name[ii].split('.')[0]          # eg: 414.jpg --> 414
                    color_folder = save_path + '/epoch_' + str(idx_epoch) + '/color/'
                    os.makedirs(color_folder, exist_ok=True)
                    color_path = os.path.join(color_folder, image_name + "_all_pred.png")
                    gray = Image.fromarray(gray)
                    color.save(color_path)

                    # save gt vis
                    gray = np.uint8(labels[ii].cpu().numpy())
                    color = colorize(gray, colormap)
                    image_name = data_name[ii].split('.')[0]          # eg: 414.jpg --> 414
                    color_folder = save_path + '/epoch_' + str(idx_epoch) + '/color/'
                    os.makedirs(color_folder, exist_ok=True)
                    color_path = os.path.join(color_folder, image_name + "_gt.png")
                    gray = Image.fromarray(gray)
                    color.save(color_path)

                    # save raw vis
                    toPIL = transforms.ToPILImage()
                    if normalized:
                        std = torch.Tensor([0.229, 0.224, 0.225]).to(device)
                        mean = torch.Tensor([0.485, 0.456, 0.406]).to(device)
                        un_norm = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
                        save_data = un_norm(data[ii])
                        pic = toPIL(save_data)
                        raw_path = os.path.join(color_folder, image_name + "_raw.jpg")
                        pic.save(raw_path)
                    else:
                        pic = toPIL(data[ii])
                        raw_path = os.path.join(color_folder, image_name + "_raw.jpg")
                        pic.save(raw_path)

            count += batch_size
            batch_idx += 1

    print_losses = {k: v * 1.0 / count for (k, v) in print_losses.items()}
    
    OD_IoU = OD_IoU / count
    OD_Dice = OD_Dice / count
    OC_IoU = OC_IoU / count
    OC_Dice = OC_Dice / count
    mean_IoU = (OD_IoU + OC_IoU) / 2
    mean_Dice = (OD_Dice + OC_Dice) / 2

    all_OD_IoU = all_OD_IoU / count
    all_OD_Dice = all_OD_Dice / count
    all_OC_IoU = all_OC_IoU / count
    all_OC_Dice = all_OC_Dice / count
    all_mean_IoU = (all_OD_IoU + all_OC_IoU) / 2
    all_mean_Dice = (all_OD_Dice + all_OC_Dice) / 2

    noisy_OD_IoU = noisy_OD_IoU / count
    noisy_OD_Dice = noisy_OD_Dice / count
    noisy_OC_IoU = noisy_OC_IoU / count
    noisy_OC_Dice = noisy_OC_Dice / count
    noisy_mean_IoU = (noisy_OD_IoU + noisy_OC_IoU) / 2
    noisy_mean_Dice = (noisy_OD_Dice + noisy_OC_Dice) / 2

    noisy_OD_IoU_noisy = noisy_OD_IoU_noisy / count
    noisy_OD_Dice_noisy = noisy_OD_Dice_noisy / count
    noisy_OC_IoU_noisy = noisy_OC_IoU_noisy / count
    noisy_OC_Dice_noisy = noisy_OC_Dice_noisy / count
    noisy_mean_IoU_noisy = (noisy_OD_IoU_noisy + noisy_OC_IoU_noisy) / 2
    noisy_mean_Dice_noisy = (noisy_OD_Dice_noisy + noisy_OC_Dice_noisy) / 2

    print_results['mean_IoU'] += mean_IoU
    print_results['mean_Dice'] += mean_Dice
    print_results['OD_IoU'] += OD_IoU
    print_results['OD_Dice'] += OD_Dice
    print_results['OC_IoU'] += OC_IoU
    print_results['OC_Dice'] += OC_Dice

    print_results['all_mean_IoU'] += all_mean_IoU
    print_results['all_mean_Dice'] += all_mean_Dice
    print_results['all_OD_IoU'] += all_OD_IoU
    print_results['all_OD_Dice'] += all_OD_Dice
    print_results['all_OC_IoU'] += all_OC_IoU
    print_results['all_OC_Dice'] += all_OC_Dice

    print_results['noisy_mean_IoU'] += noisy_mean_IoU
    print_results['noisy_mean_Dice'] += noisy_mean_Dice
    print_results['noisy_OD_IoU'] += noisy_OD_IoU
    print_results['noisy_OD_Dice'] += noisy_OD_Dice
    print_results['noisy_OC_IoU'] += noisy_OC_IoU
    print_results['noisy_OC_Dice'] += noisy_OC_Dice

    print_results['noisy_mean_IoU_noisy'] += noisy_mean_IoU_noisy
    print_results['noisy_mean_Dice_noisy'] += noisy_mean_Dice_noisy
    print_results['noisy_OD_IoU_noisy'] += noisy_OD_IoU_noisy
    print_results['noisy_OD_Dice_noisy'] += noisy_OD_Dice_noisy
    print_results['noisy_OC_IoU_noisy'] += noisy_OC_IoU_noisy
    print_results['noisy_OC_Dice_noisy'] += noisy_OC_Dice_noisy
    
    io.print_loss(mode, idx_epoch, print_losses)
    io.cprint("-----------------------------------------------------------------")
    io.print_result(mode, idx_epoch, print_results)

    return print_losses, print_results


def evaluate_save_optic_2cls(mode, io, device, save_path, model, loader, idx_epoch=0, need_save=True, normalized=True):
    # treat both pred = 1 or 2 as disk
    # one feature extractor, two cls
    # count clean pred and noisy pred
    count = 0
    print_losses = {'total': 0.0, 'seg': 0.0}

    print_results = {'mean_IoU': 0.0, 'OD_IoU': 0.0, 'OC_IoU': 0.0, 'mean_Dice': 0.0, 'OD_Dice': 0.0, 'OC_Dice': 0.0, 'noisy_mean_IoU': 0.0, 'noisy_OD_IoU': 0.0, 'noisy_OC_IoU': 0.0, 'noisy_mean_Dice': 0.0, 'noisy_OD_Dice': 0.0, 'noisy_OC_Dice': 0.0}
    
    batch_idx = 0
    criterion_seg = torch.nn.CrossEntropyLoss(reduction='mean').to(device)

    colormap = create_breats_colormap()
    colormap_confidence = create_optic_colormap_confidence()

    with torch.no_grad():
        model.eval()

        OD_IoU = 0.0
        OD_Dice = 0.0
        OC_IoU = 0.0
        OC_Dice = 0.0

        noisy_OD_IoU = 0.0
        noisy_OD_Dice = 0.0
        noisy_OC_IoU = 0.0
        noisy_OC_Dice = 0.0

        for data_all in loader:
            data, labels, data_name = data_all[0].to(device), data_all[1].long().to(device).squeeze(), data_all[2]

            batch_size, _, img_H, img_W = data.shape

            if len(labels.shape) < 3:
                labels = labels.unsqueeze(0)

            logits = model(data)

            loss = criterion_seg(logits["pred"], labels)

            print_losses['seg'] += loss.item() * batch_size
            print_losses['total'] += loss.item() * batch_size

            # evaluation metrics
            preds = logits["pred"].max(dim=1)[1]
            noisy_preds = logits["noisy_pred"].max(dim=1)[1]

            preds_confidence = logits["pred"].softmax(dim=1).max(dim=1)[0]
            # divide into 20 classes
            preds_confidence = (preds_confidence / 0.05).floor().long()

            for ii in range(batch_size):
                
                # ------------------------------------------------------------------ #
                # calculate OD IoU and Dice
                labels_OD = labels.clone()
                preds_OD = preds.clone()

                labels_OD[labels > 0] = 1
                labels_OD[labels == 0] = 0
                preds_OD[preds > 0] = 1
                preds_OD[preds == 0] = 0

                OD_IoU += cal_IoU(labels_OD, preds_OD)
                OD_Dice += cal_Dice(labels_OD, preds_OD)

                # calculate OC IoU and Dice
                labels_OC = labels.clone()
                preds_OC = preds.clone()

                labels_OC[labels == 2] = 1
                labels_OC[labels != 2] = 0
                preds_OC[preds == 2] = 1
                preds_OC[preds != 2] = 0

                OC_IoU += cal_IoU(labels_OC, preds_OC)
                OC_Dice += cal_Dice(labels_OC, preds_OC)

                # ------------------------------------------------------------------ #
                # calculate noisy OD IoU and Dice
                labels_OD = labels.clone()
                noisy_preds_OD = noisy_preds.clone()

                labels_OD[labels > 0] = 1
                labels_OD[labels == 0] = 0
                noisy_preds_OD[noisy_preds > 0] = 1
                noisy_preds_OD[noisy_preds == 0] = 0

                noisy_OD_IoU += cal_IoU(labels_OD, noisy_preds_OD)
                noisy_OD_Dice += cal_Dice(labels_OD, noisy_preds_OD)

                # calculate OC IoU and Dice
                labels_OC = labels.clone()
                noisy_preds_OC = noisy_preds.clone()

                labels_OC[labels == 2] = 1
                labels_OC[labels != 2] = 0
                noisy_preds_OC[noisy_preds == 2] = 1
                noisy_preds_OC[noisy_preds != 2] = 0

                noisy_OC_IoU += cal_IoU(labels_OC, noisy_preds_OC)
                noisy_OC_Dice += cal_Dice(labels_OC, noisy_preds_OC)

                # ------------------------------------------------------------------ #
                if need_save:
                    
                    # save pred
                    gray = np.uint8(preds[ii].cpu().numpy())
                    color = colorize(gray, colormap)
                    image_name = data_name[ii].split('.')[0]          # eg: 414.jpg --> 414
                    color_folder = save_path + '/epoch_' + str(idx_epoch) + '/color/'
                    os.makedirs(color_folder, exist_ok=True)
                    color_path = os.path.join(color_folder, image_name + "_pred.png")
                    gray = Image.fromarray(gray)
                    color.save(color_path)

                    # save gt vis
                    gray = np.uint8(labels[ii].cpu().numpy())
                    color = colorize(gray, colormap)
                    image_name = data_name[ii].split('.')[0]          # eg: 414.jpg --> 414
                    color_folder = save_path + '/epoch_' + str(idx_epoch) + '/color/'
                    os.makedirs(color_folder, exist_ok=True)
                    color_path = os.path.join(color_folder, image_name + "_gt.png")
                    gray = Image.fromarray(gray)
                    color.save(color_path)

                    # save raw vis
                    toPIL = transforms.ToPILImage()
                    if normalized:
                        std = torch.Tensor([0.229, 0.224, 0.225]).to(device)
                        mean = torch.Tensor([0.485, 0.456, 0.406]).to(device)
                        un_norm = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
                        save_data = un_norm(data[ii])
                        pic = toPIL(save_data)
                        raw_path = os.path.join(color_folder, image_name + "_raw.jpg")
                        pic.save(raw_path)
                    else:
                        pic = toPIL(data[ii])
                        raw_path = os.path.join(color_folder, image_name + "_raw.jpg")
                        pic.save(raw_path)

                    # save pred confidence

                    gray = np.uint8(preds_confidence[ii].cpu().numpy())
                    color = colorize(gray, colormap_confidence)
                    image_name = data_name[ii].split('.')[0]          # eg: 414.jpg --> 414
                    color_folder = save_path + '/epoch_' + str(idx_epoch) + '/color/'
                    os.makedirs(color_folder, exist_ok=True)
                    color_path = os.path.join(color_folder, image_name + "_pred_confidence.png")
                    gray = Image.fromarray(gray)
                    color.save(color_path)

            count += batch_size
            batch_idx += 1

    print_losses = {k: v * 1.0 / count for (k, v) in print_losses.items()}
    
    OD_IoU = OD_IoU / count
    OD_Dice = OD_Dice / count
    OC_IoU = OC_IoU / count
    OC_Dice = OC_Dice / count
    mean_IoU = (OD_IoU + OC_IoU) / 2
    mean_Dice = (OD_Dice + OC_Dice) / 2

    print_results['mean_IoU'] += mean_IoU
    print_results['mean_Dice'] += mean_Dice
    print_results['OD_IoU'] += OD_IoU
    print_results['OD_Dice'] += OD_Dice
    print_results['OC_IoU'] += OC_IoU
    print_results['OC_Dice'] += OC_Dice

    noisy_OD_IoU = noisy_OD_IoU / count
    noisy_OD_Dice = noisy_OD_Dice / count
    noisy_OC_IoU = noisy_OC_IoU / count
    noisy_OC_Dice = noisy_OC_Dice / count
    noisy_mean_IoU = (noisy_OD_IoU + noisy_OC_IoU) / 2
    noisy_mean_Dice = (noisy_OD_Dice + noisy_OC_Dice) / 2

    print_results['noisy_mean_IoU'] += noisy_mean_IoU
    print_results['noisy_mean_Dice'] += noisy_mean_Dice
    print_results['noisy_OD_IoU'] += noisy_OD_IoU
    print_results['noisy_OD_Dice'] += noisy_OD_Dice
    print_results['noisy_OC_IoU'] += noisy_OC_IoU
    print_results['noisy_OC_Dice'] += noisy_OC_Dice

    io.print_loss(mode, idx_epoch, print_losses)
    io.cprint("-----------------------------------------------------------------")
    io.print_result(mode, idx_epoch, print_results)

    return print_losses, print_results
