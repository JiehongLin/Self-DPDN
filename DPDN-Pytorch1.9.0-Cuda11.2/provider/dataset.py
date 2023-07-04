import os
import math
import cv2
import glob
import numpy as np
import _pickle as cPickle
from PIL import Image

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from data_utils import load_depth, load_composed_depth, get_bbox, fill_missing


class TrainingDataset(Dataset):
    def __init__(self, config, data_dir, data_type='real', num_img_per_epoch=-1):
        self.config = config
        self.data_dir = data_dir
        self.data_type = data_type
        self.num_img_per_epoch = num_img_per_epoch

        self.img_size = self.config.img_size
        self.sample_num = self.config.sample_num

        if data_type == 'syn':
            img_path = 'camera/train_list.txt'
            model_path = 'obj_models/camera_train.pkl'
            self.intrinsics = [577.5, 577.5, 319.5, 239.5]
        elif data_type == 'real_woLabel' or data_type == 'real_withLabel':
            img_path = 'real/train_list.txt'
            model_path = 'obj_models/real_train.pkl'
            self.intrinsics = [591.0125, 590.16775, 322.525, 244.11084]
        else:
            assert False, 'wrong data type of {} in data loader !'.format(data_type)

        self.img_list = [os.path.join(img_path.split('/')[0], line.rstrip('\n'))
                        for line in open(os.path.join(self.data_dir, img_path))]
        self.img_index = np.arange(len(self.img_list))

        self.models = {}
        with open(os.path.join(self.data_dir, model_path), 'rb') as f:
            self.models.update(cPickle.load(f))

        self.mean_shapes = np.load(os.path.join(self.data_dir, 'mean_shapes.npy')).astype(np.float32)

        self.xmap = np.array([[i for i in range(640)] for j in range(480)])
        self.ymap = np.array([[j for i in range(640)] for j in range(480)])
        self.sym_ids = [0, 1, 3]    # 0-indexed
        self.norm_scale = 1000.0    # normalization scale
        self.colorjitter = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

        print('{} images found.'.format(len(self.img_list)))
        print('{} models loaded.'.format(len(self.models))) 

    def __len__(self):
        if self.num_img_per_epoch == -1:
            return len(self.img_list)
        else:
            return self.num_img_per_epoch

    def reset(self):
        assert self.num_img_per_epoch != -1
        num_img = len(self.img_list)
        if num_img <= self.num_img_per_epoch:
            self.img_index = np.random.choice(num_img, self.num_img_per_epoch)
        else:
            self.img_index = np.random.choice(num_img, self.num_img_per_epoch, replace=False)

    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir, self.img_list[self.img_index[index]])
        if self.data_type == 'syn':
            depth = load_composed_depth(img_path)
        else:
            depth = load_depth(img_path)
        if depth is None:
            index = np.random.randint(self.__len__())
            return self.__getitem__(index)
        depth = fill_missing(depth, self.norm_scale, 1)

        # mask
        if self.data_type == 'syn' or self.data_type == 'real_withLabel':
            with open(img_path + '_label.pkl', 'rb') as f:
                gts = cPickle.load(f)
            num_instance = len(gts['instance_ids'])
            assert(len(gts['class_ids'])==len(gts['instance_ids']))
            mask = cv2.imread(img_path + '_mask.png')[:, :, 2] #480*640

            idx = np.random.randint(0, num_instance)
            cat_id = gts['class_ids'][idx] - 1 # convert to 0-indexed
            rmin, rmax, cmin, cmax = get_bbox(gts['bboxes'][idx])
            mask = np.equal(mask, gts['instance_ids'][idx])
            
        else:
            img_path_parsing = img_path.split('/')
            with open(os.path.join(self.data_dir, 'segmentation_results', 'train_trainedwoMask', img_path_parsing[-2], img_path_parsing[-1] + '.pkl'), 'rb') as f:
                preds = cPickle.load(f)
            num_instance = len(preds['pred_class_ids'])
            if num_instance > 0:
                idx = np.random.randint(0, num_instance)
                cat_id = preds['pred_class_ids'][idx] - 1 # convert to 0-indexed
                if cat_id<0:
                    index = np.random.randint(self.__len__())
                    return self.__getitem__(index)
                rmin, rmax, cmin, cmax = get_bbox(preds['pred_bboxes'][idx])
                mask = 255 * preds['pred_mask'][:, :, idx].astype('uint8') > 0
            else:
                index = np.random.randint(self.__len__())
                return self.__getitem__(index)                

        mask = np.logical_and(mask , depth > 0)

        # choose
        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]
        if len(choose)<=0:
            index = np.random.randint(self.__len__())
            return self.__getitem__(index)
        if len(choose) <= self.sample_num:
            choose_idx = np.random.choice(len(choose), self.sample_num)
        else:
            choose_idx = np.random.choice(len(choose), self.sample_num, replace=False)
        choose = choose[choose_idx]

        # pts
        cam_fx, cam_fy, cam_cx, cam_cy = self.intrinsics
        pts2 = depth.copy() / self.norm_scale
        pts0 = (self.xmap - cam_cx) * pts2 / cam_fx
        pts1 = (self.ymap - cam_cy) * pts2 / cam_fy
        pts = np.transpose(np.stack([pts0, pts1, pts2]), (1,2,0)).astype(np.float32) # 480*640*3
        pts = pts[rmin:rmax, cmin:cmax, :].reshape((-1, 3))[choose, :]
        pts = pts + np.clip(0.001*np.random.randn(pts.shape[0], 3), -0.005, 0.005)

        # rgb
        rgb = cv2.imread(img_path + '_color.png')[:, :, :3]
        rgb = rgb[:, :, ::-1] #480*640*3
        rgb = rgb[rmin:rmax, cmin:cmax, :]
        rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        # color jitter
        rgb = self.colorjitter(Image.fromarray(np.uint8(rgb)))
        rgb = self.transform(np.array(rgb))

        # update choose
        crop_w = rmax - rmin
        ratio = self.img_size / crop_w
        col_idx = choose % crop_w
        row_idx = choose // crop_w
        choose = (np.floor(row_idx * ratio) * self.img_size + np.floor(col_idx * ratio)).astype(np.int64)

        # prior
        prior = self.mean_shapes[cat_id].astype(np.float32)

        ret_dict = {}
        ret_dict['pts'] = torch.FloatTensor(pts) # N*3
        ret_dict['rgb'] = torch.FloatTensor(rgb)
        ret_dict['choose'] = torch.IntTensor(choose).long()
        ret_dict['prior'] = torch.FloatTensor(prior)
        ret_dict['category_label'] = torch.IntTensor([cat_id]).long()

        if self.data_type == 'syn' or self.data_type == 'real_withLabel':
            model = self.models[gts['model_list'][idx]].astype(np.float32)
            translation = gts['translations'][idx].astype(np.float32)
            rotation = gts['rotations'][idx].astype(np.float32)
            size = gts['scales'][idx] * gts['sizes'][idx].astype(np.float32)

            if cat_id in self.sym_ids:
                theta_x = rotation[0, 0] + rotation[2, 2]
                theta_y = rotation[0, 2] - rotation[2, 0]
                r_norm = math.sqrt(theta_x**2 + theta_y**2)
                s_map = np.array([[theta_x/r_norm, 0.0, -theta_y/r_norm],
                                    [0.0,            1.0,  0.0           ],
                                    [theta_y/r_norm, 0.0,  theta_x/r_norm]])
                rotation = rotation @ s_map
            qo = (pts - translation[np.newaxis, :]) / (np.linalg.norm(size)+1e-8) @ rotation

            ret_dict['model'] = torch.FloatTensor(model)
            ret_dict['qo'] = torch.FloatTensor(qo)
            ret_dict['translation_label'] = torch.FloatTensor(translation)
            ret_dict['rotation_label'] = torch.FloatTensor(rotation)
            ret_dict['size_label'] = torch.FloatTensor(size)

        return ret_dict


class TestDataset():
    def __init__(self, config, data_dir, setting):
        self.data_dir = data_dir
        self.setting = setting

        self.img_size = config.img_size
        self.sample_num = config.sample_num
        self.intrinsics = [591.0125, 590.16775, 322.525, 244.11084]

        result_pkl_list = glob.glob(os.path.join(self.data_dir, 'data', 'segmentation_results', 'test_trainedwithMask', 'results_*.pkl'))
        self.result_pkl_list = sorted(result_pkl_list)
        n_image = len(result_pkl_list)
        print('no. of test images: {}\n'.format(n_image))

        self.mean_shapes = np.load(os.path.join(self.data_dir, 'data', 'mean_shapes.npy')).astype(np.float32)

        self.xmap = np.array([[i for i in range(640)] for j in range(480)])
        self.ymap = np.array([[j for i in range(640)] for j in range(480)])
        self.sym_ids = [0, 1, 3]    # 0-indexed
        self.norm_scale = 1000.0    # normalization scale
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.result_pkl_list)

    def __getitem__(self, index):
        path = self.result_pkl_list[index]

        with open(path, 'rb') as f:
            data = cPickle.load(f)
        image_path = os.path.join(self.data_dir, data['image_path'])

        if self.setting == 'supervised' or self.setting == 'unsupervised_withMask':
            pred_data = data
            pred_mask = data['pred_masks']
        elif self.setting == 'unsupervised':
            with open(os.path.join(self.data_dir, 'data', 'segmentation_results', 'test_trainedwoMask', path.split('/')[-1]), 'rb') as f:
                pred_data = cPickle.load(f)
            pred_mask = pred_data['pred_mask']
        else:
            assert False
        num_instance = len(pred_data['pred_class_ids'])

        # rgb
        rgb = cv2.imread(image_path + '_color.png')[:, :, :3]
        rgb = rgb[:, :, ::-1] #480*640*3

        # pts
        cam_fx, cam_fy, cam_cx, cam_cy = self.intrinsics
        depth = load_depth(image_path) #480*640
        depth = fill_missing(depth, self.norm_scale, 1)

        xmap = self.xmap
        ymap = self.ymap
        pts2 = depth.copy() / self.norm_scale
        pts0 = (xmap - cam_cx) * pts2 / cam_fx
        pts1 = (ymap - cam_cy) * pts2 / cam_fy
        pts = np.transpose(np.stack([pts0, pts1, pts2]), (1,2,0)).astype(np.float32) # 480*640*3

        all_rgb = []
        all_pts = []
        all_prior = []
        all_cat_ids = []
        all_choose = []
        flag_instance = torch.zeros(num_instance) == 1

        for j in range(num_instance):
            inst_mask = 255 * pred_mask[:, :, j].astype('uint8')
            rmin, rmax, cmin, cmax = get_bbox(pred_data['pred_bboxes'][j])
            mask = inst_mask > 0
            mask = np.logical_and(mask, depth>0)
            choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

            if len(choose)>16:
                if len(choose) <= self.sample_num:
                    choose_idx = np.random.choice(len(choose), self.sample_num)
                else:
                    choose_idx = np.random.choice(len(choose), self.sample_num, replace=False)
                choose = choose[choose_idx]
                instance_pts = pts[rmin:rmax, cmin:cmax, :].reshape((-1, 3))[choose, :]

                instance_rgb = rgb[rmin:rmax, cmin:cmax, :].copy()
                instance_rgb = cv2.resize(instance_rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
                instance_rgb = self.transform(np.array(instance_rgb))
                crop_w = rmax - rmin
                ratio = self.img_size / crop_w
                col_idx = choose % crop_w
                row_idx = choose // crop_w
                choose = (np.floor(row_idx * ratio) * self.img_size + np.floor(col_idx * ratio)).astype(np.int64)

                cat_id = pred_data['pred_class_ids'][j] - 1 # convert to 0-indexed
                prior = self.mean_shapes[cat_id].astype(np.float32)

                all_pts.append(torch.FloatTensor(instance_pts))
                all_rgb.append(torch.FloatTensor(instance_rgb))
                all_prior.append(torch.FloatTensor(prior))
                all_cat_ids.append(torch.IntTensor([cat_id]).long())
                all_choose.append(torch.IntTensor(choose).long())
                flag_instance[j] = 1

        ret_dict = {}
        ret_dict['pts'] = torch.stack(all_pts) # N*3
        ret_dict['rgb'] = torch.stack(all_rgb)
        ret_dict['choose'] = torch.stack(all_choose)
        ret_dict['prior'] = torch.stack(all_prior)
        ret_dict['category_label'] = torch.stack(all_cat_ids).squeeze(1)

        ret_dict['gt_class_ids'] = torch.tensor(data['gt_class_ids'])
        ret_dict['gt_bboxes'] = torch.tensor(data['gt_bboxes'])
        ret_dict['gt_RTs'] = torch.tensor(data['gt_RTs'])
        ret_dict['gt_scales'] = torch.tensor(data['gt_scales'])
        ret_dict['gt_handle_visibility'] = torch.tensor(data['gt_handle_visibility'])

        ret_dict['pred_class_ids'] = torch.tensor(pred_data['pred_class_ids'])[flag_instance==1]
        ret_dict['pred_bboxes'] = torch.tensor(pred_data['pred_bboxes'])[flag_instance==1]
        ret_dict['pred_scores'] = torch.tensor(pred_data['pred_scores'])[flag_instance==1]
        ret_dict['index'] = torch.IntTensor([index])
        return ret_dict
