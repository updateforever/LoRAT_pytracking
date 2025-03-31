from timm.layers import to_2tuple

from lib.models.lorat import build_lorat
from lib.test.tracker.basetracker import BaseTracker
import torch

# for debug
import cv2
import numpy as np

from lib.test.tracker.data_utils import Preprocessor
from lib.train.data.siamfc_cropping import get_siamfc_cropping_params, get_foreground_bounding_box
from lib.train.data.siamfc_cropping import apply_siamfc_cropping, reverse_siamfc_cropping_params, apply_siamfc_cropping_to_boxes
from lib.utils.bbox.utility.image import bbox_clip_to_image_boundary_, is_bbox_intersecting_image
from lib.utils.bbox.validity import bbox_is_valid
from lib.utils.bbox.format import bbox_xyxy_to_cxcywh, bbox_xyxy_to_xywh, bbox_cxcywh_to_xyxy

from contextlib import nullcontext
from functools import partial
import torchvision.transforms as transforms
import math

# TODO:
# 1. 构建feat_mask_z来适应输入
# 2. 构建inference_mode和amp
# 3. crop的时候应该要用template_image_mean
# 4. 后处理的window_penalty
# 5. 预测在search上之后要map回到原图

class LoRAT(BaseTracker):
    def __init__(self, params):
        super(LoRAT, self).__init__(params)
        network = build_lorat(params.cfg, training=False)
        ckpt = torch.load(self.params.checkpoint, map_location='cpu')
        if 'net' in ckpt:
            ckpt = ckpt['net']
        network.load_state_dict(ckpt, strict=True)  # , map_location='cpu'
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None
        self.device = self.preprocessor.mean.device
        self._window_penalty_ratio = params.window_penalty

        self.stride = self.cfg.MODEL.BACKBONE.STRIDE if 'patch8' not in self.cfg.MODEL.BACKBONE.TYPE else 16
        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.stride
        self.feat_sz_z = self.cfg.TEST.TEMPLATE_SIZE // self.stride

        # for debug
        self.debug = 0 # params.debug
        self.use_visdom = 0  # params.debug
        print(f"Tracking: debug: {self.debug} | use_visdom: {self.use_visdom}")
        self.frame_id = 0
        if self.debug:
            if not self.use_visdom:
                self.save_dir = "debug"
                # if not os.path.exists(self.save_dir):
                #     os.makedirs(self.save_dir)
            else:
                # self.add_hook()
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = None
        self.image_normalize_transform = transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def initialize(self, image, info: dict):
        init_gt_bbox = np.asarray(info['init_bbox'], dtype=np.float64)
        init_gt_bbox[..., 2:] += init_gt_bbox[..., :2]  # xywh -> xyxy
        self.cached_bbox = init_gt_bbox.copy()


        z = torch.from_numpy(image).permute(2, 0, 1).to(self.device)
        template_curation_parameter = get_siamfc_cropping_params(init_gt_bbox, self.params.template_factor, np.array(to_2tuple(self.params.template_size)))

        z_curated, z_image_mean, template_curation_parameter = apply_siamfc_cropping(
            z.to(torch.float32), np.array(to_2tuple(self.params.template_size)), template_curation_parameter, "bilinear", False)

        z_curated.div_(255.)
        z_curated = self.image_normalize_transform(z_curated).unsqueeze(0)
        self.template_image_mean = z_image_mean

        with torch.inference_mode():
            self.z_dict1 = z_curated

        # construct "z_feat_mask"
        template_mask = torch.full((self.feat_sz_z, self.feat_sz_z), 0, dtype=torch.long)
        template_cropped_bbox = get_foreground_bounding_box(init_gt_bbox, template_curation_parameter, to_2tuple(self.stride))
        assert bbox_is_valid(template_cropped_bbox)  # xyxy format
        template_cropped_bbox = torch.from_numpy(template_cropped_bbox)
        template_mask[template_cropped_bbox[1]: template_cropped_bbox[3], template_cropped_bbox[0]: template_cropped_bbox[2]] = 1
        self.z_feat_mask = template_mask.unsqueeze(0).to(self.device)

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        image_size = np.array((W, H), dtype=np.int32)
        self.frame_id += 1

        x = torch.from_numpy(image).permute(2, 0, 1).to(self.device)

        cropping_params = get_siamfc_cropping_params(self._adjust_bbox_size(self.cached_bbox, np.array(to_2tuple(10.0), dtype=float)),
                                                     self.params.search_factor, np.asarray(to_2tuple(self.cfg.TEST.SEARCH_SIZE)))

        x_curated, _, x_cropping_params = apply_siamfc_cropping(x.to(torch.float32), np.array(to_2tuple(self.cfg.TEST.SEARCH_SIZE)),
                                                cropping_params, "bilinear", False, self.template_image_mean)
        x_patch_arr = x_curated
        resize_factor = x_cropping_params
        search_box = self.search_box(image, self.state, self.params.search_factor)  # get search box wyp
        # resize_factor = adjusted_cropping_params[0][0]

        x_curated = x_curated.unsqueeze(0) / 255.
        x_curated = self.image_normalize_transform(x_curated)

        amp_autocast_fn = self.get_torch_amp_autocast_fn(self.device.type, True, torch.float16)

        with torch.inference_mode(), amp_autocast_fn():
            out_dict = self.network.forward(
                z=self.z_dict1, x=x_curated, z_feat_mask=self.z_feat_mask)

        # add hann windows
        pred_outputs = self.postprocessing_boxwithscoremap(output=out_dict)
        pred_boxes = pred_outputs['box'].view(4).cpu().to(torch.float64).numpy()
        confidence = pred_outputs['confidence'].item()
        pred_boxes_on_full_search_image = apply_siamfc_cropping_to_boxes(
            pred_boxes, reverse_siamfc_cropping_params(x_cropping_params))
        bbox_clip_to_image_boundary_(pred_boxes_on_full_search_image, image_size)
        self.update_cropping_params(pred_boxes_on_full_search_image, image_size)
        self.state = bbox_xyxy_to_xywh(pred_boxes_on_full_search_image).tolist()

        # wyp
        all_pred_boxes = pred_outputs['soi_info']['all_bounding_boxes'].view(-1, 4).cpu().to(torch.float64).numpy()
        all_pred_boxes_on_full_search_image = np.array([
            apply_siamfc_cropping_to_boxes(box, reverse_siamfc_cropping_params(x_cropping_params))
            for box in all_pred_boxes
        ])
        bbox_clip_to_image_boundary_(all_pred_boxes_on_full_search_image, image_size)
        all_state = bbox_xyxy_to_xywh(all_pred_boxes_on_full_search_image).tolist()


        self.distractor_dataset_data = dict(score_map=pred_outputs['soi_info']['score_map'],  # 1 1 h w
                            # sample_pos=sample_pos[scale_ind, :],
                            sample_scale=resize_factor,
                            search_area_box=search_box, 
                            x_dict=x_patch_arr,  # 3 378 378   384 384 3
                            all_scoremap_boxes=all_state
                            )

        return {"target_bbox": self.state, "confidence": confidence}

    def update_cropping_params(self, predicted_bbox: np.ndarray, image_size: np.ndarray):
        assert image_size[0] > 0 and image_size[1] > 0

        if not bbox_is_valid(predicted_bbox):
            return
        if not is_bbox_intersecting_image(predicted_bbox, image_size):
            return
        self.cached_bbox = predicted_bbox

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights

    def postprocessing_boxwithscoremap(self, output):
        self._scale_factor = torch.tensor((self.params.search_size, self.params.search_size), dtype=torch.float).to(self.device)
        self._window = torch.flatten(torch.outer(torch.hann_window(self.feat_sz, periodic=False),
                                                 torch.hann_window(self.feat_sz, periodic=False))).to(self.device)

        # shape: (N, H, W), (N, H, W, 4)
        predicted_score_map = output['score_map'].detach().float().sigmoid()
        predicted_bbox = output['boxes'].detach().float()

        N, H, W = predicted_score_map.shape
        predicted_score_map = predicted_score_map.view(N, H * W)

        # window penalty
        score_map_with_penalty = predicted_score_map * (1 - self._window_penalty_ratio) + \
                                 self._window.view(1, H * W) * self._window_penalty_ratio
        _, best_idx = torch.max(score_map_with_penalty, 1, keepdim=True)
        confidence_score = torch.gather(predicted_score_map, 1, best_idx)

        confidence_score = confidence_score.squeeze(1)
        predicted_bbox = predicted_bbox.view(N, H * W, 4)
        bounding_box = torch.gather(predicted_bbox, 1, best_idx.view(N, 1, 1).expand(-1, -1, 4)).squeeze(1)
        bounding_box = (bounding_box.view(N, 2, 2) * self._scale_factor.view(1, 1, 2)).view(N, 4)

        # wyp get all boxes
        all_bounding_boxes = predicted_bbox.squeeze(1)
        all_bounding_boxes = (all_bounding_boxes.view(N, H * W, 2, 2) * self._scale_factor.view(1, 1, 1, 2)).view(N, H * W, 4)
        soi_info = {'score_map': score_map_with_penalty, 'all_bounding_boxes': all_bounding_boxes}
        return {'box': bounding_box, 'confidence': confidence_score, 'soi_info': soi_info}


    def get_torch_amp_autocast_fn(self, device_type: str, enabled: bool, dtype: torch.dtype):
        # to be removed in the future, once they are supported
        if device_type == 'mps' and enabled:
            print('Auto mixed precision is disabled. reason: Auto mixed precision is not supported on MPS.', flush=True)
            enabled = False
        if device_type == 'cpu' and dtype == torch.float16 and enabled:
            dtype = torch.bfloat16
            print(f'Warning: CPU does not support float16, use bfloat16 instead', flush=True)
        return partial(torch.amp.autocast, device_type=device_type, enabled=enabled, dtype=dtype) if enabled else nullcontext

    def _adjust_bbox_size(self, bounding_box: np.ndarray, min_wh: np.ndarray):
        bounding_box = bbox_xyxy_to_cxcywh(bounding_box)
        bounding_box[2:4] = np.maximum(bounding_box[2:4], min_wh)
        return bbox_cxcywh_to_xyxy(bounding_box)


    def search_box(self, im, state, search_area_factor):
        # 搜索框坐标
        if not isinstance(state, list):
            x, y, w, h = state.tolist()
        else:
            x, y, w, h = state
        crop_sz = math.ceil(math.sqrt(w * h) * search_area_factor)  # 模板区域sz
        x1 = round(x + 0.5 * w - crop_sz * 0.5)
        x2 = x1 + crop_sz
        y1 = round(y + 0.5 * h - crop_sz * 0.5)
        y2 = y1 + crop_sz
        x1_pad = max(0, -x1)
        x2_pad = max(x2 - im.shape[1] + 1, 0)
        y1_pad = max(0, -y1)
        y2_pad = max(y2 - im.shape[0] + 1, 0)
        im_crop = [x1+x1_pad, y1+y1_pad, x2 - x2_pad -x1 - x1_pad, y2 - y2_pad - y1 - y1_pad] # x,y,h,w

        return im_crop
    

def get_tracker_class():
    return LoRAT