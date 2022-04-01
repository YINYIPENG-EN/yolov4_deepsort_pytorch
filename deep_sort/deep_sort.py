import numpy as np
import torch
import sys

from IPython import embed

from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.detection import Detection
from .sort.tracker import Tracker

sys.path.append('deep_sort/deep/reid')
from deep_sort.deep.reid.torchreid.utils import FeatureExtractor

__all__ = ['DeepSort']


class DeepSort(object):
    def __init__(self, model_type, device, max_dist=0.2, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100):

        self.extractor = FeatureExtractor(  # 特征提取
            model_name=model_type,  # osnet_x0_25
            device=str(device)  # cpu or gpu
        )

        max_cosine_distance = max_dist  # 距离
        metric = NearestNeighborDistanceMetric(
            "euclidean", max_cosine_distance, nn_budget)  # 计算欧式距离  计算余弦距离
        '''
        Tracker类是最核心的类，Tracker中保存了所有的轨迹信息，负责初始化第一帧的轨迹、卡尔曼滤波的预测和更新、负责级联匹配、IOU匹配等等核心工作。
        '''
        self.tracker = Tracker(
            metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)
    # bbox_xywh目标检测的边界框信息
    # conf目标检测的置信度
    # classes是类别
    # ori_img是输入的图像
    def update(self, bbox_xywh, confidences, classes, ori_img, use_yolo_preds=False):

        self.height, self.width = ori_img.shape[:2]  # 获得图像尺寸
        # generate detections
        features = self._get_features(bbox_xywh, ori_img)  # 获得目标检测后的特征图
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)  # [cx,cy,w,h]-->[x1,y1,w,h]
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(
            confidences)]


        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections, classes)

        # output bbox identities
        outputs = []
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            if use_yolo_preds:
                det = track.get_yolo_pred()
                x1, y1, x2, y2 = self._tlwh_to_xyxy(det.tlwh)
            else:
                box = track.to_tlwh()
                x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            class_id = track.class_id
            outputs.append(np.array([x1, y1, x2, y2, track_id, class_id], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)
        return outputs

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), self.height - 1)
        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    def _get_features(self, bbox_xywh, ori_img):
        # 0 ---------------->x
        # |  (x1,y1)----------
        # |     |            |
        # |     |            |
        # |     |            |
        # y     ------------(x2,y2)
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)  # 获得box的左上和右下坐标
            im = ori_img[y1:y2, x1:x2]  # 将检测的目标和背景进行分离
            im_crops.append(im)  # 抠出的目标放入列表
        if im_crops:  # 如果不为空，即目标检测有检测目标
            features = self.extractor(im_crops) # 进行特征提取
        else:  # 如果是空的，即没有检测到目标，为空
            features = np.array([])
        return features
