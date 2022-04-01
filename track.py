# limit the number of cpus used by high performance libraries
import os

from IPython import embed


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
sys.path.insert(0, './yolov4')

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
import numpy as np

from yolov4.utils.general import xyxy2xywh, increment_path
from yolov4.utils.torch_utils import select_device
from yolov4.utils.plots import Annotator, colors
from yolov4.yolo import YOLO
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov4 deepsort root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def detect(opt):
    out, source, deep_sort_model, show_vid, save_vid, classes, project, name, exist_ok= \
        opt.output, opt.source, opt.deep_sort_model, opt.show_vid, opt.save_vid, \
        opt.classes, opt.project, opt.name, opt.exist_ok

    # -----------------------读取类名-------------------------------------
    names = []
    file = open(r'./yolov4/model_data/coco_classes.txt', encoding='utf-8')
    lines = file.readlines()
    for i in lines:
        names.append(i)
    # -------------------------------------------------------------------

    device = select_device(opt.device)  # cpu or gpu
    # initialize deepsort
    cfg = get_config()
    cfg.merge_from_file(opt.config_deepsort)  # 读取deep_sort/configs/deep_sort.yaml配置文件
    deepsort = DeepSort(deep_sort_model,  # deep_sort_model模型初始化  默认osnet_x0_25
                        device,  # cpu or  gpu
                        max_dist=cfg.DEEPSORT.MAX_DIST,  # 距离阈值
                        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,  # iou阈值
                        max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,  # 周期
                        )

    # Directories
    # project默认路径：runs/track/   name路径：exp,所以整体路径为：runs/track/exp
    # exist_ok是判断输出路径是否存在，不存在则会创建
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run  保存输出检测结果
    save_dir.mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    model = YOLO()

    if source == '0':
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    save_path = str(save_dir / "test.avi")
    out = cv2.VideoWriter(save_path, fourcc, 30, size)
    ret, frame = cap.read()
    if not ret:
        raise ValueError("未能正确读取摄像头（视频），请注意是否正确安装摄像头（是否正确填写视频路径）。")
    while True:
                ret, frame = cap.read()
                if not ret:
                    break
                    # 格式转变，BGRtoRGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 转变成Image
                frame = Image.fromarray(np.uint8(frame))
                # 进行检测
                det, top_boxes = model.detect_image(frame, classes)
                res = torch.FloatTensor(np.array(det[0]))
                confs = res[:, 4]
                clss = res[:, -1]
                im0 = np.array(frame)
                annotator = Annotator(im0, line_width=2, pil=not ascii)
                # 将yolov4输出的box信息，置信度，类别，原图输入到跟踪网络
                xywhs = xyxy2xywh(top_boxes)  # 获得box信息
                outputs = deepsort.update(xywhs, confs, clss, im0)
                # draw boxes for visualization
                if len(outputs) > 0:
                    for j, (output, conf) in enumerate(zip(outputs, confs)): # conf type:tensor

                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]  # 当为person的时候cls=0

                        c = int(cls)  # integer class

                        label = '{} {}'.format(id, names[c])
                        annotator.box_label(bboxes, label, color=colors(c, True))

                if show_vid:
                    im0 = cv2.cvtColor(np.asarray(im0), cv2.COLOR_RGB2BGR)
                    cv2.namedWindow("yolov4_deepsort",0)
                    cv2.imshow("yolov4_deepsort", im0)
                    if cv2.waitKey(1) == ord('q'):  # q to quit
                        raise StopIteration

                if save_vid:
                    out.write(im0)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--deep_sort_model', type=str, default='osnet_x0_25')
    parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='avi', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 16 17')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument("--config_deepsort", type=str, default="deep_sort/configs/deep_sort.yaml")
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detection per image')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()


    with torch.no_grad():
        detect(opt)
