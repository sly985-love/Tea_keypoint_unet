import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import torch
import random
# import onnx
#import onnxruntime as rt
import time
import torchvision
import argparse
import os


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 7680  # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), l[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


def toTensor(img):
    assert type(img) == np.ndarray, 'the img type is {}, but ndarry expected'.format(type(img))
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img.transpose((2, 0, 1)))
    #img = torch.from_numpy(img)
    return img.float().div(255).unsqueeze(0)  # 255也可以改为256


def get_peak_points(heatmaps):
    """

    :param heatmaps: numpy array (N,4,256,256)
    :return:numpy array (N,4,2) #
    """
    N,C,H,W = heatmaps.shape   # N= batch size C=4 hotmaps
    all_peak_points = []
    for i in range(N):
        peak_points = []
        for j in range(C):
            yy,xx = np.where(heatmaps[i, j] == heatmaps[i, j].max())
            y = yy[0]
            x = xx[0]
            peak_points.append([x, y])
        all_peak_points.append(peak_points)
    all_peak_points = np.array(all_peak_points)
    return all_peak_points


def keypoint_det(cv2img, img, model, p1, transforms_kp):
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    H, W = img.shape[:2]
    img_tensor = transforms_kp(img)
    img_tensor = img_tensor * 255
    img_tensor = img_tensor.unsqueeze(dim=0).to("cuda:0")
    pred_heatmaps = model(img_tensor)

    pred_points = get_peak_points(pred_heatmaps.cpu().data.numpy())  # (N,4,2)
    pred_points = pred_points.reshape((pred_points.shape[0], -1))  # (N,8)

    point_0 = [int(round(pred_points[0][0] * (W / 256))), int(round(pred_points[0][1] * (H / 256)))]
    point_1 = [int(round(pred_points[0][2] * (W / 256))), int(round(pred_points[0][3] * (H / 256)))]
    point_2 = [int(round(pred_points[0][4] * (W / 256))), int(round(pred_points[0][5] * (H / 256)))]
    point_3 = [int(round(pred_points[0][6] * (W / 256))), int(round(pred_points[0][7] * (H / 256)))]
    point_4 = [int(round(pred_points[0][8] * (W / 256))), int(round(pred_points[0][9] * (H / 256)))]
    point_5 = [int(round(pred_points[0][10] * (W / 256))), int(round(pred_points[0][11] * (H / 256)))]

    point_0_new = [point_0[0] + p1[0], point_0[1] + p1[1]]
    point_1_new = [point_1[0] + p1[0], point_1[1] + p1[1]]
    point_2_new = [point_2[0] + p1[0], point_2[1] + p1[1]]
    point_3_new = [point_3[0] + p1[0], point_3[1] + p1[1]]
    point_4_new = [point_4[0] + p1[0], point_4[1] + p1[1]]
    point_5_new = [point_5[0] + p1[0], point_5[1] + p1[1]]

    cv2.circle(cv2img, (point_0_new[0], point_0_new[1]), 8, (255, 0, 255), -1)
    cv2.circle(cv2img, (point_1_new[0], point_1_new[1]), 8, (255, 0, 255), -1)
    cv2.circle(cv2img, (point_2_new[0], point_2_new[1]), 8, (255, 0, 255), -1)
    cv2.circle(cv2img, (point_3_new[0], point_3_new[1]), 8, (255, 0, 255), -1)
    cv2.circle(cv2img, (point_4_new[0], point_4_new[1]), 8, (255, 0, 255), -1)
    cv2.circle(cv2img, (point_5_new[0], point_5_new[1]), 8, (255, 0, 255), -1)


if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    transforms_kp = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    torch_script_model = torch.jit.load(r"weights/best.torchscript")
    torch_script_model.eval()

    model_kp = torch.jit.load("weights/U_net_460.pt")
    model_kp.eval()

    save_path = "output"
    os.makedirs(save_path, exist_ok=True)

    img_path = "data/src_picture"
    img_list = sorted(os.listdir(img_path))

    for img_ in img_list:
        img_abs_path = img_path + "/{}".format(img_)
        cv2img = cv2.imread(img_abs_path)
        h, w = cv2img.shape[:2]
        resize_ = [640, 640]

        img = letterbox(cv2img)[0]
        img = cv2.resize(img, (resize_[1], resize_[0]))
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        im = torch.from_numpy(img).to(device)
        # im = im.half() if half else im.float()  # uint8 to fp16/32
        im = im.float()
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        torch_script_model_inference = torch_script_model(im)[0]
        torchScript_output = non_max_suppression(torch_script_model_inference, conf_thres=0.25, iou_thres=0.45, agnostic=False)

        for i, det in enumerate(torchScript_output):  # detections per image
            for d in range(det.shape[0]):
                x1, y1, x2, y2 = det[d, 0].cpu().numpy() * (w / resize_[1]), det[d, 1].cpu().numpy() * (h / resize_[0]), det[d, 2].cpu().numpy() * (w / resize_[1]), det[d, 3].cpu().numpy() * (h / resize_[0])
                x1, y1, x2, y2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
                # p1, p2 = (x1, y1), (x2, y2)
                p1, p2 = (x1 - 10, y1 - 10), (x2 + 10, y2 + 10)
                conf, cls = det[d, 4].cpu().numpy(), int(det[d, 5].cpu().numpy())
                # cv2.rectangle(cv2img, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.rectangle(cv2img, (x1 - 10, y1 - 10), (x2 + 10, y2 + 10), (255, 0, 255), 2)

                label = f'{"tea"} {conf:.2f}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                p22 = p1[0] + t_size[0], p1[1] - t_size[1] - 3
                cv2.rectangle(cv2img, p1, p22, (255, 0, 255), -1, cv2.LINE_AA)  # filled
                cv2.putText(cv2img, label, (p1[0], p1[1] - 2), 0, 1, [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)

                # Key Point ============================================================================================
                try:
                    cropped_tea_leaf = cv2img[p1[1]:p2[1], p1[0]:p2[0]]
                    keypoint_det(cv2img, cropped_tea_leaf, model_kp, p1, transforms_kp)
                except Exception as Error:
                    print(Error)

        # cv2img_resized = cv2.resize(cv2img, (1920, 1080))
        # cv2.imshow("cv2img", cv2img)
        # cv2.waitKey(0)
        cv2.imwrite("{}/{}".format(save_path, img_), cv2img)
        print("Saving to --> {}/{}".format(save_path, img_))


