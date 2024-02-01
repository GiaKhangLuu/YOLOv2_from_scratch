import torch

def xyxy2xywh(bboxes):
    """
    Convert list of (xmin, ymin, xmax, ymax) to (x_center, y_center, box_width, box_height)
    Params:
        bboxes: List of bounding boxes (xmin, ymin, xmax, ymax)
    Return:
        boxes: List of bounding boxes (x_center, y_center, box_width, box_height)
    """
    boxes = list()
    for box in bboxes:
        xmin, ymin, xmax, ymax = box

        # Compute width and height of box
        box_width = xmax - xmin
        box_height = ymax - ymin

        # Compute x, y center
        x_center = int(xmin + (box_width / 2))
        y_center = int(ymin + (box_height / 2))

        boxes.append((x_center, y_center, box_width, box_height))

    return boxes

def x1y1wh2xyxy(boxes):
    """
    Convert bounding boxes from [x1, y1, width, height] to [x1, y1, x2, y2].
    Args:
        boxes (torch.Tensor): Bounding boxes tensor of shape (N, 4).

    Returns:
        torch.Tensor: Converted bounding boxes tensor of shape (N, 4).
    """
    x1, y1, w, h = boxes.unbind(dim=-1)
    x2 = x1 + w 
    y2 = y1 + h 
    return torch.stack((x1, y1, x2, y2), dim=-1).to(torch.int)

def target_tensor_to_boxes(boxes_tensor, num_grid=13, 
                           img_w=416, img_h=416,
                           num_anchor_boxes=5,
                           output_thresh=0.7):
    '''
    Recover target tensor (tensor output of dataset) to bboxes
    Params:
        boxes_tensor: bboxes in tensor format - output of dataset.__getitem__
    Return:
        boxes: list of box, each box is [x1,y1,w,h]
    '''
    cell_w, cell_h = img_w / num_grid, img_h / num_grid
    boxes, obj_probs, cls_probs = [], [], []
    for i in range(num_grid):
        for j in range(num_grid):
            for b in range(num_anchor_boxes):
                data = boxes_tensor[i, j, b]
                x_center, y_center, w, h, obj_prob, cls_prob = data[0], data[1], data[2], data[3], data[4], data[5:]
                prob = obj_prob * max(cls_prob)
                if prob > output_thresh:
                    x1, y1 = x_center + j - w / 2, y_center + i - h / 2
                    x1, y1, w, h = x1 * cell_w, y1 * cell_h, w * cell_w, h * cell_h
                    box = [x1, y1, w, h]
                    boxes.append(box)
                    obj_probs.append(prob)
                    cls_probs.append(cls_prob.numpy().tolist())
    return torch.tensor(boxes), torch.tensor(obj_probs), torch.tensor(cls_probs)

from tqdm import tqdm

def output_tensor_to_boxes(boxes_tensor, anchor_boxes, grid_size=13, 
                           img_w=416, img_h=416, num_anchor_boxes=5,
                           output_thres=0.7, num_max_boxes=10):
    cell_w, cell_h = img_w / grid_size, img_h / grid_size
    boxes = []
    probs = []

    if isinstance(anchor_boxes, list):
        anchor_boxes = torch.tensor(anchor_boxes)

    for i in range(grid_size):
        for j in range(grid_size):
            for b in range(num_anchor_boxes):
                anchor_wh = anchor_boxes[b].clone().detach()
                pred = boxes_tensor[i, j, b]
                xy_delta = torch.sigmoid(pred[:2])
                wh = torch.exp(pred[2:4]) * anchor_wh
                obj_prob = torch.sigmoid(pred[4])
                cls_prob = torch.softmax(pred[5:], dim=-1)
                combine_prob = obj_prob * max(cls_prob)
                
                if combine_prob > output_thres:
                    x_center, y_center, w, h = xy_delta[0], xy_delta[1], wh[0], wh[1]
                    x1, y1 = x_center + j - w / 2, y_center + i - h / 2
                    x1, y1, w, h = x1 * cell_w, y1 * cell_h, w * cell_w, h * cell_h
                    box = [x1, y1, w, h, combine_prob]
                    boxes.append(box)

    #boxes = boxes[:num_max_boxes] if len(boxes) > num_max_boxes else boxes

    return boxes


def overlap(interval_1, interval_2):
    x1, x2 = interval_1
    x3, x4 = interval_2
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2,x4) - x3

def compute_iou(box1, box2):
    """Compute IOU between box1 and box2"""
    x1,y1,w1,h1 = box1[0], box1[1], box1[2], box1[3]
    x2,y2,w2,h2 = box2[0], box2[1], box2[2], box2[3]
    
    ## if box2 is inside box1
    if (x1 < x2) and (y1<y2) and (w1>w2) and (h1>h2):
        return 1
    
    area1, area2 = w1*h1, w2*h2
    intersect_w = overlap((x1,x1+w1), (x2,x2+w2))
    intersect_h = overlap((y1,y1+h1), (y2,y2+w2))
    intersect_area = intersect_w*intersect_h
    iou = intersect_area/(area1 + area2 - intersect_area)
    return iou

def nonmax_suppression(boxes, IOU_THRESH = 0.4):
    """remove ovelap bboxes"""
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    for i, current_box in enumerate(boxes):
        if current_box[4] <= 0:
            continue
        for j in range(i+1, len(boxes)):
            iou = compute_iou(current_box, boxes[j])
            if iou > IOU_THRESH:
                boxes[j][4] = 0
    boxes = [box for box in boxes if box[4] > 0]
    return boxes
