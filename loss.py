import torch

ANCHOR_BOXS = [[1.08, 1.19],
               [3.42, 4.41],
               [6.63, 11.38],
               [9.42, 5.11],
               [16.62, 10.52]]

def post_process_output(output, anchor_boxes=ANCHOR_BOXS):
    """Convert output of model to pred_xywh"""
    # xy
    xy = torch.sigmoid(output[..., :2] + 1e-6)

    # wh
    wh = output[..., 2:4]

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    anchors_wh = torch.Tensor(anchor_boxes).view(1, 1, 1, len(ANCHOR_BOXS), 2).to(device)
    #anchors_wh = anchor_boxes.view(1, 1, 1, anchor_boxes.shape[0], 2)

    wh = torch.exp(wh) * anchors_wh
    
    # objectness confidence
    obj_prob = torch.sigmoid(output[..., 4:5] + 1e-6)
    
    # class distribution
    cls_dist = torch.softmax(output[..., 5:], dim=-1)
    return xy, wh, obj_prob, cls_dist

def post_process_target(target_tensor):
    """
    Tách target tensor thành từng thành phần riêng biệt: xy, wh, object_probility, class_distribution
    """
    xy = target_tensor[..., :2]
    wh = target_tensor[..., 2:4]
    obj_prob = target_tensor[..., 4:5]
    cls_dist = target_tensor[..., 5:]
    return xy, wh, obj_prob, cls_dist

def square_error(output, target):
    return (output - target) ** 2

def yolo_loss(pred_tensor, gt_tensor, anchor_boxes):
    """
    Luồng xử lí:
        1. Tính diện tích các pred_bbox
        2. Tính diện tích các true_bbox
        3. Tính iou giữa từng pred_bbox với true_bbox tương ứng (nằm trong cùng 1 cell)
        4. Trong mỗi cell, xác định best_box - box có iou với true_bbox đạt giá trị max so với 4 pred_bbox còn lại
        5. Tính các loss thành phần theo công thức trong ảnh
        6. Tính Total_loss
    """

    coord_weight, no_obj_weight, obj_weight = 5, .5, 12.

    assert pred_tensor.shape == gt_tensor.shape
    
    pred_xy, pred_wh, pred_obj_confs, pred_cls_dist = post_process_output(pred_tensor, anchor_boxes)
    gt_xy, gt_wh, gt_obj_confs, gt_cls_dist = post_process_target(gt_tensor)

    # Compute predictions' areas
    pred_x1y1 = pred_xy - .5 * pred_wh  # (B, H, W, num_boxes, 2)
    pred_x2y2 = pred_xy + .5 * pred_wh  # (B, H, W, num_boxes, 2)
    area_pred = pred_wh[..., 0] * pred_wh[..., 1]  # (B, H, W, num_boxes)

    # Compute gts' areas
    gt_x1y1 = gt_xy - .5 * gt_wh  # (B, H, W, num_boxes, 2)
    gt_x2y2 = gt_xy + .5 * gt_wh  # (B, H, W, num_boxes, 2)
    area_gt = gt_xy[..., 0] * gt_xy[..., 1]  # (B, H, W, num_boxes)

    intersection_x1y1 = torch.max(pred_x1y1, gt_x1y1)  # (B, H, w, num_boxes, 2)
    intersection_x2y2 = torch.min(pred_x2y2, gt_x2y2)  # (B, H, W, num_boxes, 2)
    intersection_wh = torch.clamp(intersection_x2y2 - intersection_x1y1, 0)  # (B, H, W, num_boxes, 2)
    intersection_area = intersection_wh[..., 0] * intersection_wh[..., 1]  # (B, H, W, num_boxes)

    ious = intersection_area / (area_pred + area_gt - intersection_area)  # (B, H, W, num_boxes)
    max_iou = torch.max(ious, dim=-1, keepdim=True)[0]  # (B, H, W, num_boxes)
    best_iou_index = torch.unsqueeze(torch.eq(ious, max_iou).float(), dim=-1)  # (B, H, W, num_boxes, 1)
    mask = best_iou_index * gt_obj_confs  # (B, H, W, num_boxes, 1)

    center_loss = torch.sum(square_error(gt_xy, pred_xy) * mask) * 5

    # Square root to decrease value of wh, because we use torch.exp(wh) -> This will cause NAN
    # if it is too large
    coord_loss  = torch.sum(square_error(torch.sqrt(gt_wh), 
                                         torch.sqrt(pred_wh)) * mask) * 5

    # If the prediction doesn't show anything, maybe the object_confidence is too small, 
    # try to increase the weight of obj_loss, this way aims to focus to the object_confidence 
    obj_loss = torch.sum(square_error(gt_obj_confs, pred_obj_confs) * mask) * obj_weight

    # If the prediction shows a lot of center points in non_object_cell, ty to increase the 
    # weight of no_obj_weight
    no_obj_loss = torch.sum(square_error(1 - gt_obj_confs, 1 - pred_obj_confs)) * no_obj_weight

    cls_loss = torch.sum(square_error(gt_cls_dist, pred_cls_dist) * mask) 

    total_loss = coord_loss + center_loss + obj_loss + no_obj_loss + cls_loss
    return total_loss