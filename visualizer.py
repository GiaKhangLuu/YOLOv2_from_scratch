import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

BOX_COLOR = (0, 0, 255)
TEXT_COLOR = (0, 0, 255)

def plot_img(img, size=(7,7)):
    plt.figure(figsize=size)
    plt.imshow(img)
    plt.show()
    
# vẽ bounding box lên ảnh
def visualize_bbox(img, boxes, labels=None, thickness=2, color=BOX_COLOR):
    img_copy = img.cpu().permute(1,2,0).numpy().copy() if isinstance(img, torch.Tensor) else img.copy()
    boxes = boxes.numpy() if isinstance(boxes, torch.Tensor) else boxes
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        img_copy = cv2.rectangle(img_copy, (x1, y1),(x2, y2),
                                 color, thickness)
        
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        img_copy = cv2.circle(img_copy, center=center, radius=3, color=(0,255,0), thickness=2)

        if labels is not None:
            img_copy = cv2.putText(img_copy, labels[i], (x1, y1 - 10), 
                                    0, 1, TEXT_COLOR)

    return img_copy