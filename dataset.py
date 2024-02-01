import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensor, ToTensorV2
from utils import xyxy2xywh
import torch
from torch.utils.data import Dataset
from xml.etree import ElementTree
import random

class FruitDataset(Dataset):
    def __init__(self, data_file, cls_names, 
                 desired_width=416, desired_height=416, grid_size=13,
                 num_anchor_boxes=9):
        img_files, annot_files = self._get_img_and_annot_files(data_file)
        self.data = self._read_data(img_files, annot_files)  
        self.cls_names = cls_names
        self.grid_size = grid_size
        self.num_anchor_boxes = num_anchor_boxes
        self.desired_height = desired_height
        self.desired_width = desired_width

        self.transforms = A.Compose([
                A.LongestMaxSize(max_size=(desired_height, desired_width), p=1),
                A.PadIfNeeded(min_height=desired_height, min_width=desired_width, 
                              p=1, border_mode=cv2.BORDER_CONSTANT, value=0.0),
                ToTensorV2(p=1.0)
            ],
            bbox_params={
                "format": "pascal_voc",
                'label_fields': ['labels']
        })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_data = self.data[idx]
        img_fn = img_data['file_path']
        boxes = img_data["boxes"]
        obj_names = img_data["obj_names"]
        labels = np.zeros((len(obj_names), len(self.cls_names)), dtype=np.int32)

        for i, cls_name in enumerate(obj_names):
            cls_ids = self.cls_names.index(cls_name)
            labels[i, cls_ids] = 1
        img = cv2.cvtColor(cv2.imread(img_fn).astype(np.float32), cv2.COLOR_BGR2RGB) / 255.0

        try:
            if self.transforms:
                sample = self.transforms(**{
                    "image": img,
                    "bboxes": boxes,
                    "labels": labels,
                })
                img = sample['image']
                boxes = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
        except:
            return self.__getitem__(random.randint(0, len(self.data) - 1))

        target_tensor = self.boxes_to_tensor(boxes.type(torch.float32), labels)

        return img, target_tensor
    
    def _get_img_and_annot_files(self, file_dir):
        img_files = [os.path.join(file_dir, img_file) for img_file
                    in sorted(os.listdir(file_dir)) if img_file[-4:] == '.jpg']
        annot_files = [img_file[:-4] + '.xml' for img_file in img_files]

        return img_files, annot_files
    
    def _read_data(self, img_files, annot_files):
        """
        Read img from .jpg and annot from .xml 
        Return:
            list of image_data
            each element is dict {file_path: str, 
                                  boxes: list(x, y, w, h), 
                                  obj_names: list(str),
                                  img_w: int,
                                  img_h: int}
        """

        data = []
        for img_file, annot_file in zip(img_files, annot_files):
            img_data = dict()
            img_data['file_path'] = img_file
            boxes, obj_names, img_w, img_h = self.extract_annotation_file(annot_file)
            img_data['boxes'] = boxes
            img_data['obj_names'] = obj_names
            #img_data['width'], img_data['height'] = img_w, img_h

            assert len(img_data['boxes']) > 0
            assert len(img_data['boxes']) == len(img_data['obj_names'])

            data.append(img_data)

        return data

    def extract_annotation_file(self, filename):
        """
        Extract bounding boxes from an annotation file
        Params:
            filename: Annotation file name
        Returns:
            boxes: List of bounding boxes in image (x, y, x, y)
            obj_cls: List of classes in image
            img_width: Width of image
            img_height: Height of image
        """

        # Load and parse the file
        tree = ElementTree.parse(filename)
        # Get the root of the document
        root = tree.getroot()
        boxes = list()
        classes = list()

        # Extract each bounding box
        for box in root.findall('.//object'):
            cls = box.find('name').text
            xmin = int(box.find('bndbox/xmin').text)
            ymin = int(box.find('bndbox/ymin').text)
            xmax = int(box.find('bndbox/xmax').text)
            ymax = int(box.find('bndbox/ymax').text)
            xyxy = (xmin, ymin, xmax, ymax)
            boxes.append(xyxy)
            classes.append(cls)

        #boxes = xyxy2xywh(boxes)

        # Get width and height of an image
        img_w = int(root.find('.//size/width').text)
        img_h = int(root.find('.//size/height').text)

        # Some annotation files have set width and height by 0,
        # so we need to load image and get it width and height
        if (img_w == 0) or (img_h == 0):
            img = cv2.imread(filename[:-4] + '.jpg')
            img_h, img_w, _ = img.shape

        return boxes, classes, img_w, img_h


    def boxes_to_tensor(self, bboxes, one_hot_label):
        """
        Convert list of boxes (xyxy) (and labels) to tensor format
        Return:
            boxes_tensor: shape = (Batchsize, S, S, num_boxes, (4 + 1 + num_cls))
        """

        num_classes = len(self.cls_names)
        boxes_tensor = torch.zeros((self.grid_size, self.grid_size, 
                                    self.num_anchor_boxes, 5 + num_classes))
        cell_w = self.desired_width / self.grid_size
        cell_h = self.desired_height / self.grid_size

        for i, box in enumerate(bboxes):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1

            # Normalize xywh by cell_size
            x, y, w, h = x1 / cell_w, y1 / cell_h, w / cell_w, h / cell_h
            x_center, y_center = x + (w / 2), y + (h / 2)

            # Determine grid cell of center point (x_center, y_center)
            # TODO:
            # Maybe change this logic later. With this way, one cell only contains
            # single object
            grid_x, grid_y = int(np.floor(x_center)), int(np.floor(y_center))

            if grid_x < self.grid_size and grid_y < self.grid_size:
                offset_values = [x_center - grid_x, y_center - grid_y, w, h]
                boxes_tensor[grid_y, grid_x, :, :4] = torch.tensor(
                    [offset_values for _ in range(self.num_anchor_boxes)])
                boxes_tensor[grid_y, grid_x, :, 4] = torch.tensor([1.] * self.num_anchor_boxes)
                boxes_tensor[grid_y, grid_x, :, 5:] = torch.tensor(
                    np.array([one_hot_label[i, :] for _ in range(self.num_anchor_boxes)]))
        
        return boxes_tensor 

                

