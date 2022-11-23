import numpy as np
import torch
import cv2 as cv 
import albumentations as A
from albumentations.pytorch import ToTensorV2
import utils 
from universal_dataloader import dataset, visualize

IMAGE_SIZE = 416
P_1 = 0.5
P_2 = 0.1
scale = 1.0
 
transform = A.Compose(
    [
        A.LongestMaxSize(max_size=int(IMAGE_SIZE * scale)), # LongestMaxSize will resize the image so that the longest side is equal to max_size
        A.PadIfNeeded(
            min_height=int(IMAGE_SIZE * scale),
            min_width=int(IMAGE_SIZE * scale),
            border_mode=cv.BORDER_CONSTANT,
        ),  
        # A.RandomCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
        # A.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.6, p=P_1),
        # A.OneOf(
        #     [
        #         A.ShiftScaleRotate(rotate_limit=20, p=P_1, border_mode=cv.BORDER_CONSTANT),
        #         A.Affine(shear=15, p=P_1),
        #     ],
        #     p=P_1,
        # ),
        # A.HorizontalFlip(p=P_1),
        # A.Blur(p=P_2),
        # A.CLAHE(p=P_2),
        # A.Posterize(p=P_2),
        # A.ToGray(p=P_2),
        # A.ChannelShuffle(p=P_2/2),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255),
        ToTensorV2(always_apply=True, p=1.0),
    ],
    bbox_params=A.BboxParams(format="yolo", min_visibility=0.4, label_fields=[],),
)

'''
Overall model structure:
    1. There will be 3 output heads 
    2. Each head will have 3 anchors, or 3 outputs [batch, 3, S, S, 6]

Looking at a single head:
    1. We take the detection box, and find the anchor that has the highest IOU
    2. Now for that paticular anchor, we add box into grid matrix & only for that anchor. 

Function: 
    For each box:
        For each head:
            For each anchor:                                                                    #
                - Check IOU with box                                                            #
                    - If MAX IOU (from all anchors), add box to grid matrix                     #
                - If IoU > 0.5, add -1 to the grid matrix                                       # Ignore prediction. Essentially we are saying that, if box already allocated to anchor, but the overlap with another anchor is > 0.5, write -1 so we don't 'punish this anchor' in loss. 
'''
def box_to_anchors(boxes, classes, anchors, SA=[13, 16, 52], ignore_iou_thresh = 0.5):
    num_anchors = anchors.shape[0]
    num_anchors_per_scale = num_anchors // 3
    
    targets = [torch.zeros((num_anchors // 3, S, S, 6)) for S in SA]
    for i, box in enumerate(boxes):
        iou_anchors = utils.iou_width_height(torch.tensor(box[2:4]), anchors)
        anchor_indices = iou_anchors.argsort(descending=True, dim=0)
        x, y, width, height = box
        class_label = classes[i]
        head_has_anchor = [False] * 3
        
        for anchor_idx in anchor_indices:
            scale_idx = torch.div(anchor_idx, num_anchors_per_scale, rounding_mode='trunc')            
            anchor_on_scale = anchor_idx % num_anchors_per_scale
            S = SA[scale_idx]
            i, j = int(S * y), int(S * x)  # which cell
            
            # should be 'cell_taken', essentially if anchor 
            cell_taken = targets[scale_idx][anchor_on_scale, i, j, 0]
            # print(scale_idx, cell_taken, head_has_anchor[scale_idx], iou_anchors[anchor_idx])
            
            if not cell_taken and not head_has_anchor[scale_idx]:
                targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                x_cell, y_cell = S * x - j, S * y - i  # both between [0,1]
                width_cell, height_cell = (
                    width * S,
                    height * S,
                )  # can be greater than 1 since it's relative to cell
                box_coordinates = torch.tensor(
                    [x_cell, y_cell, width_cell, height_cell]
                )
                targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                head_has_anchor[scale_idx] = True
            elif not cell_taken and iou_anchors[anchor_idx] > ignore_iou_thresh:
                targets[scale_idx][anchor_on_scale, i, j, 0] = -1  # ignore prediction
    return targets


## TESTING ##
def play():    
    t = torch.randint(size=(3, 3, 3), high=255)
    t_0 = t[0]
    print(t_0/255)
    
    norm = A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255)
    tnorm = norm(image=t.numpy())['image'][0]
    print(tnorm)
    
def main():
    viz = visualize.Visualize()
    # data = dataset.DataSet("/home/server/Desktop/data/", train=True)
    # imgA, boxes, _, classes = data.__getitem__(0)
    # imgA = torch.tensor(imgA)
    # print(type(imgA))

    for i in range(10):
        data = dataset.DataSet("/home/server/Desktop/data/", train=True, transform=transform)
        imgB, boxes, _, classes = data.__getitem__(i)
        img2 = viz.draw_box(imgB, boxes=boxes, format="midpoints")
        cv.imwrite('imgs/img'+str(i)+'.png', viz.pil_to_cv(img2))
        print(boxes, type(boxes))
        
    # print("type:", type(imgB), "shape:", imgB.shape, "sum:", imgB.sum(), "mean:", imgB.mean(), "std:", imgB.std(), "max:", imgB.max(), "min:", imgB.min())
    # img1 = viz.draw_box(imgA, boxes=boxes, format="midpoints")
    # cv.imwrite('imgs/inp.png', viz.pil_to_cv(img1))
    # print("img1.shape: ", img1.size, "type: ",type(img1))
    # print("img2.shape: ", img2.size, "type: ",type(img2))
    # combined_img = viz.stack_imgs([img1, img2])

if __name__ == "__main__":
    # play()
    main()