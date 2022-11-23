import torch 
import torch.nn as nn
from utils import intersection_over_union

'''
1. nn.MSELoss 
2. nn.BCEWithLogitsLoss 
3. nn.BCE or nn.CrossEntropyLoss #'we don't have multi label loss here'

lambda_class = 1
lambda_noobj = 10
lambda_obj = 1
lamda_box = 10

def forward(predictions, target, anchors)
    > target is already in transformed 'yolo' form
    
1. No object loss 
    - BCE or no objects 
2. Object loss 
    - Set it to the IOU of where objects present 
    - BCE (predictions, targets * ious)
3. Box Coordinate loss 
    -  mse(predictions[obj], target[obj])
4. Class loss
    - simple entrophy between class where obj exists
'''

def find_nonzero(output):
    # head0, anchor0; row = 8, column = 5 ; Has data.
    dim = 13
    head = output[0]
    for k in range(3):
        for i in range(dim):
            if head[k][i].sum() != 0:
                for j in range(dim):
                    if head[k][i][j].sum() != 0:
                        print("Anchor:",k, "i:", i, "j:", j, head[k][i][j])
    
class YoloLoss(nn.Module):
    def __init__(self, anchors, S, scaled_anchors):
        super().__init__()
        # Various loss functions needed 
        self.mse = nn.MSELoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.entropy = nn.CrossEntropyLoss()
        # Constants signifying how much to pay for each respective part of the loss
        self.lambda_class = 1
        self.lambda_noobj = 10
        self.lambda_obj = 1
        self.lambda_box = 10
        # Other
        self.sigmoid = nn.Sigmoid()
        self.anchors = anchors
        self.scaled_anchors = scaled_anchors
        self.S = S
  
    '''
    target = [1.0, x, y, w, h, class]
    predic = [obj, x, y, w, h, ...classes]
        - ...classes = [0,0,0,1,0 ...]
    '''
    def forward(self, predictions, target, head=0):
        # find_nonzero(target)
    
        # where is obj and noobj (we ignore if target == -1), like masks that can be passed to grab only these values (true & falses)
        obj = target[..., 0] == 1 
        noobj = target[..., 0] == 0
        anchors = self.scaled_anchors[head].reshape(1, 3, 1, 1, 2)
        
        ### no obj loss ###
        no_object_loss = self.bce(
            (predictions[..., 0:1][noobj]), (target[..., 0:1][noobj]),
        )
        
        # ### obj loss ###
        box_preds = torch.cat([self.sigmoid(predictions[..., 1:3]), torch.exp(predictions[..., 3:5]) * self.anchors[head]], dim=-1)
        ious = intersection_over_union(box_preds[obj], target[..., 1:5][obj]).detach()
        object_loss = self.mse(self.sigmoid(predictions[..., 0:1][obj]), ious * target[..., 0:1][obj])

        # ### box cord loss ###
        predictions[..., 1:3] = self.sigmoid(predictions[..., 1:3])  # x,y coordinates
        target[..., 3:5] = torch.log(
            (1e-16 + target[..., 3:5] / anchors)
        )  # width, height coordinates
        box_loss = self.mse(predictions[..., 1:5][obj], target[..., 1:5][obj])

        # ### class loss ###
        class_loss = self.entropy(
            (predictions[..., 5:][obj]), (target[..., 5][obj].long()),
        )
        return(self.lambda_box * box_loss + self.lambda_obj * object_loss + self.lambda_noobj * no_object_loss + self.lambda_class * class_loss)