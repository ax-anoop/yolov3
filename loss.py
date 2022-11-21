# intersection over union 

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