import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import transforms, loss as loss_contructor, utils, model as model_maker
from universal_dataloader import dataset, visualize
from legacy import model as legacy_model

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 1
IMAGE_SIZE = 416
PIN_MEMORY = True
SHUFFLE = True
S = [IMAGE_SIZE // 32, IMAGE_SIZE // 16, IMAGE_SIZE // 8]

ANCHORS = [ [(0.28, 0.22), (0.38, 0.48), (0.9, 0.78)], [(0.07, 0.15), (0.15, 0.11), (0.14, 0.29)], [(0.02, 0.03), (0.04, 0.07), (0.08, 0.06)]]
SCALED_ANCHORS = (
    torch.tensor(ANCHORS)
    * torch.tensor(S).unsqueeze(1).unsqueeze(1).repeat(1, 3, 2)
    ).to(DEVICE)
ANCHORS = torch.tensor(ANCHORS[0] + ANCHORS[1] + ANCHORS[2]).to(DEVICE)

def load_data():
    train_data = dataset.DataSet(
        "/home/server/Desktop/data/", 
        train=True, 
        transform=transforms.transform, 
        box_transform=transforms.box_to_anchors, 
        box_transform_args = {"anchors": ANCHORS},
    )
    train_eval_loader = DataLoader(
        dataset=train_data,
        batch_size= BATCH_SIZE,
        # num_workers= NUM_WORKERS,
        pin_memory= PIN_MEMORY,
        shuffle=SHUFFLE,
        drop_last=False,
    )
    return train_eval_loader

def test_train():
    # Define the model, loss and data loader
    model = legacy_model.YOLOv3(num_classes=20).to(DEVICE)
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = loss_contructor.YoloLoss(anchors=ANCHORS, scaled_anchors=SCALED_ANCHORS, S=S)
    train_loader = load_data()
    
    # print(type(train_loader))
    # loop = tqdm(train_loader, leave=True)
    
    while(1):
        img, y = next(iter(train_loader))
        img = img.to(DEVICE)
        
        out = model(img)
        loss = loss_fn(out[0].to(DEVICE), y[0].to(DEVICE))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(loss.item())
        
    # Test our loss function
    # print(img.shape)
    # outputs = model_leg(img.to(DEVICE))
    # yolo_loss(outputs, targets.to(DEVICE))

def main():
    test_train()
    
if __name__ == '__main__':
    main()
    # head0, anchor0; row = 8, column = 5 ; Has data.
    # print(head0[0][7][4])
    # print(head0[1][7][4])
    # head0, anchor0; row = 8, column = 5 ; Has data.
    # heads = [head0, head1, head2]
    # SS, c = [13, 16, 52], 0
    # for head in heads:
    #     for k in range(3):
    #         for i in range(SS[c]):
    #             if head[k][i].sum() != 0:
    #                 for j in range(SS[c]):
    #                     if head[k][i][j].sum() != 0:
    #                         print("head:",c,"Anchor:",k, "i:", i, "j:", j)
    #     c+=1