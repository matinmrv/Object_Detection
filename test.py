import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
from func import (
        non_max_suppression,
        mean_average_precision,
        plot_image,
        cellboxes_to_boxes,
        get_bboxes,
        load_checkpoint,
)

seed = 90
EPOCHS = 1000
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0
NUM_WORKERS = 2
BATCH_SIZE = 16
DEVICE = "cpu"
PIN_MEMORY = True
LOAD_MODEL_FILE = "/home/matin/Documents/Yolo/overfit.pth.tar"
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes

transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])
def main():

    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    train_dataset = VOCDataset("100examples.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR)
    test_dataset = VOCDataset("test.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR)
    
    common_loader_config = {
            "batch_size": BATCH_SIZE,
            "num_workers": NUM_WORKERS,
            "pin_memory": PIN_MEMORY,
            "shuffle": True,
            "drop_last": True,
        }

    train_loader = DataLoader(dataset=train_dataset, **common_loader_config)
    test_loader = DataLoader(dataset=test_dataset, **common_loader_config)

    for epoch in range(EPOCHS):
        for x, y in test_loader:
            for idx in range(8):
                x = x.to(DEVICE)
                bboxes = cellboxes_to_boxes(model(x))
                bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4,
                        box_format="midpoint")
                plot_image(x[idx].permute(1, 2, 0).to("cpu"), bboxes)

if __name__ == "__main__":
    main()
