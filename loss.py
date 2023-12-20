
import torch
import torch.nn as nn
from func import intersection_over_union

class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20, lambda_noobj=0.5, lambda_coord=5):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S, self.B, self.C = S, B, C
        self.lambda_noobj, self.lambda_coord = lambda_noobj, lambda_coord

    def forward(self, predictions, target):
        predictions = predictions.view(-1, self.S, self.S, self.C + self.B * 5)

        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)

        iou_maxes, bestbox = torch.max(ious, dim=0)
        exists_box = target[..., 20].unsqueeze(3)

        # BOX COORDINATES
        box_predictions = exists_box * (bestbox * predictions[..., 26:30] + (1 - bestbox) * predictions[..., 21:25])
        box_targets = exists_box * target[..., 21:25]

        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(torch.abs(box_predictions[..., 2:4] + 1e-6))
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(box_predictions.view(-1), box_targets.view(-1))

        # OBJECT LOSS
        pred_box = bestbox * predictions[..., 25:26] + (1 - bestbox) * predictions[..., 20:21]
        object_loss = self.mse(exists_box * pred_box, exists_box * target[..., 20:21])

        # NO OBJECT LOSS
        no_object_loss = self.mse((1 - exists_box) * predictions[..., 20:21], (1 - exists_box) * target[..., 20:21])
        no_object_loss += self.mse((1 - exists_box) * predictions[..., 25:26], (1 - exists_box) * target[..., 20:21])

        # CLASS LOSS
        class_loss = self.mse(exists_box * predictions[..., :20], exists_box * target[..., :20])

        # TOTAL LOSS
        loss = self.lambda_coord * box_loss + object_loss + self.lambda_noobj * no_object_loss + class_loss

        return loss