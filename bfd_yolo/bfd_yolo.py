from ultralytics import YOLO
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss, BboxLoss
from ultralytics.utils.metrics import bbox_iou
import torch

class BFD_YOLO(YOLO):
    @property
    def task_map(self):
        task_map = super().task_map
        task_map["detect"]["model"] = BFD_DetectionModel
        #task_map["detect"]["trainer"] =  BFD_Trainer
        return task_map

'''
class BFD_Trainer(DetectionTrainer):
    def compute_loss(self, preds, targets, **kwargs):
        loss = focal_siou_loss(preds, targets)
        return loss
'''

class BFD_DetectionModel(DetectionModel):
    def init_criterion(self):
        """Initialize the loss criterion for the DetectionModel."""
        return E2EDetectLoss(self) if getattr(self, "end2end", False) else BFD_DetectionLoss(self) #v8DetectionLoss(self)

class BFD_DetectionLoss(v8DetectionLoss):
    def __init__(self, model, tal_topk: int = 10):
        super().__init__(model, tal_topk)

        self.bbox_loss = BFD_BboxLoss(self.reg_max).to(self.device)

class BFD_BboxLoss(BboxLoss):
    """Criterion class for computing training losses for bounding boxes."""

    def __init__(self, reg_max: int = 16, tau: float = 2.0):
        """Initialize the BboxLoss module with regularization maximum and DFL settings."""
        super().__init__(reg_max)
        self.tau = tau

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute IoU and DFL losses for bounding boxes."""
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        iou, siou_loss = FocalSIoU_bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False)# compute here

        focal_loss_iou = ((1.0 - iou)**tau * (-torch.log(iou)) * siou_loss)
        focal_loss_iou = (focal_loss_iou * weight).sum() / target_scores_sum

        #loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return focal_loss_iou, loss_dfl

def FocalSIoU_bbox_iou(
    box1: torch.Tensor,
    box2: torch.Tensor,
    xywh: bool = True,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Calculate the Intersection over Union (IoU) between bounding boxes. And SIoU loss.
    """
    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (
        b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)
    ).clamp_(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    # Convex box
    cw = torch.maximum(b1_x2, b2_x2) - torch.minimum(b1_x1, b2_x1)
    ch = torch.maximum(b1_y2, b2_y2) - torch.minimum(b1_y1, b2_y1)
    # Center distance
    s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5
    s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5
    sigma = torch.sqrt(s_cw ** 2 + s_ch ** 2) + eps

    sin_alpha_1 = torch.abs(s_cw) / sigma
    sin_alpha_2 = torch.abs(s_ch) / sigma
    threshold = math.sqrt(2) / 2###
    sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
    
    # Angle cost
    angle_cost = 1 - 2 * torch.pow(torch.sin(torch.arcsin(sin_alpha) - math.pi / 4), 2)###

    # Distance cost
    rho_x = (s_cw / (cw + eps)) ** 2#**2?
    rho_y = (s_ch / (ch + eps)) ** 2#**2?
    gamma = 2 - angle_cost
    distance_cost = 2 - torch.exp(-gamma * rho_x) - torch.exp(-gamma * rho_y)

    # Shape cost
    omiga_w = torch.abs(w1 - w2) / torch.maximum(w1, w2)
    omiga_h = torch.abs(h1 - h2) / torch.maximum(h1, h2)
    shape_cost = torch.pow(1 - torch.exp(-omiga_w), 4) + torch.pow(1 - torch.exp(-omiga_h), 4)

    # Return IoU & SIoU loss
    return iou, 1 - (iou + 0.5 * (distance_cost + shape_cost))