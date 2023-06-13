from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torchvision.models.detection.roi_heads import RoIHeads
from torch import nn, Tensor
from torchvision.ops import boxes as box_ops

class FastRCNNPredictorWithAuthor(nn.Module):
  """
  Custom FasterRCNNPredictor with another branch that supports the author classification.
  """
  def __init__(self, in_channels, num_classes, num_authors):
    super().__init__()
    self.cls_score = nn.Linear(in_channels, num_classes)
    self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
    self.cls_author = nn.Linear(in_channels, num_authors)

  def forward(self, x):
    if x.dim() == 4:
      torch._assert(
          list(x.shape[2:]) == [1, 1],
          f"x has the wrong shape, expecting the last two dimensions to be [1,1] instead of {list(x.shape[2:])}",
          )
    x = x.flatten(start_dim=1)
    scores = self.cls_score(x)
    bbox_deltas = self.bbox_pred(x)
    scores_authors = self.cls_author(x)

    return scores, bbox_deltas, scores_authors


def fastrcnn_loss_with_authors(class_logits, box_regression, labels, regression_targets, authors_logits, authors):
    # type: (Tensor, Tensor, List[Tensor], List[Tensor], List[Tensor], List[Tensor]) -> Tuple[Tensor, Tensor]
    """
    Computes the loss for Faster R-CNN.

    Args:
        class_logits (Tensor)
        box_regression (Tensor)
        labels (list[BoxList])
        regression_targets (Tensor)
        authors_logits (Tensor)
        authors (list[BoxList])
    Returns:
        classification_loss (Tensor)
        box_loss (Tensor)
        authors_loss (Tensor)
    """
    labels = torch.cat(labels, dim=0)
    authors = torch.cat(authors, dim=0)
    regression_targets = torch.cat(regression_targets, dim=0)

    classification_loss = F.cross_entropy(class_logits, labels)
    authors_loss = F.cross_entropy(authors_logits, authors)

    # get indices that correspond to the regression targets for
    # the corresponding ground truth labels, to be used with
    # advanced indexing
    sampled_pos_inds_subset = torch.where(labels > 0)[0]
    labels_pos = labels[sampled_pos_inds_subset]
    N, num_classes = class_logits.shape
    box_regression = box_regression.reshape(N, box_regression.size(-1) // 4, 4)

    box_loss = F.smooth_l1_loss(
        box_regression[sampled_pos_inds_subset, labels_pos],
        regression_targets[sampled_pos_inds_subset],
        beta=1 / 9,
        reduction="sum",
    )
    box_loss = box_loss / labels.numel()

    return classification_loss, box_loss, authors_loss


class CustomRoIHeads(RoIHeads):

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels, gt_authors):
        # type: (List[Tensor], List[Tensor], List[Tensor], List[Tensor], List[Tensor]) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        """
        Custom assign_targets_to_proposals from default RoIHeads created to support author classification.
        """
        matched_idxs = []
        labels = []
        authors = []
        for proposals_in_image, gt_boxes_in_image, gt_labels_in_image, gt_authors_in_image in zip(proposals, gt_boxes, gt_labels, gt_authors):

            if gt_boxes_in_image.numel() == 0:
                # Managing the background image
                device = proposals_in_image.device
                # Create tensor filled with zeros to represent the background image
                clamped_matched_idxs_in_image = torch.zeros(
                    (proposals_in_image.shape[0],), dtype=torch.int64, device=device
                )
                labels_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
                authors_in_image = torch.zeros((proposals_in_image.shape[0],), dtype=torch.int64, device=device)
            else:
                # Calculate match quality matrix containing the pairwise IoU values for every elements in ground truth boxes and proposals
                match_quality_matrix = box_ops.box_iou(gt_boxes_in_image, proposals_in_image) # MxN tensor : M ground truth elements and N predicted elements
                # Assign to each predicted box a ground truth element
                matched_idxs_in_image = self.proposal_matcher(match_quality_matrix) # an N tensor where N[i] is a matched gt in [0, M-1] or a negative value if the prediction i could not be matched
                # Clamp matched indexes to ensure that they are valid
                clamped_matched_idxs_in_image = matched_idxs_in_image.clamp(min=0)
                # Select labels and authors from ground truth corresponding to matched indexes
                labels_in_image = gt_labels_in_image[clamped_matched_idxs_in_image]
                labels_in_image = labels_in_image.to(dtype=torch.int64)

                authors_in_image = gt_authors_in_image[clamped_matched_idxs_in_image]
                authors_in_image = authors_in_image.to(dtype=torch.int64)

                # Label background (below the low threshold) matches in [0, low_threshold)
                bg_inds = matched_idxs_in_image == self.proposal_matcher.BELOW_LOW_THRESHOLD
                labels_in_image[bg_inds] = 0
                authors_in_image[bg_inds] = 0

                # Label ignore proposals (between low and high thresholds) matches in [low_threshold, high_threshold)
                ignore_inds = matched_idxs_in_image == self.proposal_matcher.BETWEEN_THRESHOLDS
                labels_in_image[ignore_inds] = -1  # -1 is ignored by sampler
                authors_in_image[ignore_inds] = -1

            matched_idxs.append(clamped_matched_idxs_in_image)
            labels.append(labels_in_image)
            authors.append(authors_in_image)

        return matched_idxs, labels, authors

    def select_training_samples(
        self,
        proposals,  # type: List[Tensor]
        targets,  # type: Optional[List[Dict[str, Tensor]]]
    ):
        """
        Custom select training samples from default RoIHeads created to support author classification.
        Args:
            proposals: List[Tensor]
            targets: List[Dict[str, Tensor]]
        Return:
            proposals: List[Tensor] : proposals
            matched_idxs: List[Tensor] : matching ground truth indices for each proposal
            labels: List[Tensor] : ground truth label for each proposal
            regression_targets: List[Tensor]
            authors: List[Tensor] : ground truth author for each proposal
        """
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]
        # perform validation check on targets
        self.check_targets(targets)
        if targets is None:
            raise ValueError("targets should not be None")
        # get data type and device of first proposal
        dtype = proposals[0].dtype
        device = proposals[0].device
        
        # get ground truth bboxes, labels and authors from targets
        gt_boxes = [t["boxes"].to(dtype) for t in targets]  # ground truth boxes
        gt_labels = [t["labels"] for t in targets]          # ground truth object labels
        gt_authors = [t["author"] for t in targets]         # ground truth authors labels

        # append ground-truth bboxes to proposals: (List[Tensor], List[Tensor]) -> List[Tensor]
        proposals = self.add_gt_proposals(proposals, gt_boxes)

        # get matching gt indices, labels and authors for each proposal using custom assign_targets_to_proposals
        matched_idxs, labels, authors = self.assign_targets_to_proposals(proposals, gt_boxes, gt_labels, gt_authors)
       
        # sample a fixed proportion of positive-negative proposals
        sampled_inds = self.subsample(labels)

        matched_gt_boxes = []
        num_images = len(proposals)

        for img_id in range(num_images):
            # the various ground truths are retrieved from the sampled samples
            img_sampled_inds = sampled_inds[img_id]
            proposals[img_id] = proposals[img_id][img_sampled_inds]
            labels[img_id] = labels[img_id][img_sampled_inds]
            authors[img_id] = authors[img_id][img_sampled_inds]
            matched_idxs[img_id] = matched_idxs[img_id][img_sampled_inds]

            gt_boxes_in_image = gt_boxes[img_id]
            # if there is no gt boxes in the image, create a dummy box
            if gt_boxes_in_image.numel() == 0:
                gt_boxes_in_image = torch.zeros((1, 4), dtype=dtype, device=device)
            
            matched_gt_boxes.append(gt_boxes_in_image[matched_idxs[img_id]])
        # encode the set of bounding boxes into the representation used for training the regressor
        regression_targets = self.box_coder.encode(matched_gt_boxes, proposals)
        return proposals, matched_idxs, labels, regression_targets, authors

    def postprocess_detections(
        self,
        class_logits,  # type: Tensor
        authors_logits, # type: Tensor
        box_regression,  # type: Tensor
        proposals,  # type: List[Tensor]
        image_shapes,  # type: List[Tuple[int, int]]
    ):
        """
        Custom postprocess_detections from default RoIHeads created to support author classification.
        """
        # type: (...) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]
        device = class_logits.device
        num_classes = class_logits.shape[-1] # Number of classes

        # get number of authors from authors_logits
        num_authors_classes = authors_logits.shape[-1] # Number of authors

        # get number of boxes in the image
        boxes_per_image = [boxes_in_image.shape[0] for boxes_in_image in proposals]

        # Decode the predicted bounding box regression values
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        pred_scores = F.softmax(class_logits, -1)
        pred_authors_scores = F.softmax(authors_logits, -1)
        # Split predictions into lists per image
        pred_boxes_list = pred_boxes.split(boxes_per_image, 0)
        pred_scores_list = pred_scores.split(boxes_per_image, 0)
        pred_authors_scores_list = pred_authors_scores.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        all_authors_labels = []
        all_authors_scores = []

        for boxes, scores, authors_scores, image_shape in zip(pred_boxes_list, pred_scores_list, pred_authors_scores_list, image_shapes):
            # Clip the bounding boxes so that they are contained within the boundaries of the image.
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape) # boxes: tensor of shape [N, 4]

            # Create labels for each prediction
            labels = torch.arange(num_classes, device=device)
            labels = labels.view(1, -1).expand_as(scores)
            # create one tensor of authors scores for each classes/bbox
            authors_scores = authors_scores.unsqueeze(1).repeat(1, num_classes, 1)
            
            # Remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores[:, 1:]
            labels = labels[:, 1:]
            authors_scores = authors_scores[:, 1:]
            # Batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.reshape(-1)
            labels = labels.reshape(-1)
            authors_scores = authors_scores.reshape(-1, num_authors_classes)
            # Remove low scoring boxes
            inds = torch.where(scores > self.score_thresh)[0]
            boxes, scores, labels, authors_scores = boxes[inds], scores[inds], labels[inds], authors_scores[inds]
            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels, authors_scores = boxes[keep], scores[keep], labels[keep], authors_scores[keep]

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions (num_detections_per_img)
            keep = keep[: self.detections_per_img]
            boxes, scores, labels, authors_scores = boxes[keep], scores[keep], labels[keep], authors_scores[keep]

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            # save the indices of the maximum scores as authors labels
            all_authors_labels.append(torch.add(torch.max(authors_scores, 1).indices, 1))
            all_authors_scores.append(torch.max(authors_scores, 1).values)

        return all_boxes, all_scores, all_labels, all_authors_labels, all_authors_scores

    def forward(
        self,
        features,
        proposals,
        image_shapes,
        targets=None
    ):
        '''
        Custom forward method created to support author classification.
        The section relating to the mask has been omitted as it is not necessary for this specific task.
        '''

        if targets is not None:
            # perform some validation check on targets
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                if not t["boxes"].dtype in floating_point_types:
                    raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
                if not t["labels"].dtype == torch.int64:
                    raise TypeError(f"target labels must of int64 type, instead got {t['labels'].dtype}")
                if not t["author"].dtype == torch.int64:
                    raise TypeError(f"target authors must of int64 type, instead got {t['authors'].dtype}")
                if self.has_keypoint():
                    if not t["keypoints"].dtype == torch.float32:
                        raise TypeError(f"target keypoints must of float type, instead got {t['keypoints'].dtype}")

        if self.training:
            # getting training samples based on proposals and targets
            proposals, matched_idxs, labels, regression_targets, authors = self.select_training_samples(proposals, targets)
        else:
            labels = None
            authors = None
            regression_targets = None
            matched_idxs = None
        # crop and resize feature maps in the location indicates by the bounding boxes
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        # get the output from box head (module that take the cropped feature map as input - FastRCNNConvFCHead in case of a ResNet50 or a default TwoMLPHead)
        box_features = self.box_head(box_features)
        # obtain class logits, authors logits and box regression from custom box predictor
        class_logits, box_regression, authors_logits = self.box_predictor(box_features)

        result: List[Dict[str, torch.Tensor]] = []
        losses = {}
        if self.training:
            if labels is None:
                raise ValueError("labels cannot be None")
            if authors is None:
                raise ValueError("authors cannot be None")
            if regression_targets is None:
                raise ValueError("regression_targets cannot be None")
            # calculate the losses
            loss_classifier, loss_box_reg, loss_authors = fastrcnn_loss_with_authors(class_logits, box_regression, labels, regression_targets, authors_logits, authors) # calculate the losses
            
            losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg, "loss_authors": loss_authors}
        else:
            # if the model is not in training mode, get bounding boxes predictions, class labels and confidence scores
            boxes, scores, labels, authors, authors_scores = self.postprocess_detections(class_logits, authors_logits, box_regression, proposals, image_shapes)
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    {
                        "boxes": boxes[i],
                        "labels": labels[i],
                        "scores": scores[i],
                        "authors": authors[i],
                        "authors_scores": authors_scores[i]
                    }
                )

        # return final detection outputs and the computed losses

        return result, losses