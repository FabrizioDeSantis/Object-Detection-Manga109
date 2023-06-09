from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch
import numpy as np
from pprint import pprint

def calculate_mAP(model, classes, device, val_loader, args):
    """
    This function calculate the mean Average Precision for the test set.
    Mean Average Precision from from torchmetrics.detection.mean_ap was used to calculate the metric.
    Predicted boxes and targets have to be in Pascal VOC format (xmin-top left, ymin-top left, xmax-bottom right, ymax-bottom right).
    Args:
        model: the pretrained model
        classes: List[str]
        device
        val_loader: DataLoader
        args
    """
    model.to(device).eval()
    # get detection threshold from args
    detection_threshold=args.detection_threshold
    # enable per-class metrics (set to False for performance improvement)
    metric = MeanAveragePrecision(class_metrics=True)
    # support lists that will contain the dictionaries with predictions and targets. Each dictionary corresponds to a single image
    predictions_avg = [] # predictions
    actual_avg = []      # targets

    with torch.no_grad():

        for i, data in enumerate(val_loader):

            images, targets = data

            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # generate outputs
            outputs = model(images)

            # load all detection to CPU for further operations
            outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
            targets = [{k: v.to('cpu') for k, v in t.items()} for t in targets]

            for (out, targ) in zip(outputs, targets):
                # support lists that will contain the coordinates of the bounding boxes of each image
                bounding_boxes_predicted_img = []
                bounding_boxes_target_img = []
                # support lists that will contain the labels (target and predicted) of each image
                labels_pred_img = []
                labels_target_img = []
                # support list that will contain the confidence scores of eache image
                scrs_img = []
                # carry further only if there are detected boxes
                if len(out["boxes"]) != 0:

                    boxes = out["boxes"].data.numpy()
                    scores = out["scores"].data.numpy()
                    # filtering according to the threshold
                    boxes = boxes[scores >= detection_threshold].astype(np.int32)
                    
                    # get all predicted and actual classes
                    predicted_classes = [classes[i] for i in out["labels"].cpu().numpy()]                    
                    actual_classes = [classes[i] for i in targ["labels"].cpu().numpy()]

                    actual_classes = targ["labels"].data.numpy()
                    target_boxes = targ["boxes"].data.numpy()
                    predicted_classes = out["labels"].data.numpy()
                    # filtering according to the detection threshold
                    predicted_classes = predicted_classes[scores >= detection_threshold].astype(np.int32)
                    scores = scores[scores >= detection_threshold].astype(np.float32)
                    # iterate over predicted boxes and target boxes and save all the necessary parameters
                    for i in range(0, len(boxes)):
                        bs = []
                        bs.append(boxes[i][0])
                        bs.append(boxes[i][1])
                        bs.append(boxes[i][2])
                        bs.append(boxes[i][3])
                        bounding_boxes_predicted_img.append(bs)
                        scrs_img.append(scores[i])
                        labels_pred_img.append(predicted_classes[i])

                    for i in range(0, len(target_boxes)):
                        bs = []
                        bs.append(target_boxes[i][0])
                        bs.append(target_boxes[i][1])
                        bs.append(target_boxes[i][2])
                        bs.append(target_boxes[i][3])
                        bounding_boxes_target_img.append(bs)
                        labels_target_img.append(actual_classes[i])
                # create dictionaries for current image
                predicted_metrics_dictionary = dict(boxes = torch.tensor(bounding_boxes_predicted_img),
                            scores = torch.tensor(scrs_img),
                            labels = torch.tensor(labels_pred_img)
                    )
                actual_metrics_dictionary = dict(boxes = torch.tensor(bounding_boxes_target_img),
                            labels = torch.tensor(labels_target_img)
                    )
                
                predictions_avg.append(predicted_metrics_dictionary)
                actual_avg.append(actual_metrics_dictionary)
    # calculate mean average precision
    metric.update(predictions_avg, actual_avg)
    # display results
    pprint(metric.compute())


def calculate_mAP_authors(model, classes, authors, device, val_loader, args):
    """
    This function calculate the mean Average Precision for the test set.
    This function is similar to the previous one but and implements the calculation of the metric also for the classification of the author.
    Mean Average Precision from from torchmetrics.detection.mean_ap was used to calculate the metric.
    Predicted boxes and targets have to be in Pascal VOC format (xmin-top left, ymin-top left, xmax-bottom right, ymax-bottom right).
    Args:
        model: the pretrained model
        classes: List[str]
        authors: List[int]
        device
        val_loader: DataLoader
        args
    """

    model.to(device).eval()
    # get detection threshold from args
    detection_threshold=args.detection_threshold
    # enable per-class metrics (set to False for performance improvement)
    metric = MeanAveragePrecision(class_metrics=True)
    # support lists that will contain the dictionaries with predictions and targets. Each dictionary corresponds to a single image
    predictions_avg = [] # predictions
    actual_avg = []      # targets
    predictions_authors_avg = [] # authors predictions
    actual_authors_avg = []      # target authors

    with torch.no_grad():
        
        for i, data in enumerate(val_loader):

            images, targets = data
            
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # generate outputs
            outputs = model(images)

            # load all detection to CPU for further operations
            outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
            targets = [{k: v.to('cpu') for k, v in t.items()} for t in targets]

            for (out, targ) in zip(outputs, targets):
                # support lists that will contain the coordinates of the bounding boxes of each image
                bounding_boxes_pred = []
                bounding_boxes_targ = []
                # support lists that will contain the labels (original classes and authors) of each image
                labels_pred = []
                authors_pred = []
                # support lists that will contain the labels (original classes and authors) of each image
                labels_target = []
                authors_target = []
                # support lists that will contain the confidence scores (original classes and authors) of each image
                scrs = []
                scrs_authors = []
                # carry further only if there are detected boxes
                if len(out["boxes"]) != 0:

                    boxes = out["boxes"].data.numpy()

                    scores = out["scores"].data.numpy()
                    authors_scores = out["authors_scores"].data.numpy()

                    # filtering according to the threshold
                    boxes = boxes[scores >= detection_threshold].astype(np.int32)
                    
                    # get all predicted classes
                    #predicted_classes = [classes[i] for i in out["labels"].cpu().numpy()]                    
                    #actual_classes = [classes[i] for i in targ["labels"].cpu().numpy()]
                    #actual_authors = [authors[i] for i in targ["author"].cpu().numpy()]

                    actual_classes = targ["labels"].data.numpy()
                    actual_authors = targ["author"].data.numpy()
                    target_boxes = targ["boxes"].data.numpy()

                    predicted_classes = out["labels"].data.numpy()
                    predicted_authors = out["authors"].data.numpy()

                    support_list = predicted_classes.copy()
                    
                    predicted_classes = predicted_classes[scores >= detection_threshold].astype(np.int32)
                    predicted_authors = predicted_authors[scores >= detection_threshold].astype(np.int32)
                    authors_scores = authors_scores[scores >= detection_threshold].astype(np.float32)

                    #predicted_authors = [predicted_authors[i] for i in range(len(support_list)) if scores[i]>=detection_threshold]
                    #authors_scores = [authors_scores[i] for i in range(len(support_list)) if scores[i]>=detection_threshold]

                    scores = scores[scores >= detection_threshold].astype(np.float32)    

                    # iterate over predicted boxes and target boxes and save all the necessary parameters
                    for i in range(0, len(boxes)):
                        bs = []  
                        bs.append(boxes[i][0])
                        bs.append(boxes[i][1])
                        bs.append(boxes[i][2])
                        bs.append(boxes[i][3])
                        bounding_boxes_pred.append(bs)
                        scrs.append(scores[i])
                        scrs_authors.append(authors_scores[i])
                        labels_pred.append(predicted_classes[i])
                        authors_pred.append(predicted_authors[i])

                    for i in range(0, len(target_boxes)):
                        bs = []
                        bs.append(target_boxes[i][0])
                        bs.append(target_boxes[i][1])
                        bs.append(target_boxes[i][2])
                        bs.append(target_boxes[i][3])
                        bounding_boxes_targ.append(bs)
                        labels_target.append(actual_classes[i])
                        authors_target.append(actual_authors[i])
                # create dictionaries for current image
                predicted_metrics_dictionary = dict(boxes = torch.tensor(bounding_boxes_pred),
                            scores = torch.tensor(scrs),
                            labels = torch.tensor(labels_pred)
                    )
                    
                predicted_metrics_authors_dictionary = dict(boxes = torch.tensor(bounding_boxes_pred),
                            scores = torch.tensor(scrs_authors),
                            labels = torch.tensor(authors_pred)
                    )
                    
                actual_metrics_dictionary = dict(boxes = torch.tensor(bounding_boxes_targ),
                            labels = torch.tensor(labels_target)
                    )
                    
                actual_metrics_authors_dictionary = dict(boxes = torch.tensor(bounding_boxes_targ),
                            labels = torch.tensor(authors_target)
                    )
                    
                predictions_avg.append(predicted_metrics_dictionary)
                actual_avg.append(actual_metrics_dictionary)
                ####################################################
                predictions_authors_avg.append(predicted_metrics_authors_dictionary)
                actual_authors_avg.append(actual_metrics_authors_dictionary)
    # calculate mean average precision for original classes
    metric.update(predictions_avg, actual_avg)
    # display results for original classes
    pprint(metric.compute())
    # calculate mean average precision for authors
    metric.update(predictions_authors_avg, actual_authors_avg)
    # display results for authors 
    pprint(metric.compute())