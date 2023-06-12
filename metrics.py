from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch
import numpy as np
from utils import save_predictions_on_tb
from pprint import pprint

def calculate_mAP(model, classes, device, val_loader, args, writer, draw_prediction):
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

            for out in outputs:
              boxes = out["boxes"].data.numpy()
              scores = out["scores"].data.numpy()
              # get all predicted classes
              predicted_classes = out["labels"].data.numpy()
              # filtering according to the threshold
              boxes = boxes[scores >= detection_threshold].astype(np.int32)
              # filtering according to the detection threshold
              predicted_classes = predicted_classes[scores >= detection_threshold].astype(np.int32)
              scores = scores[scores >= detection_threshold].astype(np.float32)
              # create dictionary for current image
              predicted_metrics_dictionary = dict(boxes = torch.tensor(boxes),
                            scores = torch.tensor(scores),
                            labels = torch.tensor(predicted_classes)
              )
                
              predictions_avg.append(predicted_metrics_dictionary)

            for targ in targets:
              # create dictionary for current image
              actual_metrics_dictionary = dict(boxes = targ["boxes"],
                            labels = targ["labels"]
              )
              actual_avg.append(actual_metrics_dictionary)
            
            if draw_prediction and i==0:
                """
                save first predicted image of val loader on Tensorboard
                """
                pc = [classes[i] for i in out[0]["labels"].cpu().numpy()] # get names of predicted classes
                ac = [classes[i] for i in targ[0]["labels"].cpu().numpy()]
                save_predictions_on_tb(targets[0]["boxes"], outputs[0]["boxes"], pc, ac, images[0], writer)

    # calculate mean average precision
    metric.update(predictions_avg, actual_avg)
    # display results
    pprint(metric.compute())


def calculate_mAP_authors(model, device, val_loader, args):
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
        
        for _, data in enumerate(val_loader):

            images, targets = data
            
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # generate outputs
            outputs = model(images)

            # load all detection to CPU for further operations
            outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
            targets = [{k: v.to('cpu') for k, v in t.items()} for t in targets]
            # iterate over predictions
            for out in outputs:
              boxes = out["boxes"].data.numpy()
              scores = out["scores"].data.numpy()
              authors_scores = out["authors_scores"].data.numpy()
              # get all predicted classes
              predicted_classes = out["labels"].data.numpy()
              predicted_authors = out["authors"].data.numpy()

              # filtering according to detection threshold
              boxes = boxes[scores >= detection_threshold].astype(np.int32)
              predicted_classes = predicted_classes[scores >= detection_threshold].astype(np.int32)
              predicted_authors = predicted_authors[scores >= detection_threshold].astype(np.int32)
              authors_scores = authors_scores[scores >= detection_threshold].astype(np.float32)
              scores = scores[scores >= detection_threshold].astype(np.float32)    

              # create dictionaries for current image
              predicted_metrics_dictionary = dict(boxes = torch.tensor(boxes),
                            scores = torch.tensor(scores),
                            labels = torch.tensor(predicted_classes)
              )
                    
              predicted_metrics_authors_dictionary = dict(boxes = torch.tensor(boxes),
                            scores = torch.tensor(scores),
                            labels = torch.tensor(predicted_authors)
              )
              predictions_avg.append(predicted_metrics_dictionary)
              predictions_authors_avg.append(predicted_metrics_authors_dictionary)
            # iterate over targets
            for targ in targets:
                    
              actual_metrics_dictionary = dict(boxes = targ["boxes"],
                            labels = targ["labels"]
              )
                    
              actual_metrics_authors_dictionary = dict(boxes = targ["boxes"],
                            labels = targ["author"]
              )
              actual_avg.append(actual_metrics_dictionary)
              actual_authors_avg.append(actual_metrics_authors_dictionary)
    # calculate mean average precision for original classes
    metric.update(predictions_avg, actual_avg)
    # display results for original classes
    pprint(metric.compute())
    # calculate mean average precision for authors
    metric.update(predictions_authors_avg, actual_authors_avg)
    # display results for authors 
    pprint(metric.compute())