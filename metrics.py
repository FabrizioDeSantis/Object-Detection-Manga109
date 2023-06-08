from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch
import numpy as np
from pprint import pprint

def calculate_mAP(model, classes, device, val_loader, args):

    model.to(device).eval()

    detection_threshold=args.detection_threshold
    
    metric = MeanAveragePrecision(class_metrics=True)

    predictions_avg = []
    actual_avg = []

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
                
                bounding_boxes_predicted_img = []
                bounding_boxes_target_img = []
                
                labels_pred_img = []
                labels_target_img = []
                
                scrs_img = []
                
                if len(out["boxes"]) != 0:

                    boxes = out["boxes"].data.numpy()

                    scores = out["scores"].data.numpy()

                    # filtering according to the threshold
                    boxes = boxes[scores >= detection_threshold].astype(np.int32)
                    
                    # get all predicted classes
                    predicted_classes = [classes[i] for i in out["labels"].cpu().numpy()]                    
                    actual_classes = [classes[i] for i in targ["labels"].cpu().numpy()]

                    actual_classes = targ["labels"].data.numpy()
                    target_boxes = targ["boxes"].data.numpy()

                    predicted_classes = out["labels"].data.numpy()
                    
                    predicted_classes = predicted_classes[scores >= detection_threshold].astype(np.int32)

                    scores = scores[scores >= detection_threshold].astype(np.float32)

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

                predicted_metrics_dictionary = dict(boxes = torch.tensor(bounding_boxes_predicted_img),
                            scores = torch.tensor(scrs_img),
                            labels = torch.tensor(labels_pred_img)
                    )
                actual_metrics_dictionary = dict(boxes = torch.tensor(bounding_boxes_target_img),
                            labels = torch.tensor(labels_target_img)
                    )
                    
                predictions_avg.append(predicted_metrics_dictionary)
                actual_avg.append(actual_metrics_dictionary)

    metric.update(predictions_avg, actual_avg)

    pprint(metric.compute())


def calculate_mAP_authors(model, classes, authors, device, val_loader):

    model.to(device).eval()

    # threshold
    detection_threshold=0.50

    # metric = MeanAveragePrecision(class_metrics=True)
    metric = MeanAveragePrecision(class_metrics=True)

    predictions_avg = []
    actual_avg = []
    ##############
    predictions_authors_avg = []
    actual_authors_avg = []

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

                bounding_boxes_pred = []
                bounding_boxes_targ = []
                
                labels_pred = []
                authors_pred = []
                
                scrs = []
                scrs_authors = []
                
                labels_target = []
                authors_target = []

                if len(out["boxes"]) != 0:

                    boxes = out["boxes"].data.numpy()

                    scores = out["scores"].data.numpy()
                    authors_scores = out["authors_scores"].data.numpy()

                    # filtering according to the threshold
                    boxes = boxes[scores >= detection_threshold].astype(np.int32)
                    
                    # get all predicted classes
                    predicted_classes = [classes[i] for i in out["labels"].cpu().numpy()]                    
                    actual_classes = [classes[i] for i in targ["labels"].cpu().numpy()]
                    actual_authors = [authors[i] for i in targ["author"].cpu().numpy()]

                    actual_classes = targ["labels"].data.numpy()
                    actual_authors = targ["author"].data.numpy()
                    target_boxes = targ["boxes"].data.numpy()

                    predicted_classes = out["labels"].data.numpy()
                    predicted_authors = out["authors"].data.numpy()

                    support_list = predicted_classes.copy()
                    
                    predicted_classes = predicted_classes[scores >= detection_threshold].astype(np.int32)

                    predicted_authors = [predicted_authors[i] for i in range(len(support_list)) if scores[i]>=detection_threshold]
                    authors_scores = [authors_scores[i] for i in range(len(support_list)) if scores[i]>=detection_threshold]

                    scores = scores[scores >= detection_threshold].astype(np.float32)    

                    id = targ["image_id"].data.numpy()

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
                        tmp=[]
                        bs = []  
                        tmp.append(id[0])
                        tmp.append(actual_classes[i])
                        tmp.append(target_boxes[i][0])
                        tmp.append(target_boxes[i][1])
                        tmp.append(target_boxes[i][2])
                        tmp.append(target_boxes[i][3])
                        bs.append(target_boxes[i][0])
                        bs.append(target_boxes[i][1])
                        bs.append(target_boxes[i][2])
                        bs.append(target_boxes[i][3])
                        bounding_boxes_targ.append(bs)
                        labels_target.append(actual_classes[i])
                        authors_target.append(actual_authors[i])

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

    metric.update(predictions_avg, actual_avg)

    pprint(metric.compute())

    metric.update(predictions_authors_avg, actual_authors_avg)

    pprint(metric.compute())