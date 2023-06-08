import manga109api_custom
from pprint import pprint
from PIL import Image, ImageDraw
import torch
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

import roi_heads_custom

import argparse

argParser = argparse.ArgumentParser()
argParser.add_argument("-lr", "--learning_rate", type = float, nargs = '?', const = 1, default = 0.005, help = "Learning rate parameter for optimizer. Default is: 0.005")
argParser.add_argument("-bs", "--batch_size", type = int, nargs = '?', const = 1, default = 4, help = "Batch size parameter. Default is: 4")
argParser.add_argument("-add_auth", "--add_authors", type = int, nargs = '?', const = 1, default = 1, help = "1 if you want to include authors, 0 otherwise. Default is: 0")
argParser.add_argument("-res", "--resize_to", type = int, nargs = '?', const = 1, default = 512, help = "Resize dimensions of the input images. Default is: 512")
argParser.add_argument("-ne", "--num_epochs", type = int, nargs = '?', const = 1, default = 20, help = "Number of epochs for training (Early stopping is implemented). Default is: 20")
argParser.add_argument("-fn", "--file_name", type = str, nargs = '?', const = 1, default = "model", help = "Name of the file where the trained model will stored (is not needed to insert .pt extension)")
argParser.add_argument("-model_name", "--model_name", type = str, nargs = '?', const = 1, default="fasterrcnn", help = "Name of the model. Available models are: FasterRCNN (fasterrcnn), RetinaNet (retinanet). Default is fasterrcnn")
argParser.add_argument("-bb", "--backbone", type = str, nargs = '?', const = 1, default = "resnet50", help = "Name of the backbone for a FasterRCNN model. Available backbones are: resnet50, resnet50v2, mobilenet. Default is resnet50")
args = argParser.parse_args()

if args.model_name=="ssd":
  RESIZE_TO=300
else:
  RESIZE_TO=args.resize_to
BATCH_SIZE=args.batch_size
NUM_EPOCHS=args.num_epochs
LEARNING_RATE=args.learning_rate
NUM_WORKERS=0
PATH="model/"+args.file_name+".pt"

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# training images and XML files directory
# validation images and XML files directory

CLASSES = ["__background__", "body", "face", "frame", "text"]

NUM_CLASSES=len(CLASSES)

print("---- TRAINING PARAMETERS ----")
print("Batch size: " + str(BATCH_SIZE))
print("Resize: " + str(RESIZE_TO))
print("Model: " + args.model_name)
print("Backbone: " + args.backbone)
print("Learning rate: " + str(LEARNING_RATE))
print("Epochs: " + str(NUM_EPOCHS))
print("Authors: " + ("Included" if args.add_authors else "Not included"))
print("Classes: " + str(NUM_CLASSES))
print("-----------------------------")

import re

file_path = "autori2.txt"
data = pd.read_csv(file_path, sep="\t", header=None)
data.columns = ["id", "author", "title"]
ids = data["id"].tolist()
authors = data["author"].tolist()
titles = data["title"].tolist()

AUTHORS = authors.copy()
AUTHORS.insert(0, "background")

authors_list = []

for id, author, title in zip(ids, authors, titles):
  temp = []
  temp.append(id)
  temp.append(author)
  temp.append(title)
  authors_list.append(temp)

manga109_root_dir="Manga109/Manga109_released_2021_12_30"
p = manga109api_custom.Parser(root_dir=manga109_root_dir, authors_list=authors_list)

def load_all_images(train_images, author_labels):
  for book in p.books:
    annotation, author = p.get_annotation(book=book)
    for i in range(0, len(annotation["page"])):
      temp=[]
      if not(len(annotation["page"][i]["frame"])==0 and len(annotation["page"][i]["face"])==0 and len(annotation["page"][i]["body"])==0 and len(annotation["page"][i]["text"])==0):
        path=p.img_path(book=book, index=i)
        temp.append(path)
        temp.append(annotation["page"][i])
        train_images.append(temp)
        author_labels.append(author)

images=[]
authors_labels = []
load_all_images(images, authors_labels)

train_images, val_images, y_train, y_test = train_test_split(images, authors_labels, shuffle=True, stratify=authors_labels, test_size=0.2, random_state = 33)

# import random
# random.shuffle(images)

# train_images, val_images = np.split(images, [int(len(images)*0.8)])

df_train = pd.DataFrame(train_images, columns=["path", "annotation"])
df_val = pd.DataFrame(val_images, columns=["path", "annotation"])

# we will need custom transforms since each transforms need to be applied on the bounding box too

from torch.utils.data import DataLoader
from datasetManga109 import CustomDataset, get_train_transform, get_valid_transform

train_dataset = CustomDataset(df_train, RESIZE_TO, RESIZE_TO, CLASSES, get_train_transform())
val_dataset = CustomDataset(df_val, RESIZE_TO, RESIZE_TO, CLASSES, get_valid_transform())

def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch)) 

print(f"Number of training images: {len(train_dataset)}")
print(f"Number of validation images: {len(val_dataset)}")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    collate_fn=collate_fn
)

# Define the Faster RCNN model
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import math
import torch.nn as nn

def create_model(n_classes):
  if args.model_name == "fasterrcnn":
    if args.backbone == "resnet50":
      model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    elif args.backbone == "resnet50v2":
      model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=True)
    elif args.backbone == "mobilenet":
      model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(pretrained=True)
    model = create_fasterrcnn(n_classes, model)
  elif args.model_name == "retinanet":
    model = create_retinanet(n_classes)
  return model

class FastRCNNPredictorWithAuthor(nn.Module):
  
  def __init__(self, in_channels, num_classes, num_authors):
    super().__init__()
    self.cls_score = nn.Linear(in_channels, num_classes)
    self.bbox_pred = nn.Linear(in_channels, num_classes * 4)
    self.cls_author = nn.Linear(in_channels, num_authors+1)

  def forward(self, x):
    if x.dim() == 4:
      torch._assert(
          list(x.shape[2:]) == [1, 1],
          f"x has the wrong shape, expecting the last two dimensions to be [1,1] instead of {list(x.shape[2:])}",
          )
    x = x.flatten(start_dim=1)
    scores = self.cls_score(x)
    bbox_deltas = self.bbox_pred(x)
    prediction_author = self.cls_author(x)

    return scores, bbox_deltas, prediction_author

from torchvision.models.detection.ssd import SSDClassificationHead

def create_fasterrcnn(n_classes, model):
  # get the number of input features
  in_features = model.roi_heads.box_predictor.cls_score.in_features
  if args.add_authors:
    from torchvision.models.detection import roi_heads
    roi_heads.fastrcnn_loss = roi_heads_custom.fastrcnn_loss_with_authors
    model.roi_heads.box_predictor = FastRCNNPredictorWithAuthor(in_features, 5, len(authors_list))

    model.roi_heads.assign_targets_to_proposals = roi_heads_custom.myAssign_targets_to_proposals.__get__(model.roi_heads)
    model.roi_heads.select_training_samples = roi_heads_custom.mySelect_training_samples.__get__(model.roi_heads)
    model.roi_heads.postprocess_detections = roi_heads_custom.myPostprocess_detections.__get__(model.roi_heads)
    model.roi_heads.forward = roi_heads_custom.forward.__get__(model.roi_heads, model.roi_heads.__class__)
  else:
  # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, n_classes)
  return model

def create_retinanet(n_classes):
  model = torchvision.models.detection.retinanet_resnet50_fpn(pretrained=True)
  num_anchors = model.head.classification_head.num_anchors
  model.head.classification_head.num_classes = n_classes

  cls_logits = torch.nn.Conv2d(256, num_anchors * n_classes, kernel_size = 3, stride=1, padding=1)
  torch.nn.init.normal_(cls_logits.weight, std=0.01)  # as per pytorch code
  torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))  # as per pytorcch code 
  # assign cls head to model
  model.head.classification_head.cls_logits = cls_logits

  return model

model = create_model(n_classes=NUM_CLASSES)

model = model.to(DEVICE)

# get the model parameters
params = [p for p in model.parameters() if p.requires_grad]
# define the optimizer
optimizer = torch.optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
# optimizer = torch.optim.Adam(params, lr=0.003, weight_decay=0.0005)

# define the train and test function
from tqdm import tqdm

writer = SummaryWriter('./runs/faster-rcnn-experiment')

def train(train_data_loader, model):

  print("Training...")
  train_iter = 0
  train_losses = []

  progress_bar = tqdm(train_data_loader, total=len(train_data_loader))

  for i, data in enumerate(progress_bar):
    
    optimizer.zero_grad()

    images, targets = data
    images = list(image.to(DEVICE) for image in images)
    targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
    
    loss_dict = model(images, targets) # return the loss

    loss_classifier = loss_dict["loss_classifier"]
    # loss_box_reg = loss_dict["loss_box_reg"]
    # loss_objectness = loss_dict["loss_objectness"]
    # loss_rpn_box_reg = loss_dict["loss_rpn_box_reg"]

    losses = sum(loss for loss in loss_dict.values())
    loss_value = losses.item()
    train_losses.append(loss_value)
    losses.backward()
    optimizer.step()
    train_iter+=1

    if i % 100 == 99:
      writer.add_scalar("epoch_avg_train_loss", np.average(train_losses), epoch*len(train_data_loader) + i)
      writer.add_scalar("epoch_avg_classifier_train_loss", loss_classifier, epoch*len(train_data_loader) + i)

    progress_bar.set_description(desc=f"Loss: {loss_value:.4f}, Loss classifier: {loss_classifier:.4f}")

  return train_losses

def train_with_author(train_data_loader, model):

  print("Training...")
  train_iter = 0
  train_losses = []

  progress_bar = tqdm(train_data_loader, total=len(train_data_loader))

  for i, data in enumerate(progress_bar):
    
    optimizer.zero_grad()

    images, targets = data
    images = list(image.to(DEVICE) for image in images)
    targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
    
    loss_dict = model(images, targets) # return the loss

    loss_classifier = loss_dict["loss_classifier"]
    # loss_box_reg = loss_dict["loss_box_reg"]
    # loss_objectness = loss_dict["loss_objectness"]
    # loss_rpn_box_reg = loss_dict["loss_rpn_box_reg"]

    losses = sum(loss for loss in loss_dict.values())
    author_loss = loss_dict["loss_authors"]
    loss_value = losses.item()
    train_losses.append(loss_value)
    losses.backward()
    optimizer.step()
    train_iter+=1

    if i % 100 == 99:
      writer.add_scalar("epoch_avg_train_loss", np.average(train_losses), epoch*len(train_data_loader) + i)
      writer.add_scalar("epoch_avg_author_train_loss", author_loss, epoch*len(train_data_loader) + i)
      writer.add_scalar("epoch_avg_classifier_train_loss", loss_classifier, epoch*len(train_data_loader) + i)

    progress_bar.set_description(desc=f"Loss: {loss_value:.4f}, Loss classifier: {loss_classifier:.4f}, Loss author: {author_loss:.4f}")
    # progress_bar.set_description(desc=f"Loss: {loss_value:.4f} Loss classifier: {loss_classifier:.4f} Loss Box Reg: {loss_box_reg:.4f} Loss objectness: {loss_objectness:.4f} Loss RPN Box Reg: {loss_rpn_box_reg:.4f}")
  
  return train_losses

def validate(val_data_loader, model):
  print("Validating...")
  val_iter = 0
  val_losses = []

  progress_bar = tqdm(val_data_loader, total=len(val_data_loader))

  for i, data in enumerate(progress_bar):
    images, targets = data
    images = list(image.to(DEVICE) for image in images)
    targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
    with torch.no_grad():
        loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    loss_classifier = loss_dict["loss_classifier"]
    loss_value = losses.item()
    val_losses.append(loss_value)
    val_iter+=1

    if i % 100 == 99:
      writer.add_scalar("epoch_avg_val_loss", np.average(val_losses), epoch*len(val_data_loader) + i)
      writer.add_scalar("epoch_avg_classifier_val_loss", loss_classifier, epoch*len(val_data_loader) + i)

    progress_bar.set_description(desc=f"Loss: {loss_value:.4f}, Loss classifier: {loss_classifier:.4f}")
  return val_losses

def validate_with_author(val_data_loader, model):
  print("Validating...")
  val_iter = 0
  val_losses = []

  progress_bar = tqdm(val_data_loader, total=len(val_data_loader))

  for i, data in enumerate(progress_bar):
    images, targets = data
    images = list(image.to(DEVICE) for image in images)
    targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
    with torch.no_grad():
        loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    loss_classifier = loss_dict["loss_classifier"]
    author_loss = loss_dict["loss_authors"]
    loss_value = losses.item()
    val_losses.append(loss_value)
    val_iter+=1

    if i % 100 == 99:
      writer.add_scalar("epoch_avg_val_loss", np.average(val_losses), epoch*len(val_data_loader) + i)
      writer.add_scalar("epoch_avg_author_val_loss", author_loss, epoch*len(val_data_loader) + i)
      writer.add_scalar("epoch_avg_classifier_val_loss", loss_classifier, epoch*len(val_data_loader) + i)

    progress_bar.set_description(desc=f"Loss: {loss_value:.4f}, Loss classifier: {loss_classifier:.4f}, Loss author: {author_loss:.4f}")
  return val_losses

def save_model(epoch, model, optimizer):
    """
    Function to save the trained model till current epoch, or whenver called
    """
    torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, 'model.pth')

# Training of the model
import time
model.train()
earlystopping = False 

avg_train_losses = []
avg_val_losses = []

#torch.save(model, PATH)

for epoch in range(0, NUM_EPOCHS):
    print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")
    start = time.time()
    if args.add_authors:
      train_loss = train_with_author(train_loader, model)
      val_loss = validate_with_author(val_loader, model)
    else:
      train_loss = train(train_loader, model)
      val_loss = validate(val_loader, model)
    avg_train = np.average(train_loss)
    avg_val = np.average(val_loss)
    avg_train_losses.append(avg_train)
    avg_val_losses.append(avg_val)
    print(f"Epoch #{epoch+1} train loss: {avg_train:.5f}")
    print(f"Epoch #{epoch+1} validation loss: {avg_val:.5f}")
    end = time.time()
    print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")
    if epoch > 5:     #Early stopping with a patience of 1 and a minimum of 5 epochs 
      if avg_val_losses[-1]>=avg_val_losses[-2]:
          print("Early Stopping Triggered With Patience 1")
          earlystopping = True 
      else:
        torch.save(model, PATH)
      if earlystopping:
        break
    else:
      #torch.save(model, PATH)
      save_model(epoch, model, optimizer)



#model = torch.load(PATH)
checkpoint = torch.load('model.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])

model.to(DEVICE).eval()

from torchmetrics.detection.mean_ap import MeanAveragePrecision

# evaluation
detection_threshold=0.50
frame_count = 0
total_fps = 0

# metric = MeanAveragePrecision(class_metrics=True)
metric = MeanAveragePrecision(class_metrics=True)
predictions_avg = []
actual_avg = []
##############
predictions_authors_avg = []
actual_authors_avg = []

# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

predicted_boxes_for_test = []
target_boxes_for_test = []

with torch.no_grad():
  for i, data in enumerate(val_loader):

    images, targets = data
    images = list(image.to(DEVICE) for image in images)
    targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

    # generate outputs
    outputs = model(images)

    # get the current fps

    # load all detection to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        
    targets = [{k: v.to('cpu') for k, v in t.items()} for t in targets]

    bounding_boxes_pred = []
    labels_pred = []
    authors_pred = []
    scrs = []
    scrs_authors = []
    bounding_boxes_targ = []
    labels_target = []
    authors_target = []

    for (out, targ) in zip(outputs, targets):
      if len(out["boxes"]) != 0:
        boxes = out["boxes"].data.numpy()
        scores = out["scores"].data.numpy()
        authors_scores = out["authors_scores"].data.numpy()
        # filtering according to the threshold
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        # get all predicted classes
        predicted_classes = [CLASSES[i] for i in out["labels"].cpu().numpy()]
        actual_boxes = targ["boxes"].data.numpy()
        draw_actual_boxes = actual_boxes.copy()
        actual_classes = [CLASSES[i] for i in targ["labels"].cpu().numpy()]
        actual_authors = [AUTHORS[i] for i in targ["author"].cpu().numpy()]

        # draw the bounding boxes and write the class name on top of it

        actual_classes = targ["labels"].data.numpy()
        actual_authors = targ["author"].data.numpy()
        target_boxes = targ["boxes"].data.numpy()

        predicted_classes = out["labels"].data.numpy()
        predicted_authors = out["authors"].data.numpy()

        print(len(predicted_classes))
        print(len(predicted_authors))

        lista1 = predicted_classes.copy()
        
        predicted_classes = predicted_classes[scores >= detection_threshold].astype(np.int32)

        predicted_authors = [predicted_authors[i] for i in range(len(lista1)) if scores[i]>=detection_threshold]
        authors_scores = [authors_scores[i] for i in range(len(lista1)) if scores[i]>=detection_threshold]

        scores = scores[scores >= detection_threshold].astype(np.float32)    

        print(len(predicted_classes))
        print(predicted_authors)

        

        print(authors_scores)

        id = targ["image_id"].data.numpy()

        for i in range(0, len(boxes)):
          tmp=[]   
          bs = []  
          tmp.append(id[0])
          tmp.append(predicted_classes[i])
          tmp.append(scores[i])
          tmp.append(boxes[i][0])
          tmp.append(boxes[i][1])
          tmp.append(boxes[i][2])
          tmp.append(boxes[i][3])
          bs.append(boxes[i][0])
          bs.append(boxes[i][1])
          bs.append(boxes[i][2])
          bs.append(boxes[i][3])
          bounding_boxes_pred.append(bs)
          scrs.append(scores[i])
          scrs_authors.append(authors_scores[i])
          labels_pred.append(predicted_classes[i])
          authors_pred.append(predicted_authors[i])
          predicted_boxes_for_test.append(tmp)

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
          target_boxes_for_test.append(tmp)

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

print(len(predictions_avg))
print(len(actual_avg))

from pprint import pprint
pprint(metric.compute())

metric.update(predictions_authors_avg, actual_authors_avg)

print(len(predictions_authors_avg))
print(len(actual_authors_avg))

pprint(metric.compute())