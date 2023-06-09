import manga109api_custom
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from solver import Solver
from torch.utils.data import DataLoader
from utils import load_all_images
from datasetManga109 import CustomDataset, get_train_transform, get_valid_transform, get_train_transform_aug
from metrics import calculate_mAP, calculate_mAP_authors
import os
from inference import get_prediction

import argparse

def get_args():

  argParser = argparse.ArgumentParser()

  # general network parameters

  argParser.add_argument("-mode", "--mode", type = int, nargs = '?', const = 1, default = 0, help = "0 for training, 1 for loading a checkpoint, 2 for inference. Default is 0 (training mode)")

  argParser.add_argument("-lr", "--learning_rate", type = float, nargs = '?', const = 1, default = 0.005, help = "Learning rate parameter for optimizer. Default is: 0.005")
  argParser.add_argument("-bs", "--batch_size", type = int, nargs = '?', const = 1, default = 8, help = "Batch size parameter. Default is: 8")
  argParser.add_argument("-add_auth", "--add_authors", type = int, nargs = '?', const = 1, default = 0, help = "1 if you want to include authors, 0 otherwise. Default is: 0")
  argParser.add_argument("-res", "--resize_to", type = int, nargs = '?', const = 1, default = 512, help = "Resize dimensions of the input images. Default is: 512")
  argParser.add_argument("-num_epochs", "--num_epochs", type = int, nargs = '?', const = 1, default = 20, help = "Number of epochs for training (Early stopping is implemented). Default is: 20")
  argParser.add_argument("-min_ep", "--num_min_epochs", type = int, nargs = '?', default = 1, help = "Minimum number of epochs before using early stopping.")
  argParser.add_argument("-fn", "--file_name", type = str, nargs = '?', const = 1, default = "model.pth", help = "Name of the file where the trained model will stored")
  argParser.add_argument("-model", "--model", type = str, nargs = '?', const = 1, default="fasterrcnn", help = "Name of the model. Available models are: FasterRCNN (fasterrcnn), RetinaNet (retinanet), SSD300 (ssd). Default is fasterrcnn")
  argParser.add_argument("-bb", "--backbone", type = str, nargs = '?', const = 1, default = "resnet50", help = "Name of the backbone for a FasterRCNN model. Available backbones are: resnet50, resnet50v2, mobilenet. Default is resnet50")
  argParser.add_argument("-opt", "--optimizer", type = str, nargs = '?', const = 1, default = "SGD", help = "Name of the optimzer. Available optimizers are: SGD, Adam. Default is SGD")
  argParser.add_argument("-checkpoint_path", "--checkpoint_path", type = str, nargs = '?', const = 1, default = "./", help = "Checkpoint path. Default is ./")
  argParser.add_argument("-seed", "--seed", type = int, nargs= '?', const = 1, default = 42, help = "Random seed for dataset division. Default is 42")
  argParser.add_argument("-print_every", "--print_every", type = int, nargs= '?', const = 1, default = 250, help = "Parameter used to determine every how many iterations to save the loss on the tensorboard. Default is 250.")
  argParser.add_argument("-early_stopping", "--early_stopping", type = int, nargs = '?', const = 1, default = 1, help = "Parameter that controls early stopping. 0 = no early stopping. Values greater than 0 represent the value of patience. Eg: 1 = early stopping with patience 1")
  argParser.add_argument("-pretrained", "--pretrained", type = int, nargs= '?', const = 1, default = 1, help = "Use pretrained model.")
  argParser.add_argument("-dataset", "--dataset_dir", type = str, nargs = '?', const = 1, default = "Manga109/Manga109_released_2021_12_30", help = "Directory path of dataset")
  argParser.add_argument("-inference_path", "--inference_path", type = str, nargs = '?', const = 1, default = "./inference_images", help = "Path where the images for inference are saved.")
  argParser.add_argument("-dataset_transform", "--dataset_transform", type = str, nargs = '?', const = 1, default = 0, help = "1 if you want to use transformations, 0 otherwise.")
  
  argParser.add_argument("-det_thresh", "--detection_threshold", type = float, nargs = '?', const = 1, default = 0.50, help = "Detection threshold for the metric computation. Default is: 0.50")
  argParser.add_argument("-split", "--split", type = float, nargs = '?', const = 1, default = 0.20, help = "The value used to split the dataset into train and validation subsets. Default is: 0.20 (80% training and 20% validation).")
  argParser.add_argument("-map_authors", "--map_authors", type = int, nargs = '?', const = 1, default = 1, help = "Calculate mAP for author classification (available only if the author classification is enabled).")
  
  # classes customization

  argParser.add_argument("-body", "--body", type = int, nargs = '?', const = 1, default = 1, help = "1 if you want to train the model to recognize 'body' class, 0 otherwise.")
  argParser.add_argument("-face", "--face", type = int, nargs = '?', const = 1, default = 1, help = "1 if you want to train the model to recognize 'face' class, 0 otherwise.")
  argParser.add_argument("-frame", "--frame", type = int, nargs = '?', const = 1, default = 1, help = "1 if you want to train the model to recognize 'frame' class, 0 otherwise.")
  argParser.add_argument("-text", "--text", type = int, nargs = '?', const = 1, default = 1, help = "1 if you want to the model to recognize 'text' class, 0 otherwise.")
  
  # Specific FasterRCNN parameters

  # anchors customization

  # sizes

  argParser.add_argument("-size32", "--size32", type = int, nargs = '?', const = 1, default = 1, help = "1 if you want to add size 32 for anchors, 0 otherwise")
  argParser.add_argument("-size64", "--size64", type = int, nargs = '?', const = 1, default = 1, help = "1 if you want to add size 32 for anchors, 0 otherwise")
  argParser.add_argument("-size128", "--size128", type = int, nargs = '?', const = 1, default = 1, help = "1 if you want to add size 32 for anchors, 0 otherwise")
  argParser.add_argument("-size256", "--size256", type = int, nargs = '?', const = 1, default = 1, help = "1 if you want to add size 32 for anchors, 0 otherwise")
  argParser.add_argument("-size512", "--size512", type = int, nargs = '?', const = 1, default = 1, help = "1 if you want to add size 32 for anchors, 0 otherwise")

  # aspect ratios

  argParser.add_argument("-ar05", "--ar05", type = int, nargs = '?', const = 1, default = 1, help = "1 if you want to add aspect ratio 1:1 for anchors, 0 otherwise")
  argParser.add_argument("-ar1", "--ar1", type = int, nargs = '?', const = 1, default = 1, help = "1 if you want to add aspect ratio 1:2 for anchors, 0 otherwise")
  argParser.add_argument("-ar2", "--ar2", type = int, nargs = '?', const = 1, default = 1, help = "1 if you want to add aspect ratio 2:1 for anchors, 0 otherwise")

  # thresholds for the RPN network

  argParser.add_argument("-rpn_nms_th", "--rpn_nms_threshold", type = float, nargs = '?', const = 1, default = 0.7, help = "NMS threshold used for postprocessing the RPN proposals. Deafult is: 0.7")
  argParser.add_argument("-rpn_fg_iou_th", "--rpn_fg_iou_threshold", type = float, nargs = '?', const = 1, default = 0.7, help = "Minimum IoU between the anchor and the GT box so that they can be considered as positive during training of the RPN. Deafult is: 0.7")
  argParser.add_argument("-rpn_bg_iou_th", "--rpn_bg_iou_threshold", type = float, nargs = '?', const = 1, default = 0.3, help = "Maximum IoU between the anchor and the GT box so that they can be considered as negative during training of the RPN. Deafult is: 0.7")
  argParser.add_argument("-rpn_score_th", "--rpn_score_threshold", type = float, nargs = '?', const = 1, default = 0.0, help = "During inference, only return proposals with a classification score greater than rpn_score_thresh. Default is: 0.0")

  # thresholds for the classification newtwork

  argParser.add_argument("-box_nms_th", "--box_nms_threshold", type = float, nargs = '?', const = 1, default = 0.5, help = "NMS threshold for the prediction head. Used during inference. Default is: 0.5")
  argParser.add_argument("-box_fg_iou_th", "--box_fg_iou_threshold", type = float, nargs = '?', const = 1, default = 0.5, help = "Minimum IoU between the proposal and the GT box so that they can beconsidered as positive during training of the classification head. Deafult is: 0.5")
  argParser.add_argument("-box_bg_iou_th", "--box_bg_iou_threshold", type = float, nargs = '?', const = 1, default = 0.5, help = "Maximum IoU between the proposal and the GT box so that they can beconsidered as negative during training of the classification head. Deafult is: 0.5")
  argParser.add_argument("-box_score_th", "--box_score_threshold", type = float, nargs = '?', const = 1, default = 0.05, help = "During inference, only return proposals with a classification score greater than box_score_thresh. Default is: 0.05")
  
  # number of detection per image

  argParser.add_argument("-box_detections", "--box_detections_per_img", type = int, nargs = '?', const = 1, default = 100, help = "Maximum number of detections per image, for all classes. Default is: 100")

  return argParser.parse_args()

def check_args_integrity(args):
  if args.detection_threshold < 0 or args.detection_threshold > 1:
    print("Error. Detection threshold (detection_threshold) must be between 0 and 1.")
    os._exit(1)
  if args.num_epochs < 0:
    print("Error. Number of epochs (num_epochs) must be a positive number.")
    os._exit(1)
  if args.num_min_epochs < 0:
    print("Error. Minimum number of epochs can't be a negative value")
  if args.batch_size < 0:
    print("Error. Batch size (batch_size) must be a positive number.")
    os._exit(1)
  if args.print_every < 0:
    print("Error. The number of iterations to print the tensorboard stats must be positive.")
    os._exit(1)
  if args.optimizer != "SGD" and args.optimizer != "Adam":
    print("Error. The optimizer must be SGD or Adam")
    os._exit(1)
  if args.model != "fasterrcnn" and args.model != "retinanet" and args.model != "ssd":
    print("Error. The model name must be fasterrcnn or retinanet")
    os._exit(1)
  if args.backbone != "resnet50" and args.backbone != "resnet50v2" and args.backbone != "mobilenet":
    print("Error. Backbone name must be resnet50, resnet50v2 or mobilenet")
    os._exit(1)
  if args.resize_to <= 0:
    print("Error. resize_to must be a positive number")
    os._exit(1)
  if args.learning_rate <= 0:
    print("Error. Learning rate must be positive")
    os._exit(1)
  if args.early_stopping < 0:
    print("Error. Early stopping value must be positive")
    os._exit(1)
  if args.body != 1 and args.body !=0:
    print("Error. Class selector must be 1 or 0")
    os._exit(1)
  if args.face != 1 and args.face !=0:
    print("Error. Class selector must be 1 or 0")
    os._exit(1)
  if args.frame != 1 and args.frame !=0:
    print("Error. Class selector must be 1 or 0")
    os._exit(1)
  if args.text != 1 and args.text !=0:
    print("Error. Class selector must be 1 or 0")
    os._exit(1)
  if args.size32 == 0 and args.size64 == 0 and args.size128 == 0 and args.size256 == 0 and args.size512 == 0:
    print("Insert at least one size for anchors")
    os._exit(1)
  if args.ar05 == 0 and args.ar1 == 0 and args.ar2 == 0:
    print("Insert at least one aspect ratio for anchors")
    os._exit(1)
  """   
  If model is not a FasterRCNN and the user adds the authors, add_authors is automatically set to 0 
  because the authors classification is not supported with the other models
  """
  if args.add_authors and args.model != "fasterrcnn":
    args.add_authors = 0
  if args.model != "fasterrcnn":
    args.backbone = "-"

def main(args):

  check_args_integrity(args)

  RESIZE_TO=args.resize_to
  BATCH_SIZE=args.batch_size
  NUM_EPOCHS=args.num_epochs
  LEARNING_RATE=args.learning_rate
  NUM_WORKERS=0

  DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

  # create class list
  CLASSES = ["__background__"]
  if args.body:
    CLASSES.append("body")
  if args.face:
    CLASSES.append("face")
  if args.frame:
    CLASSES.append("frame")
  if args.text:
    CLASSES.append("text")

  NUM_CLASSES=len(CLASSES)

  print("---- TRAINING PARAMETERS ----")
  print("Batch size: " + str(BATCH_SIZE))
  print("Resize: " + str(RESIZE_TO))
  print("Model: " + args.model)
  print("Backbone: " + args.backbone)
  print("Pretrained: " + ("Yes" if args.pretrained else "No"))
  print("Optimizer: " + str(args.optimizer))
  print("Learning rate: " + str(LEARNING_RATE))
  print("Epochs: " + str(NUM_EPOCHS))
  print("Authors: " + ("Included" if args.add_authors else "Not included"))
  print("Classes: " + str(NUM_CLASSES))
  print("-----------------------------")

  '''
  Reading the authors from a dedicated file with ID, NAME and BOOK TITLE
  Information about the authors is also used for the division of the dataset
  '''

  file_path = "autori.txt"
  data = pd.read_csv(file_path, sep="\t", header=None)
  data.columns = ["id", "author", "title"]
  ids = data["id"].tolist()
  authors = data["author"].tolist()
  titles = data["title"].tolist()

  AUTHORS = authors.copy()
  AUTHORS.insert(0, "background")
  NUM_AUTHORS = len(AUTHORS)

  authors_list = []

  # Preparing a list of authors to be given as input to the parser

  for id, author, title in zip(ids, authors, titles):
    temp = []
    temp.append(id)
    temp.append(author)
    temp.append(title)
    authors_list.append(temp)

  manga109_root_dir = args.dataset_dir
  
  # Custom parser from manga109api_custom
  p = manga109api_custom.Parser(root_dir=manga109_root_dir, authors_list=authors_list)

  images=[]
  authors_labels = []
  images, authors_labels = load_all_images(p, CLASSES)

  '''
  Dataset split in training and test.
  Author labels are used to divide the dataset in order to have a dataset split that takes into account the author.
  '''
  train_images, val_images, _ , _ = train_test_split(images, authors_labels, shuffle=True, stratify=authors_labels, test_size=args.split, random_state=args.seed)

  # convert list in Pandas DataFrame
  df_train = pd.DataFrame(train_images, columns=["path", "annotation"])
  df_val = pd.DataFrame(val_images, columns=["path", "annotation"])

  if args.dataset_transform:
    train_dataset = CustomDataset(df_train, RESIZE_TO, RESIZE_TO, CLASSES, get_train_transform_aug())
  else:
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

  writer = SummaryWriter('./runs/faster-rcnn-experiment')

  """
  solver
  """
  solver = Solver(train_data_loader=train_loader, val_data_loader=val_loader, device=DEVICE, writer=writer, args=args, n_classes=NUM_CLASSES, n_authors=NUM_AUTHORS)
  solver.load_model(DEVICE)
  if args.mode == 2:
    # inference mode
    solver.load_model(DEVICE)
    get_prediction(inference_model = solver.model, classes = CLASSES, authors = AUTHORS, args = args)
  else:
    if args.mode == 1:
      # load a checkpoint
      solver.load_model(DEVICE)
    """
    training and metrics computation
    """
    if args.add_authors:
      solver.train_with_authors()
      calculate_mAP_authors(model=solver.model, classes=CLASSES, authors=AUTHORS, device=solver.device, val_loader=solver.val_loader, args = args)
    else:
      solver.train()
      calculate_mAP(model=solver.model, classes=CLASSES, device=solver.device, val_loader=solver.val_loader, args=args)

if __name__=="__main__":
  args = get_args()
  main(args)