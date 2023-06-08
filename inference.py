import cv2
import matplotlib.pyplot as plt
import torch
from PIL import Image
import argparse
import os
from datasetManga109 import pil_loader
from torchvision.transforms import functional as F
import torchvision.transforms as transforms
import numpy as np

def load_image_tensor(image_paths, device):
  input_images = []
  for im in image_paths:
    image = pil_loader(im)
    image_tensor = F.pil_to_tensor(image)
    image_tensor = F.convert_image_dtype(image_tensor, torch.float)
    input_images.append(image_tensor.to(device))
  return input_images

def get_results(detector, images):
  with torch.no_grad():
    outputs = detector(images)
    return outputs
  
def load_images(inference_path):
  types = ['.jpg', '.jpeg', '.png']
  paths = []
  for root, _, files in os.walk(inference_path):
    for file in files:
      estensione = os.path.splitext(file)[1].lower()
      if estensione in types:
        img_path = os.path.join(root, file)
        paths.append(img_path)
  return paths
  
def get_prediction(inference_model, classes, authors, args):
  
  inference_model.eval()

  images_path = load_images(args.inference_path)

  DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

  NUM_CLASSES = len(classes)

  COLORS = np.random.uniform(0, 255, size=(NUM_CLASSES, 3))

  AUTHORS = authors.copy()
  AUTHORS.insert(0, "__background__")
  NUM_AUTHORS = len(AUTHORS)

  to_pil = transforms.ToPILImage()

  image_tensor = load_image_tensor(images_path, DEVICE)

  outputs = get_results(inference_model, image_tensor)
  outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

  images = [np.array(to_pil(im)) for im in image_tensor]

  for (out, img, name) in zip(outputs, images, images_path):

    boxes = out["boxes"].data.numpy()
    scores = out["scores"].data.numpy()
    # filtering according to the threshold
    boxes = boxes[scores >= (args.detection_threshold)].astype(np.int32)
    draw_boxes = boxes.copy()
    # get all predicted classes
    predicted_classes = [classes[i] for i in out["labels"].cpu().numpy()]

    for j, box in enumerate(draw_boxes):
          class_name = predicted_classes[j]
          color = COLORS[classes.index(class_name)]
          cv2.rectangle(img, (int(box[0]), int(box[1])),(int(box[2]), int(box[3])), color, 2)
          cv2.putText(img, class_name, 
                                (int(box[0]), int(box[1]-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 
                                2, lineType=cv2.LINE_AA)
    plt.figure(figsize = (10, 10))
    plt.imshow(img)
    path = os.path.splitext(name)[0].lower()
    ex = os.path.splitext(name)[1].lower()
    plt.savefig(path + "_inference_result" + ex)
    plt.show()