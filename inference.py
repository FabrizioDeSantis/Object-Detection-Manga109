import cv2
import matplotlib.pyplot as plt
import torch
import os
from datasetManga109 import pil_loader
from torchvision.transforms import functional as F
import torchvision.transforms as transforms
import numpy as np

def load_image_tensor(image_paths, device):
  """
  This function takes the path of each images and convert the image to tensor
  Args:
    image_paths: List[str]
  Returns:
    input_images: List[str]
  """
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
  """
  This function takes the path of the directory where images are stored and returns all paths in a single list
  Args:
    inference_path: str
  Returns:
    paths: List[str]
  """
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
  """
  Function used to make predictions on images that do not belong to the original dataset.
  Images with predicted bounding boxes are saved in the same folder as the images used for inference.
  Args:
    inference_model: the pretrained model
    classes: List[str]
    authors: List[int]
  """
  
  inference_model.eval()

  # get the paths of the images on which we want to make predictions
  images_path = load_images(args.inference_path)

  DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

  NUM_CLASSES = len(classes)

  COLORS = np.random.uniform(0, 255, size=(NUM_CLASSES, 3))

  AUTHORS = authors.copy()
  AUTHORS.insert(0, "__background__")

  to_pil = transforms.ToPILImage()

  # convert images to tensor
  images_tensor = load_image_tensor(images_path, DEVICE)

  # generate outputs
  outputs = get_results(inference_model, images_tensor)
  # load all detection to CPU for further operations
  outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
  # convert images 
  images = [np.array(to_pil(im)) for im in images_tensor]

  for (out, img, name) in zip(outputs, images, images_path):
    # carry further only if there are detected boxes
    if len(out["boxes"]) != 0:
      boxes = out["boxes"].data.numpy()
      scores = out["scores"].data.numpy()
      # filtering according to the detection threshold
      boxes = boxes[scores >= (args.detection_threshold)].astype(np.int32)
      draw_boxes = boxes.copy()
      # get all predicted classes
      predicted_classes = [classes[i] for i in out["labels"].cpu().numpy()]
      # draw the bounding boxes and write the class name on top of it
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
      # get path and name of the image
      path = os.path.splitext(name)[0].lower()
      # get extension
      ex = os.path.splitext(name)[1].lower()
      # save image
      plt.savefig(path + "_inference_result" + ex)
      plt.show()