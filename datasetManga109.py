import res.transforms as T
import torch
from PIL import Image
from torch.utils.data import Dataset
import glob as glob
import os
    
# define the training tranforms
def get_train_transform():
    return T.Compose([
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float)
    ])
# define the training transforms with data augmentation
def get_train_transform_aug():
   return T.Compose([
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomPhotometricDistort(p=0.5),
        T.RandomZoomOut(p=0.5)
    ])

# define the validation tranforms
def get_valid_transform():
    return T.Compose([
        T.PILToTensor(),
        T.ConvertImageDtype(torch.float)
    ])
 
def pil_loader(path):
  with open(path, "rb") as f:
    img = Image.open(f)
    return img.convert("RGB")

""" Custom class used to create the training and validation sets. """
class CustomDataset(Dataset):
  # Initialize configurations 
  def __init__(self, images, width, height, classes, transforms=None):

    self.transforms=transforms
    self.images = images
    self.height = height
    self.width = width
    self.classes = classes

    self.image_paths = images["path"].to_list()
    self.images_annotations = images["annotation"]
    self.all_images = ["".join(path.split(os.path.sep)[-2:]) for path in self.image_paths]

  # Method used to get (image, target)
  def __getitem__(self, idx):
    image_path = self.image_paths[idx]
    # read image
    image = pil_loader(image_path)
    # resize image
    image_resized = image.resize((self.height, self.width))

    boxes = []
    labels = []
    authors = []

    image_width, image_height = image.size

    for annotation_type in self.classes[1:]:
      # extract all annotations of the current class
      rois = self.images_annotations[idx][annotation_type]
      for roi in rois:
        labels.append(self.classes.index(annotation_type))
        authors.append(roi["author"])
        # xmin = left corner x-coordinates
        xmin = roi["@xmin"]
        # xmax = right corner x-coordinates
        xmax = roi["@xmax"]
        # ymin = left corner y-coordinates
        ymin = roi["@ymin"]
        # ymax = right corner y-coordinates
        ymax = roi["@ymax"]

        # resize bounding box according to the desired size
        xmin_final = (xmin/image_width)*self.width
        xmax_final = (xmax/image_width)*self.width
        ymin_final = (ymin/image_height)*self.height
        yamx_final = (ymax/image_height)*self.height

        boxes.append([xmin_final, ymin_final, xmax_final, yamx_final])

    # bounding box to tensor
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    # area of the bounding boxes
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    # no crowd instances
    iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
    # labels to tensor
    labels = torch.as_tensor(labels, dtype=torch.int64)
    authors = torch.as_tensor(authors, dtype=torch.int64)
    # prepare the final dictionary
    target = {}
    target["boxes"] = boxes
    target["labels"] = labels
    target["area"] = area
    target["author"] = authors
    target["iscrowd"] = iscrowd
    image_id = torch.tensor([idx])
    target["image_id"] = image_id
    # apply the image transforms
    if self.transforms:
        image_resized, target = self.transforms(image_resized, target)

    return image_resized, target

  def __len__(self):
        return len(self.all_images)