import cv2
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms

def early_stopping(losses, patience):
    """
    Early stopping implementation with given patience value.
    If patience = 2 and there is no improvement for two epochs, early stopping will be triggered.
    """
    best_loss = float('inf') # initialize best loss value
    num_epochs_no_improvement = 0  # counter to keep track of the number of epochs without improvement
    
    for epoch, loss in enumerate(losses):
        if loss < best_loss:
            best_loss = loss
            num_epochs_no_improvement = 0
        else:
            num_epochs_no_improvement += 1
        # check if number of epochs without improvement on loss value is equal or greater than the patience value.
        if num_epochs_no_improvement >= patience:
            print(f"Early stopping triggered at epoch {epoch}!")
            return True
    return False

def check_annotation_validity(annotation, index, classes):
    """
    This function check the presence of at least one annotated box in the image.
    """
    count = 0
    for cl in classes[1:]:
        if len(annotation["page"][index][str(cl)])==0:
            count += 1
    if count == len(classes[1:]):
        return False
    return True

def load_all_images(parser, classes):
    """
    Using the parser from manga109api_custom, for every book load all the image paths and annotations and store them in a list of lists.
    Args:
      Parser (object)
    Returns:
      images (List[List[str, Dict]])
      author_labels (List[int])
    """
    images = []
    author_labels = []
    for book in parser.books:
      annotation, author = parser.get_annotation(book=book)
      for i in range(0, len(annotation["page"])):
        temp=[]
        if check_annotation_validity(annotation, i, classes):
          path=parser.img_path(book=book, index=i)
          temp.append(path)
          temp.append(annotation["page"][i])
          images.append(temp)
          author_labels.append(author)
    return images, author_labels

def save_predictions_on_tb(draw_actual_boxes, draw_boxes, predicted_classes, actual_classes, classes, img, writer = None):
    """
    This function save target and predicted images with bounding boxes on Tensorboard.
    Create a grid with original image, predicted bboxes and classes and target bboxes and classes.
    Args:
        draw_actual_boxes: coordinates of gt bboxes
        draw_boxes: coordinates of predicted bboxes
        predicted_classes: list of predicted classes
        actual_classes: list of gt classes
        img: original image
        writer: SummaryWriter or None
    If writer is not None, when using this function the grid with the images will be plotted.
    Otherwise, the grid will be saved on tensorboard.
    """
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
    fig = plt.figure(figsize=(20,5))
    img2 = img.copy()
    img3 = img.copy()
    # draw bboxes and labels on predicted image
    for j, box in enumerate(draw_actual_boxes):
      class_name = actual_classes[j]
      color = COLORS[classes.index(class_name)]
      cv2.rectangle(img2, (int(box[0]), int(box[1])),(int(box[2]), int(box[3])), color, 2)
      cv2.putText(img2, class_name, 
                                (int(box[0]), int(box[1]-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 
                                2, lineType=cv2.LINE_AA)
    # draw bboxes and labels on target image
    for j, box in enumerate(draw_boxes):
      class_name = predicted_classes[j]
      color = COLORS[classes.index(class_name)]
      cv2.rectangle(img, (int(box[0]), int(box[1])),(int(box[2]), int(box[3])), color, 2)
      cv2.putText(img, class_name, 
                                (int(box[0]), int(box[1]-5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 
                                2, lineType=cv2.LINE_AA)
    # make grid
    plt.subplot(1,3,1)
    plt.imshow(img3)
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(img2)
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.imshow(img)
    plt.title('Prediction')
    plt.axis('off')
    if writer is not None:
        writer.add_figure("Prediction", fig)
    else:
       plt.show()