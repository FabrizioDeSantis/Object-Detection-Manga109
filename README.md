# FasterRCNN-Manga109-Object-Detection
The purpose of this deep learning project is to conduct an object detection task using models like FasterRCNN. The models were trained on the Manga109 dataset, a dataset compiled by the Aizawa Yamasaki Matsui Laboratory, University of Tokyo. Manga109 is composed of 109 manga volumes drawn by professional mangaka in Japan. The project consists of the following Python modules:

- `object-detection-main.py`: This module is responsible for launching the simulation.
- `model.py`: This module is responsibile for creating the model. The supported models are: FasterRCNN, RetinaNet, SSD300. In particular, you can modify the FasterRCNN template to set some custom parameters and possibly add author classification.
- `custom_roi_heads.py`: This module implements a custom RoIHeads for author classification, a custom fasterrcnn loss with author classification and a custom FastRCNNPredictor for custom classes.
- `datasetManga109.py`: This module implements the CustomDataset used for training and validating the model.
- `solver.py`: This module includes methods for training, validation, with or without the author classification. It also provides functionality for saving and loading the model and visualizing model weights.
- `manga109api_custom.py`: This module is an extension of manga109api from https://github.com/manga109/manga109api/tree/main/manga109api. The parser has been extended to support adding author information to annotations. 
- `metrics.py`: This module is responsible for the calculation of evaluation metrics, in particular for mAP (mean Average Precision) computation.
- `utils.py`: This module contains various function for different purposes, such as checking for annotations in images, the early stopping implementation and uploading image information.
- `inference.py`: This module is responsible for making inference on given images.

## The parameters that can be provided through the command line and allow customization of the execution are:

| Argument              | Description                                                                                                                        |
|-----------------------|------------------------------------------------------------------------------------------------------------------------------------|
| model                 | The name of the model (e.g. FasterRCNN)                                                                                            |
| bb                    | The name of the backbon for the FasterRCNN model (e.g. resnet50v2, resnet50, mobilenet)                                            |
| pretrained            | Determines whether to use a pre-trained model or not                                                                               |
| fn                    | The name of the model to be saved or loaded                                                                                        |
| add_auth              | Enabling the author classification                                                                                                 |
| num_epochs            | The total number of training epochs                                                                                                |
| min_ep                | The minimum number of training epochs before enabling early stopping                                                               |
| bs                    | The learning rate for optimization                                                                                                 |
| lr                    | The number of workers in the data loader                                                                                           |
| print_every           | The frequency of printing losses during training and validation                                                                    |
| seed                  | The random seed used to ensure reproducibility                                                                                     |
| opt                   | The optimizer used for training (SGD or Adam)                                                                                      |
| early_stopping        | The threshold for early stopping during training (0 = disabled)                                                                    |
| mode                  | Determines the mode of the execution (0 = training, 1 = resume training, 2 = inference)                                            |
| split                 | The threshold used to split the dataset into train and validation subsets                                                          |
| dataset               | The path to retrieve the dataset                                                                                                   |
| checkpoint_path       | The path to save and retrieve the trained model                                                                                    |
| inference_path        | The path that contains the images for inference                                                                                    |
| dataset_aug           | Determines if data augmentation is applied to the images                                                                           |
| res                   | Resize dimensions of the input images for preprocessing                                                                            |
| det_thresh            | Value of detection treshold for inference and metrics computation                                                                  |
| body                  | Include "body" class                                                                                                               |
| face                  | Include "face" class                                                                                                               |
| text                  | Include "text" class                                                                                                               |
| frame                 | Include "frame" class                                                                                                              |
| size32                | Include size 32 for anchors                                                                                                        |
| size64                | Include size 64 for anchors                                                                                                        |
| size128               | Include size 128 for anchors                                                                                                       |
| size256               | Include size 256 for anchors                                                                                                       |
| size512               | Include size 512 for anchors                                                                                                       |
| ar05                  | Include aspect ratio 1:2 for anchors                                                                                               |
| ar1                   | Include aspect ratio 1:1 for anchors                                                                                               |
| ar2                   | Include aspect ratio 2:1 for anchors                                                                                               |
| rpn_nms_th            | NMS threshold used for postprocessing the RPN proposals                                                                            |
| rpn_fg_th             | Minimum IoU between the anchor and the GT box so that they can be considered as positive during RPN training                       |
| rpn_bg_th             | Maximum IoU between the anchor and the GT box so that they can be considered as negative during RPN training                       |
| rpn_score_th          | During inference, only return proposals with a classification score greater than rpn_score_th                                      |
| box_nms_th            | NMS threshold used for postprocessing the RPN proposals                                                                            |
| box_fg_th             | Minimum IoU between the proposal and the GT box so that they can be considered as positive during the classification head training |
| box_bg_th             | Maximum IoU between the proposal and the GT box so that they can be considered as negative during the classification head training |
| box_score_th          | During inference, only return proposals with a classification score greater than box_score_th                                      |
| box_detections        | Maximum number of detections per image, for all classes                                                                            |
| map_authors           | Calculate mAP for author classification (available only if the author classification is enabled)                                   |

### Prerequisites

- [Python](https://www.python.org/downloads/) 3.5 or later installed on your system.
- The following modules:
  - [os](https://docs.python.org/3/library/os.html)
  - [json](https://docs.python.org/3/library/json.html)
  - [torch](https://pytorch.org/)
  - [numpy](https://numpy.org/)
  - [tqdm](https://tqdm.github.io/)
  - [matplotlib](https://matplotlib.org/)
  - [torchvision](https://pytorch.org/vision/stable/index.html)
  - [cv2](https://docs.opencv.org/4.5.2/)
  - [PIL](https://pillow.readthedocs.io/en/stable/)
  - [math](https://docs.python.org/3/library/math.html)
  - [random](https://docs.python.org/3/library/random.html)
  - [argparse](https://docs.python.org/3/library/argparse.html)
  - [torch.utils.tensorboard](https://pytorch.org/docs/stable/tensorboard.html)
  - [torchmetrics](https://torchmetrics.readthedocs.io/en/stable/)
  - [pandas](https://pandas.pydata.org/)

### Usage

Example of script launch:

```shell
python object-detection-main.py -model=fasterrcnn -bb=resnet50v2 -min_ep=0 -early_stopping=1 -num_epochs=10 -lr=0.0001 -opt=SGD -add_auth=1 -bs=4 -res=512 -size32=0 -size64=0 -ar05=0 -ar2=0 -frame=0
```