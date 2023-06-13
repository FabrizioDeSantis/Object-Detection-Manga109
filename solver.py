import torch
from tqdm import tqdm
import numpy as np
import model
import os
from utils import early_stopping

class Solver(object):
  
  """ Solver for training and testing """
  def __init__(self, train_data_loader, val_data_loader, device, writer, args, n_classes, n_authors):
    """ Initialize configuration """

    self.args = args
    self.n_classes = n_classes
    self.n_authors = n_authors
    self.model_name = self.args.file_name

    # model definition
    self.model = model.create_model(self.n_classes, self.n_authors, self.args).to(device)

    '''
    If the model is pretrained, select only the model parameters that require the gradient calculation.
    Else, all model parameters are updated during training.
    '''
    if self.args.pretrained:
      params = [p for p in self.model.parameters() if p.requires_grad]
      names = [n for n,p in self.model.named_parameters() if p.requires_grad]
    else:
      for p in self.model.parameters():
        p.requires_grad=True
      params = [p for p in self.model.parameters()]
      names = [n for n,p in self.model.named_parameters()]
    # print(names)
    # choose optimizer
    if self.args.optimizer == "SGD":
        self.optimizer = torch.optim.SGD(params, lr=self.args.learning_rate, momentum=0.9)
    elif self.args.optimizer == "Adam":
        self.optimizer = torch.optim.Adam(params, lr=self.args.learning_rate)

    # other training parameters
    self.epochs = self.args.num_epochs
    self.train_loader = train_data_loader
    self.val_loader = val_data_loader

    self.device = device
    self.writer = writer

  def save_model(self):
    # function to save the model
    ext = os.path.splitext(self.model_name)[1].lower()
    if ext == ".pth":
      check_path = os.path.join(self.args.checkpoint_path, self.model_name)
      torch.save(self.model.state_dict(), check_path)
    elif ext == ".pt":
      torch.save(self.model, self.model_name)
    print("Model saved!")

  def load_model(self, device):
    # function to load the model
    check_path = os.path.join(self.args.checkpoint_path, self.model_name)
    self.model.load_state_dict(torch.load(check_path, map_location = torch.device(device)))
    print("Model loaded!")

  def train(self):
    """ Method used to train the model with early stopping implementatinon. """
    print("Training...")
    # put the model in training mode
    self.model.train()
    # keep track of average validation loss
    avg_val_losses = []

    for epoch in range(0, self.epochs):
      # record the training losses for each batch in this epoch
      train_losses = []
      # classifier_losses = []
      # create a terminal progress bar
      progress_bar = tqdm(self.train_loader, total=len(self.train_loader))

      # iterate over training data
      for i, data in enumerate(progress_bar):
        # clear the gradients of all optimized variables
        self.optimizer.zero_grad()

        images, targets = data
        images = list(image.to(self.device) for image in images)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
        # return the losses
        loss_dict = self.model(images, targets)
        # loss classifier
        # loss_classifier = loss_dict["loss_classifier"]
        # classifier_losses.append(loss_classifier.item())
        # calculate the sum of the losses to obtain the main loss
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_losses.append(loss_value)
        # backward pass: compute gradient of the loss with respect to model parameters
        losses.backward()
        # perform a single optimization step
        self.optimizer.step()
        # save metrics on tensorboard
        if i % self.args.print_every == (self.args.print_every-1):
          self.writer.add_scalar("epoch_avg_train_loss", np.average(train_losses), epoch*len(self.train_loader) + i)
          #self.writer.add_scalar("epoch_avg_classifier_train_loss", np.average(classifier_losses), epoch*len(self.train_loader) + i)

        progress_bar.set_description(desc=f"Loss: {loss_value:.4f}")
      # return the validation loss
      val_loss=self.validate(epoch)
      val_loss = np.average(val_loss)
      avg_val_losses.append(val_loss)

      print(f"Epoch #{epoch+1} train loss: {np.average(train_losses):.3f}")   
      print(f"Epoch #{epoch+1} validation loss: {val_loss:.3f}")   
      # early stopping
      if self.args.early_stopping:
        # check if the minimum number of training epochs is reached before enabling early stopping
        if epoch > self.args.num_min_epochs:
          if early_stopping(avg_val_losses, self.args.early_stopping):
            # if early stopping is triggered, exit from training
            break
          else:
            if val_loss == min(avg_val_losses):
              self.save_model()
        else:
          if val_loss == min(avg_val_losses):
            self.save_model()
      else:
        if val_loss == min(avg_val_losses):
            self.save_model()

    self.writer.flush()
    self.writer.close()
    print("Finished training")

  def validate(self, epoch):
    # record the validation losses for each batch in this epoch
    val_losses = []
    #classifier_losses = []

    # create a terminal progress bar
    progress_bar = tqdm(self.val_loader, total=len(self.val_loader))

    # iterate over validation data
    for i, data in enumerate(progress_bar):
      images, targets = data
      images = list(image.to(self.device) for image in images)
      targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
      # no need to calculate the gradients for outputs
      with torch.no_grad():
          # return the losses
          loss_dict = self.model(images, targets)
      # obtain the main loss
      losses = sum(loss for loss in loss_dict.values())
      #loss_classifier = loss_dict["loss_classifier"]
      #classifier_losses.append(loss_classifier.item())
      loss_value = losses.item()
      val_losses.append(loss_value)
      # 
      if i % self.args.print_every == (self.args.print_every-1):
        self.writer.add_scalar("epoch_avg_val_loss", np.average(val_losses), epoch*len(self.val_loader) + i)
        #self.writer.add_scalar("epoch_avg_classifier_train_loss", np.average(classifier_losses), epoch*len(self.val_loader) + i)

      progress_bar.set_description(desc=f"Loss: {loss_value:.4f}")

    print("Finished validating")

    return val_losses
  
  """
  The following training and validation functions are similar to the previous ones.
  In this case, the losses relating to the author classificaation are also retrieved and printed.
  """

  def train_with_authors(self):
    """ Method used to train the model with author classification and early stopping implementatinon. """
    self.model.train()

    avg_val_losses=[]

    print("Training...")

    for epoch in range(0, self.epochs):

      train_losses = []
      classifier_losses = []
      author_losses = []

      progress_bar = tqdm(self.train_loader, total=len(self.train_loader))

      for i, data in enumerate(progress_bar):
            
        self.optimizer.zero_grad()

        images, targets = data
        images = list(image.to(self.device) for image in images)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
        loss_dict = self.model(images, targets) # return the loss

        loss_classifier = loss_dict["loss_classifier"]
        classifier_losses.append(loss_classifier.item())
        author_loss = loss_dict["loss_authors"]
        author_losses.append(author_loss.item())

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_losses.append(loss_value)
        losses.backward()
        self.optimizer.step()

        if i % self.args.print_every == (self.args.print_every-1):
          self.writer.add_scalar("epoch_avg_train_loss", np.average(train_losses), epoch*len(self.train_loader) + i)
          self.writer.add_scalar("epoch_avg_author_train_loss", np.average(author_losses), epoch*len(self.train_loader) + i)
          self.writer.add_scalar("epoch_avg_classifier_train_loss", np.average(classifier_losses), epoch*len(self.train_loader) + i)

        progress_bar.set_description(desc=f"Loss: {loss_value:.4f}, Loss classifier: {loss_classifier:.4f}")

      val_loss=self.validate_with_author(epoch)
      val_loss = (np.average(val_loss))
      avg_val_losses.append(val_loss)

      print(f"Epoch #{epoch+1} train loss: {np.average(train_losses):.3f}")   
      print(f"Epoch #{epoch+1} validation loss: {val_loss:.3f}")  

      if self.args.early_stopping:
        # check if the minimum number of training epochs is reached before enabling early stopping
        if epoch > self.args.num_min_epochs:
          if early_stopping(avg_val_losses, self.args.early_stopping):
            # if early stopping is triggered, exit from training
            break
          else:
            if val_loss == min(avg_val_losses):
              self.save_model()
        else:
          if val_loss == min(avg_val_losses):
            self.save_model()
      else:
        if val_loss == min(avg_val_losses):
            self.save_model()

    self.writer.flush()
    self.writer.close()
    print("Finished training")

  def validate_with_author(self, epoch):
    val_losses = []
    author_losses = []
    classifier_losses = []
    progress_bar = tqdm(self.val_loader, total=len(self.val_loader))
    for i, data in enumerate(progress_bar):
      images, targets = data
      images = list(image.to(self.device) for image in images)
      targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
      with torch.no_grad():
          loss_dict = self.model(images, targets)
      losses = sum(loss for loss in loss_dict.values())
      loss_classifier = loss_dict["loss_classifier"]
      classifier_losses.append(loss_classifier.item())
      author_loss = loss_dict["loss_authors"]
      author_losses.append(author_loss.item())
      loss_value = losses.item()
      val_losses.append(loss_value)

      if i % self.args.print_every == (self.args.print_every-1):
        self.writer.add_scalar("epoch_avg_val_loss", np.average(val_losses), epoch*len(self.val_loader) + i)
        self.writer.add_scalar("epoch_avg_author_val_loss", np.average(author_losses), epoch*len(self.val_loader) + i)
        self.writer.add_scalar("epoch_avg_classifier_val_loss", np.average(classifier_losses), epoch*len(self.val_loader) + i)

      progress_bar.set_description(desc=f"Loss: {loss_value:.4f}, Loss classifier: {loss_classifier:.4f}, Loss author: {author_loss:.4f}")

    print("Finished validating")

    return val_losses