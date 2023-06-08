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

    # Define the model

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
    # Choose optimizer
    if self.args.optimizer == "SGD":
        self.optimizer = torch.optim.SGD(params, lr=self.args.learning_rate, momentum=0.9, weight_decay=0.0005)
    elif self.args.optimizer == "Adam":
        self.optimizer = torch.optim.Adam(params, lr=self.args.learning_rate)

    self.epochs = self.args.num_epochs
    self.train_loader = train_data_loader
    self.val_loader = val_data_loader

    self.device = device
    self.writer = writer

  def save_model(self):
    # function to save the model
    check_path = os.path.join(self.args.checkpoint_path, self.model_name)
    torch.save(self.model.state_dict(), check_path)
    print("Model saved!")

  def load_model(self, device):
    # function to load the model
    check_path = os.path.join(self.args.checkpoint_path, self.model_name)
    self.model.load_state_dict(torch.load(check_path, map_location = torch.device(device)))
    print("Model loaded!")

  def train(self):

    self.model.train()

    print("Training...")

    avg_val_losses = []

    for epoch in range(0, self.epochs):

      train_losses = []

      progress_bar = tqdm(self.train_loader, total=len(self.train_loader))

      for i, data in enumerate(progress_bar):
            
        self.optimizer.zero_grad()

        images, targets = data
        images = list(image.to(self.device) for image in images)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
        loss_dict = self.model(images, targets) # return the loss

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_losses.append(loss_value)
        losses.backward()
        self.optimizer.step()

        if i % self.args.print_every == (self.args.print_every-1):
          self.writer.add_scalar("epoch_avg_train_loss", np.average(train_losses), epoch*len(self.train_loader) + i)

        progress_bar.set_description(desc=f"Loss: {loss_value:.4f}")

      val_loss=self.validate(epoch)
      avg_val_losses.append(np.average(val_loss))

      print(f"Epoch #{epoch+1} train loss: {sum(train_losses)//len(self.train_loader):.3f}")   
      print(f"Epoch #{epoch+1} validation loss: {sum(val_loss)//len(self.val_loader):.3f}")   

      if self.args.early_stopping:
        if epoch > self.args.num_min_epochs:
          if early_stopping(avg_val_losses, self.args.early_stopping):
            break
          else:
            self.save_model()
        else:
          self.save_model()
      else:
        self.save_model()

    self.writer.flush()
    self.writer.close()
    print("Finished training")

  def validate(self, epoch):
    val_losses = []
    progress_bar = tqdm(self.val_loader, total=len(self.val_loader))
    for i, data in enumerate(progress_bar):
      images, targets = data
      images = list(image.to(self.device) for image in images)
      targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
      with torch.no_grad():
          loss_dict = self.model(images, targets)
      losses = sum(loss for loss in loss_dict.values())
      # loss_classifier = loss_dict["loss_classifier"]
      loss_value = losses.item()
      val_losses.append(loss_value)

      if i % self.args.print_every == (self.args.print_every-1):
        self.writer.add_scalar("epoch_avg_val_loss", np.average(val_losses), epoch*len(self.val_loader) + i)
        # self.writer.add_scalar("epoch_avg_classifier_val_loss", loss_classifier, epoch*len(self.val_loader) + i)

      progress_bar.set_description(desc=f"Loss: {loss_value:.4f}")

    print("Finished validating")

    return val_losses

  def train_with_authors(self):

    self.model.train()

    avg_val_losses=[]

    print("Training...")

    for epoch in range(0, self.epochs):

      train_losses = []

      progress_bar = tqdm(self.train_loader, total=len(self.train_loader))

      for i, data in enumerate(progress_bar):
            
        self.optimizer.zero_grad()

        images, targets = data
        images = list(image.to(self.device) for image in images)
        targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
            
        loss_dict = self.model(images, targets) # return the loss

        loss_classifier = loss_dict["loss_classifier"]
        author_loss = loss_dict["loss_authors"]
        # loss_box_reg = loss_dict["loss_box_reg"]
        # loss_objectness = loss_dict["loss_objectness"]
        # loss_rpn_box_reg = loss_dict["loss_rpn_box_reg"]

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_losses.append(loss_value)
        losses.backward()
        self.optimizer.step()

        if i % self.args.print_every == (self.args.print_every-1):
          self.writer.add_scalar("epoch_avg_train_loss", np.average(train_losses), epoch*len(self.train_loader) + i)
          self.writer.add_scalar("epoch_avg_author_train_loss", author_loss, epoch*len(self.train_loader) + i)
          self.writer.add_scalar("epoch_avg_classifier_train_loss", loss_classifier, epoch*len(self.val_loader) + i)

        progress_bar.set_description(desc=f"Loss: {loss_value:.4f}, Loss classifier: {loss_classifier:.4f}")

      val_loss=self.validate_with_author(epoch)
      avg_val_losses.append(val_loss)

      print(f"Epoch #{epoch+1} train loss: {sum(train_losses)//len(self.train_loader):.3f}")   
      print(f"Epoch #{epoch+1} validation loss: {sum(val_loss)//len(self.val_loader):.3f}")  

      if self.args.early_stopping:
        if epoch > self.args.num_min_epochs:
          if early_stopping(avg_val_losses, self.args.early_stopping):
            break
          else:
            self.save_model()
        else:
          self.save_model()
      else:
        self.save_model()

    self.writer.flush()
    self.writer.close()
    print("Finished training")

  def validate_with_author(self, epoch):
    val_losses = []
    progress_bar = tqdm(self.val_loader, total=len(self.val_loader))
    for i, data in enumerate(progress_bar):
      images, targets = data
      images = list(image.to(self.device) for image in images)
      targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
      with torch.no_grad():
          loss_dict = self.model(images, targets)
      losses = sum(loss for loss in loss_dict.values())
      loss_classifier = loss_dict["loss_classifier"]
      author_loss = loss_dict["loss_authors"]
      loss_value = losses.item()
      val_losses.append(loss_value)

      if i % self.args.print_every == (self.args.print_every-1):
        self.writer.add_scalar("epoch_avg_val_loss", np.average(val_losses), epoch*len(self.val_loader) + i)
        self.writer.add_scalar("epoch_avg_author_val_loss", author_loss, epoch*len(self.val_loader) + i)
        self.writer.add_scalar("epoch_avg_classifier_val_loss", loss_classifier, epoch*len(self.val_loader) + i)

      progress_bar.set_description(desc=f"Loss: {loss_value:.4f}, Loss classifier: {loss_classifier:.4f}, Loss author: {author_loss:.4f}")

    print("Finished validating")

    return val_losses

    