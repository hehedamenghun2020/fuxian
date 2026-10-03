"""
This is training endpoint.
@author
- Van Tuan Nguyen (vantuan.nguyen@lqdtu.edu.vn)
- Razvan Beuran (razvan@jaist.ac.jp)
@create date 2023-12-11 00:28:29
"""

from tqdm import tqdm
import torch
from torch import nn
import pickle
import os
import copy
import time

import logging

# Whether to save per-client model checkpoints and training tracking.
# These files are never read by any downstream code (aggregation and
# evaluation use the in-memory state_dict), so saving is disabled by
# default to keep the project folder clean. Set to True to re-enable.
SAVE_CHECKPOINTS = True

# Configure the logging module
logging.basicConfig(level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                    format='%(asctime)s - %(levelname)s - %(message)s')

class ClientTrainer(object):
    """
    Class for training a client model.

    Args:
        model (nn.Module): The model to train.
        loss_function (nn.Module, optional): The loss function to use. Defaults to nn.MSELoss.
        optimizer (torch.optim.Optimizer, optional): The optimizer to use. Defaults to torch.optim.Adam.
        epoch (int, optional): The number of epochs to train for. Defaults to 10.
        batch_size (int, optional): The batch size for training. Defaults to 100.
        lr_rate (float, optional): The learning rate for the optimizer. Defaults to 1e-3.
        patience (int, optional): The number of epochs to wait for improvement in validation loss before early stopping. Defaults to 5.
        save_dir (str, optional): The directory to save the trained model and training tracking information. Defaults to "Checkpoint/ClientModel/".
    """

    def __init__(self, model=None, loss_function=nn.MSELoss, optimizer=torch.optim.Adam,
                    epoch=10, batch_size=100, lr_rate=1e-3, update_type="avg",
                    patience=3, save_dir="Checkpoint/ClientModel/", fedprox_mu=0.001) -> None:
        
        if model is None:
            logging.info("Have to indicate the model to train.")
            return None
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            logging.info("Created saving dir.")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.previous_global_model = copy.deepcopy(self.model)
        self.loss_function = loss_function
        self.lr_rate = lr_rate
        self.optimizer = optimizer(self.model.parameters(), lr=self.lr_rate)
        self.epoch = epoch
        self.batch_size = batch_size
        # self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9, \
        #     last_epoch=-1, verbose=False)
        self.patience = patience
        self.save_dir = save_dir
        self.update_type = update_type
        self.fedprox_mu = fedprox_mu
    
    
    def save_model(self):
        """
        Save the trained model to the specified directory.
        Retry to handle transient file locking on Windows (e.g. antivirus
        scanning the just-written checkpoint), which otherwise raises
        OSError [Errno 22] / RuntimeError "File cannot be opened".
        """
        if not SAVE_CHECKPOINTS:
            return
        logging.info("Saving model to {}".format(self.save_dir))
        os.makedirs(self.save_dir, exist_ok=True)
        save_file = os.path.join(self.save_dir, "model.cpt")
        for attempt in range(5):
            try:
                torch.save(
                    self.model.state_dict(),
                    save_file,
                    _use_new_zipfile_serialization=False,
                )
                return
            except Exception as e:
                logging.warning("Save attempt {} failed: {}".format(attempt + 1, e))
                time.sleep(1)
        # Fallback to the new zipfile serialization
        torch.save(self.model.state_dict(), save_file)
    
    
    def save_tracking_information(self, tracking=None):
        """
        Save training tracking information with retries to avoid transient
        Windows file locking (e.g. antivirus scanning the file).
        """
        if not SAVE_CHECKPOINTS or tracking is None:
            return
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, "training_tracking.pkl")
        for attempt in range(5):
            try:
                with open(path, "wb") as f:
                    pickle.dump(tracking, f)
                return
            except Exception as e:
                logging.warning("Tracking save attempt {} failed: {}".format(attempt + 1, e))
                time.sleep(1)
        with open(path, "wb") as f:
            pickle.dump(tracking, f)
    
    def run(self, train_loader, valid_loader=None):
        """
        Run the training process.

        Args:
            train_loader (torch.utils.data.DataLoader): The data loader for training data.
            valid_loader (torch.utils.data.DataLoader, optional): The data loader for validation data. Defaults to None.
        """
        
        # if self.update_type == "fedprox" or self.update_type == "mse_avg":
        if self.update_type == "fedprox":
            print("Using FedProx or MSE-AVG")
            min_valid_loss = float("inf")
            worse_count = 0
            training_tracking = []
            for epoch in range(self.epoch):
                self.model.train()
                epoch_loss = 0
                for i, batch_input in zip(tqdm(range(len(train_loader)), desc='Training batch: ...'), train_loader):
                    _, _, loss = self.model(batch_input[0].to(self.device))
                    
                    # Add the proximal term to the loss
                    prox_term = 0.0
                    for param, global_param in zip(self.model.parameters(), self.previous_global_model.parameters()):
                        prox_term += torch.sum(torch.square(param - global_param.to(self.device)))
                    loss += self.fedprox_mu * prox_term
                    
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    epoch_loss += loss.item()
                    
                epoch_loss = epoch_loss / len(train_loader)
                # self.lr_scheduler.step()
                if valid_loader is not None:
                    valid_loss = 0
                    self.model.eval()
                    with torch.no_grad():
                        for i, batch_input in zip(tqdm(range(len(valid_loader)), desc='Validating batch: ...'), valid_loader):
                            _, _, loss = self.model(batch_input[0].to(self.device))
                            
                        
                            # Add the proximal term to the loss
                            prox_term = 0.0
                            for param, global_param in zip(self.model.parameters(), self.previous_global_model.parameters()):
                                prox_term += torch.sum(torch.square(param - global_param.to(self.device)))
                            loss += self.fedprox_mu * prox_term

                            valid_loss += loss.item()
                            
                        valid_loss = valid_loss / len(valid_loader)
                        training_tracking.append((epoch_loss, valid_loss))
                        logging.info(f"Epoch {epoch+1} - Training loss: {epoch_loss} - Validating loss: {valid_loss}")

                    if valid_loss < min_valid_loss:
                        min_valid_loss = valid_loss
                        self.save_model()
                        worse_count = 0
                    else:
                        worse_count += 1
                        if worse_count >= self.patience:
                            logging.info(f"Early stopping in epoch {epoch+1}.")
                            self.save_tracking_information(training_tracking)
                            break
                
                self.save_tracking_information(training_tracking)
        else:
            min_valid_loss = float("inf")
            worse_count = 0
            training_tracking = []
            for epoch in range(self.epoch):
                self.model.train()
                epoch_loss = 0
                for i, batch_input in zip(tqdm(range(len(train_loader)), desc='Training batch: ...'), train_loader):
                    _, _, loss = self.model(batch_input[0].to(self.device))
                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    epoch_loss += loss.item()
                epoch_loss = epoch_loss / len(train_loader)
                # self.lr_scheduler.step()
                if valid_loader is not None:
                    valid_loss = 0
                    self.model.eval()
                    with torch.no_grad():
                        for i, batch_input in zip(tqdm(range(len(valid_loader)), desc='Validating batch: ...'), valid_loader):
                            _, _, loss = self.model(batch_input[0].to(self.device))
                            valid_loss += loss.item()
                        
                        valid_loss = valid_loss / len(valid_loader)
                        training_tracking.append((epoch_loss, valid_loss))
                        logging.info(f"Epoch {epoch+1} - Training loss: {epoch_loss} - Validating loss: {valid_loss}")

                    if valid_loss < min_valid_loss:
                        min_valid_loss = valid_loss
                        self.save_model()
                        worse_count = 0
                    else:
                        worse_count += 1
                        if worse_count >= self.patience:
                            logging.info(f"Early stopping in epoch {epoch+1}.")
                            self.save_tracking_information(training_tracking)
                            break
                
                self.save_tracking_information(training_tracking)
            
