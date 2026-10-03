"""
This is the global model update function.
@author
- Van Tuan Nguyen (vantuan.nguyen@lqdtu.edu.vn)
- Razvan Beuran (razvan@jaist.ac.jp)
@create date 2023-12-14 11:41:08
"""

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.neighbors import KernelDensity
from tqdm import tqdm
from Utils import similarity_score
from DataLoader import IoTDataProccessor

import logging

# Configure the logging module
logging.basicConfig(level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                    format='%(asctime)s - %(levelname)s - %(message)s')

class GlobalAggregator(object):
    def __init__(self, model, update_type="avg"):
        """
            - initialize the SAE model in global: the model architecture: input-dim, 
                output-dim, latent-dim
            - 
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.update_type = update_type
        self.model = model.to(self.device)
    
    def create_dev_dataset(self, dataset):
        """
        Choose a development dataset for updating the global model (Fusion-based)
        
        params: 
            - dataset: (pytorch Dataset) the dataset concatenated from clients's dev sets.
        """
        logging.info("Creating development dataset...")
        self.dev_dataset = dataset["dataset"]
        # self.dev_label = dataset["label"]
        data_processor = IoTDataProccessor(scaler="standard")
        
        self.dev_dataset, self.dev_label = data_processor.fit_transform(self.dev_dataset)
        
        if self.update_type == "fusion_avg":
            self.dev_kde_scores = KernelDensity(kernel='gaussian', bandwidth="scott") \
                .fit(self.dev_dataset).score_samples(self.dev_dataset)
            
    def fusion_avg(self, local_models=None):
        """
        Perform fusion-based updating by calculating the average weights of local models.

        Args:
            local_models (list): List of local models.

        Returns:
            None
        """
        logging.info("Fusion-based updating...")
        update_weights = []
        weighted = []
        for i, local_model in zip(tqdm(range(len(local_models)), desc='Calculating similarity...'), local_models):
            self.model.load_state_dict(local_model)
            self.model.eval()
            with torch.no_grad():
                _, generated_data, _ = self.model(torch.Tensor(self.dev_dataset).to(self.device))
                sim_score = similarity_score(self.dev_kde_scores, generated_data.cpu().numpy())    # Calculate similarity score
                weighted.append(1/sim_score)    # Because sim_score close to 0 when more similar, so we use 1/sim_score
                update_weights.append((local_model, 1/sim_score))
        print(weighted)
        avg_weights = {}
        for key in update_weights[0][0].keys():
            avg_weights[key] = sum([w[key] * alpha  for w, alpha in update_weights]) \
                / sum([alpha for w, alpha in update_weights])
        self.model.load_state_dict(avg_weights)
        
    def fed_mse_avg(self, local_models=None):
        """
        Perform fusion-based updating by calculating the average weights of local models
        based on MSE loss of each AE-based model.

        Args:
            local_models (list): List of local models.

        Returns:
            None
        """
        logging.info("MSE-based updating...")
        update_weights = []
        weighted = []
        for i, local_model in zip(tqdm(range(len(local_models)), desc='Calculating similarity...'), local_models):
            self.model.load_state_dict(local_model[0])
            self.model.eval()
            with torch.no_grad():
                _, generated_data, _ = self.model(torch.Tensor(self.dev_dataset).to(self.device))
                sim_score = torch.nn.MSELoss(reduction='mean')(torch.Tensor(self.dev_dataset).to(self.device), generated_data)  # Calculate similarity score
                weighted.append(1/sim_score)    # Because sim_score close to 0 when more similar, so we use 1/sim_score
                update_weights.append((local_model[0], 1/sim_score))
        avg_weights = {}
        for key in update_weights[0][0].keys():
            avg_weights[key] = sum([w[key] * alpha  for w, alpha in update_weights]) \
                / sum([alpha for w, alpha in update_weights])
                
        self.model.load_state_dict(avg_weights)
    
    # def fed_mse_avg(self, local_models):
    #     update_weights = []
    #     weighted = []
    #     total_samples = sum(model[2] for model in local_models)

    #     for i, local_model in zip(tqdm(range(len(local_models)), desc='Calculating similarity...'), local_models):
    #         self.model.load_state_dict(local_model[0])
    #         self.model.eval()
    #         with torch.no_grad():
    #             _, generated_data, _ = self.model(torch.Tensor(self.dev_dataset).to(self.device))
    #             sim_score = torch.nn.MSELoss(reduction='mean')(torch.Tensor(self.dev_dataset).to(self.device), generated_data)  # Calculate similarity score
    #             weight = (1 / sim_score) * (local_model[2] / total_samples)  # Combine MSE and number of samples
    #             weighted.append(weight)
    #             update_weights.append((local_model[0], weight))

    #     avg_weights = {}
    #     for key in update_weights[0][0].keys():
    #         avg_weights[key] = sum([w[key] * alpha for w, alpha in update_weights]) \
    #             / sum([alpha for w, alpha in update_weights])

    #     self.model.load_state_dict(avg_weights)
    
    
    # def fed_mse_avg(self, local_models):
    #     update_weights = []
    #     mse_weights = []
    #     total_samples = sum(model[2] for model in local_models)

    #     for i, local_model in zip(tqdm(range(len(local_models)), desc='Calculating similarity...'), local_models):
    #         self.model.load_state_dict(local_model[0])
    #         self.model.eval()
    #         with torch.no_grad():
    #             _, generated_data, _ = self.model(torch.Tensor(self.dev_dataset).to(self.device))
    #             sim_score = torch.nn.MSELoss(reduction='mean')(torch.Tensor(self.dev_dataset).to(self.device), generated_data)  # Calculate similarity score
    #             mse_weight = 1 / sim_score  # Lower MSE should have higher weight
    #             mse_weights.append(mse_weight)
    #             update_weights.append((local_model[0], mse_weight, local_model[2]))

    #     # Normalize MSE weights
    #     mse_weights_sum = sum(mse_weights)
    #     normalized_mse_weights = [weight / mse_weights_sum for weight in mse_weights]

    #     # Calculate combined weights
    #     combined_weights = []
    #     for (local_model, normalized_mse_weight, num_samples) in zip(local_models, normalized_mse_weights, [model[2] for model in local_models]):
    #         combined_weight = normalized_mse_weight * (num_samples / total_samples)
    #         combined_weights.append(combined_weight)

    #     # Aggregate the global model using the combined weights
    #     avg_weights = {}
    #     for key in update_weights[0][0].keys():
    #         avg_weights[key] = sum(w[key] * weight for w, weight in zip([uw[0] for uw in update_weights], combined_weights))

    #     self.model.load_state_dict(avg_weights)
    
    def fed_avg(self, local_models=None):
        """
        Perform federated averaging to aggregate the weights of local models.

        Args:
            local_models (list): List of dictionaries representing the weights of local models.

        Returns:
            None
        """
        # avg_weights = {}
        # for key in local_models[0].keys():
        #     avg_weights[key] = sum([w[key] for w in local_models]) / len(local_models)
        
        # self.model.load_state_dict(avg_weights)
        
        # Sum the total number of samples across all clients
        total_samples = sum(model[2] for model in local_models)
        
        # Initialize the averaged weights dictionary
        avg_weights = {}

        # Iterate over each key in the state_dict of the first local model
        for key in local_models[0][0].keys():
            # Initialize the key in avg_weights
            avg_weights[key] = sum(model[0][key] * (model[2] / total_samples) for model in local_models)

        # Load the averaged weights into the global model
        self.model.load_state_dict(avg_weights)
    
    def fedprox(self, local_models=None, mu=0.01):
        """
        Perform federated optimization using the FedProx algorithm to aggregate the weights of local models.

        Args:
            local_models (list): List of dictionaries representing the weights of local models.
            mu (float): Proximal term coefficient.

        Returns:
            None
        """
        logging.info("FedProx updating...")
        # global_weights = self.model.state_dict()
        # update_weights = []
        # for local_model in local_models:
        #     for key in local_model.keys():
        #         local_model[key] -= mu * (local_model[key] - global_weights[key])
        #     update_weights.append(local_model)
        
        # avg_weights = {}
        # for key in global_weights.keys():
        #     avg_weights[key] = sum([w[key] for w in update_weights]) / len(update_weights)
        
        # self.model.load_state_dict(avg_weights)
        
        # Sum the total number of samples across all clients
        total_samples = sum(model[2] for model in local_models)
        
        # Initialize the averaged weights dictionary
        avg_weights = {}

        # Iterate over each key in the state_dict of the first local model
        for key in local_models[0][0].keys():
            # Initialize the key in avg_weights
            avg_weights[key] = sum(model[0][key] * (model[2] / total_samples) for model in local_models)

        # Load the averaged weights into the global model
        self.model.load_state_dict(avg_weights)
    
    def update(self, local_models=None):
        """
        Update the global model using the local models.

        Args:
            local_models (list): List of local models to be used for updating the global model.

        Returns:
            None
        """
        if self.update_type == "avg":
            self.fed_avg(local_models)
        if self.update_type == "fusion_avg":
            self.fusion_avg(local_models)
        if self.update_type == "mse_avg":
            self.fed_mse_avg(local_models)
        if self.update_type == "fedprox":
            self.fedprox(local_models, mu=0.01)
        
        self.model.eval()
        with torch.no_grad():
            _, _, val_loss = self.model(torch.Tensor(self.dev_dataset).to(self.device))
            self.val_loss = val_loss.item()
    
    
