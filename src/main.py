"""
This is training endpoint.
@author
- Van Tuan Nguyen (vantuan.nguyen@lqdtu.edu.vn)
- Razvan Beuran (razvan@jaist.ac.jp)
@create date 2023-12-11 00:28:29
"""

import os
import json
import pickle   
import pandas as pd
import numpy as np
import torch
import argparse
import copy
import random
import json
import time
from torch.utils.data import DataLoader, random_split, ConcatDataset
from Model import Shrink_Autoencoder
from Model import CA_Shrink_Autoencoder
from Model import SE_Shrink_Autoencoder
from Model import Autoencoder
from Model import ECA_GF_Autoencoder
from Model import ECA_AE_Autoencoder
from Model import GF_AE_Autoencoder
from DataLoader import load_data
from DataLoader import IoTDataset
from DataLoader import IoTDataProccessor
from Trainer import ClientTrainer
from Trainer import GlobalAggregator
from Evaluator import Evaluator

import logging

# Configure the logging module
logging.basicConfig(level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
                    format='%(asctime)s - %(levelname)s - %(message)s')


num_participants = 0.5
epoch = 100
num_rounds = 20
lr_rate = 1e-5
shrink_lambda = 10
noise_sigma = 0.06
beta = 0.5
network_size = 10
data_seed = 1234
# no_Exp = f"nonIID_Exp1_Rerun_{epoch}epoch_10client_lr0001_lamda{shrink_lambda}_ratio{num_participants*100}"
no_Exp = f"SAE_CEN_nonIID_Exp_{epoch}epoch_{network_size}client_{num_rounds}rounds_lr{lr_rate}_lambda{shrink_lambda}_ratio{num_participants*100}_dataseed{data_seed}"

num_runs = 1
batch_size = 12

new_device = True
min_val_loss = float("inf")
global_patience = 1
global_worse = 0
metric = "AUC" #AUC or classification
# model_type = "autoencoder"   #autoencoder; hybrid;
# update_type = "mse_avg"  #avg; fusion_avg; mse_avg
dim_features = 115   #nba-iot: 115; cic-2023: 46

scen_name = 'FL-IoT' 

#config_file = f"Configuration/scen2-nba-iot-50clients.json"
config_file = "Configuration/scen2-nba-iot-10clients-noniid.json"
# config_file = "Configuration/cic-config.json"

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_experiment_name(model_type):
    exp_names = {
        "autoencoder": "AE-MSE",
        "ae_cen": "AE-CEN",
        "hybrid": "SAE-CEN",
        "hybrid_ca": "CA-SAE-CEN",
        "hybrid_se": "SE-SAE-CEN",
        "eca_ae_cen": "ECA-AE-CEN",
        "gf_ae_cen": "GF-AE-CEN",
        "hybrid_eca_gf_ae": "ECA-GF-AE-CEN",
        "eca_gf_ae_mse": "ECA-GF-AE-MSE",
        "hybrid_eca_gf_ae_shrink": "ECA-GF-AE-CEN-shrink",
    }
    return exp_names.get(model_type, model_type)

def get_score_type(model_type):
    if model_type in ["autoencoder", "eca_gf_ae_mse"]:
        return "reconstruction MSE"
    if model_type == "ae_cen":
        return "latent + CEN distance"
    if model_type == "hybrid_eca_gf_ae":
        return "fused latent + CEN distance"
    if model_type == "hybrid_eca_gf_ae_shrink":
        return "fused latent + CEN distance"
    return "latent + CEN distance"

def get_score_label(score_type):
    return "MSE" if score_type == "reconstruction MSE" else "CEN"

def get_eval_model_type(model_type):
    return "autoencoder" if get_score_type(model_type) == "reconstruction MSE" else "hybrid"

def build_global_model(model_type):
    if model_type in ["autoencoder", "ae_cen"]:
        return Autoencoder(input_dim=dim_features,
                           output_dim=dim_features,
                           latent_dim=11,
                           hidden_neus=50)
    if model_type == "hybrid":
        return Shrink_Autoencoder(input_dim=dim_features,
                                  output_dim=dim_features,
                                  shrink_lambda=shrink_lambda,
                                  latent_dim=11,
                                  hidden_neus=50)
    if model_type == "hybrid_ca":
        return CA_Shrink_Autoencoder(input_dim=dim_features,
                                     output_dim=dim_features,
                                     shrink_lambda=shrink_lambda,
                                     latent_dim=11,
                                     hidden_neus=50,
                                     beta=beta)
    if model_type == "hybrid_se":
        return SE_Shrink_Autoencoder(input_dim=dim_features,
                                     output_dim=dim_features,
                                     shrink_lambda=shrink_lambda,
                                     latent_dim=11,
                                     hidden_neus=50,
                                     reduction=16)
    if model_type in ["hybrid_eca_gf_ae", "eca_gf_ae_mse"]:
        return ECA_GF_Autoencoder(input_dim=dim_features,
                                  output_dim=dim_features,
                                  latent_dim=11,
                                  hidden_neus=50,
                                  eca_k_size=3,
                                  noise_sigma=noise_sigma)
    if model_type == "hybrid_eca_gf_ae_shrink":
        return ECA_GF_Autoencoder(input_dim=dim_features,
                                  output_dim=dim_features,
                                  latent_dim=11,
                                  hidden_neus=50,
                                  eca_k_size=3,
                                  noise_sigma=noise_sigma,
                                  shrink_lambda=shrink_lambda)
    if model_type == "eca_ae_cen":
        return ECA_AE_Autoencoder(input_dim=dim_features,
                                  output_dim=dim_features,
                                  latent_dim=11,
                                  hidden_neus=50,
                                  eca_k_size=3)
    if model_type == "gf_ae_cen":
        return GF_AE_Autoencoder(input_dim=dim_features,
                                 output_dim=dim_features,
                                 latent_dim=11,
                                 hidden_neus=50)
    raise ValueError(f"Unsupported model_type: {model_type}")

def count_trainable_params(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)

def safe_float(value):
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)

def print_and_log(text):
    print(text)
    logging.info("\n" + text)

def print_round_report(exp_name, score_type, update_type, run, round_num,
                       client_auc_scores, mean_auc, std_auc, global_loss,
                       selected_idx, params, train_time_sec, inference_times):
    score_label = get_score_label(score_type)
    lines = [
        f"Exp: {exp_name} | score={score_label} | update={update_type} | run={run} | round={round_num}",
        f"score_type = {score_type}",
        f"Params: {params:,}",
        "Client AUCs:",
    ]
    for client_name, auc_score in client_auc_scores.items():
        lines.append(f"  {client_name}: {auc_score:.6f}")
    avg_inference_time_ms = 0.0
    if inference_times:
        avg_inference_time_ms = float(np.mean(list(inference_times.values())) * 1000.0)
    lines.extend([
        f"Mean AUC: {mean_auc:.6f}",
        f"Std AUC: {std_auc:.6f}",
        f"Global loss: {global_loss:.6f}",
        f"Join clients: {selected_idx}",
        f"Train time per round: {train_time_sec:.2f} s",
        f"Inference time per client: {avg_inference_time_ms:.2f} ms",
    ])
    print_and_log("\n".join(lines))

def print_run_summary(run, exp_name, score_type, update_type, best_summary, final_summary):
    lines = [
        f"Run {run} Summary:",
        f"Exp: {exp_name} | score={get_score_label(score_type)} | update={update_type}",
        f"score_type = {score_type}",
        f"Best round: round_{best_summary['round']}",
        "Best round selection: min global_loss",
        f"Best mean AUC: {best_summary['mean_auc']:.6f}",
        f"Best round std AUC: {best_summary['std_auc']:.6f}",
        f"Best round global loss: {best_summary['global_loss']:.6f}",
        f"Final round: round_{final_summary['round']}",
        f"Final round mean AUC: {final_summary['mean_auc']:.6f}",
        f"Final round std AUC: {final_summary['std_auc']:.6f}",
        f"Final round global loss: {final_summary['global_loss']:.6f}",
    ]
    print_and_log("\n".join(lines))

def print_final_summary(exp_name, score_type, update_type, dataset_label, run_auc_summary):
    final_mean_aucs = [item["final_mean_auc"] for item in run_auc_summary]
    final_std_aucs = [item["final_std_auc"] for item in run_auc_summary]
    best_mean_aucs = [item["best_mean_auc"] for item in run_auc_summary]
    lines = [
        "========== Final Summary ==========",
        f"Model: {exp_name}",
        f"Score type: {score_type}",
        f"Update: {update_type}",
        f"Dataset: {dataset_label}",
        f"Runs: {len(run_auc_summary)}",
        "",
        "Run final mean AUCs:",
    ]
    for item in run_auc_summary:
        lines.append(f"Run {item['run']}: {item['final_mean_auc']:.6f}")
    lines.extend([
        "",
        f"Overall final AUC: {float(np.mean(final_mean_aucs)):.6f} +/- {float(np.std(final_mean_aucs)):.6f}",
        f"Average client std: {float(np.mean(final_std_aucs)):.6f}",
        f"Best round AUC: {float(np.mean(best_mean_aucs)):.6f} +/- {float(np.std(best_mean_aucs)):.6f}",
        "Best rounds selected by min global_loss:",
    ])
    for item in run_auc_summary:
        lines.append(f"Run {item['run']}: round_{item['best_round']}")
    lines.append("===================================")
    print_and_log("\n".join(lines))

if __name__ == "__main__":
        random.seed(data_seed)
        np.random.seed(data_seed)
        try:
            logging.info("Loading configuration...")
            with open(config_file, "r") as config_file:
                config = json.load(config_file)
        except Exception as e:
            logging.exception("Failed to load configuration.")
            raise
        
        devices_list = random.sample(config['devices_list'], network_size)
        # devices_list = config['devices_list']
        client_info = []
        # random.seed(data_seed)
        # np.random.seed(data_seed)
        for device in devices_list:
            logging.info("Creating metadata for client...")
            normal_data_path = os.path.join(config['data_path'], device["normal_data_path"])
            abnormal_data_path = os.path.join(config['data_path'], device["abnormal_data_path"])
            test_new_normal_data_path = os.path.join(config['data_path'], device["test_normal_data_path"])
            
            logging.info("Loading data from {}...".format(device['name']))
            
            # normal_data = load_data(normal_data_path, header="infer")
            normal_data = load_data(normal_data_path)
            normal_data = normal_data.sample(frac=1).reset_index(drop=True)
            # abnormal_data = load_data(abnormal_data_path, header="infer")

            abnormal_data = load_data(abnormal_data_path)
            abnormal_data = abnormal_data.sample(frac=1).reset_index(drop=True)
            
            # new normal data from new devices
            if new_device:
                new_normal_data = load_data(test_new_normal_data_path)
            
            device_name = device['name']
            print(f"{device_name} has {len(normal_data)} normal data and {len(abnormal_data)} abnormal data")
            # now, need to split data before normalization
            train_normal_size = int(0.4 * len(normal_data))
            valid_normal_size = int(0.1 * len(normal_data))
            dev_normal_size = int(0.4 * len(normal_data))
            test_normal_size = len(normal_data) - train_normal_size - valid_normal_size - dev_normal_size
            
            train_normal_data = normal_data[:train_normal_size]
            valid_normal_data = normal_data[train_normal_size:train_normal_size+valid_normal_size]
            dev_normal_data = normal_data[train_normal_size+valid_normal_size:train_normal_size+valid_normal_size+dev_normal_size]
            test_normal_data = normal_data[train_normal_size+valid_normal_size+dev_normal_size:]

            data_processor = IoTDataProccessor(scaler="standard")
            processed_train_data, train_label = data_processor.fit_transform(train_normal_data)
            processed_valid_data, valid_label = data_processor.transform(valid_normal_data)
            # processed_dev_data, dev_label = data_processor.transform(dev_normal_data)
            processed_test_data, test_label = data_processor.transform(test_normal_data)
            processed_abnormal_data, abnormal_label = data_processor.transform(abnormal_data, type="abnormal")
            
            if new_device:
                processed_new_normal_data, new_normal_label = data_processor.transform(new_normal_data)
                processed_test_data = np.concatenate([processed_test_data, processed_new_normal_data], axis=0)
                processed_test_label = np.concatenate([test_label, new_normal_label], axis=0)
                test_dataset = IoTDataset(processed_test_data, processed_test_label)
            else:
                test_dataset = IoTDataset(processed_test_data, test_label)
            
            train_dataset = IoTDataset(processed_train_data, train_label)
            valid_dataset = IoTDataset(processed_valid_data, valid_label)
            # dev_dataset = IoTDataset(processed_dev_data, dev_label)
            
            
            # indices = np.random.choice(processed_abnormal_data.shape[0], 3000, replace=False)
            # unique_values, counts = np.unique(abnormal_label[indices], return_counts=True)
            # print(f"Abnormal data: {unique_values} - {counts}")
            # abnormal_dataset = IoTDataset(processed_abnormal_data[indices], abnormal_label[indices])
            abnormal_dataset = IoTDataset(processed_abnormal_data, abnormal_label)
            
            test_dataset = ConcatDataset([test_dataset, abnormal_dataset])

            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                pin_memory=True
            )
            valid_loader = DataLoader(
                dataset=valid_dataset,
                batch_size=batch_size,
                pin_memory=True
            )
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=batch_size,
                pin_memory=True
            )
            
            # indices = np.random.choice(processed_dev_data.shape[0], 200, replace=False)
            client_info.append({
                "device": device['name'],
                "save_dir": "",
                "train_loader": train_loader,
                "valid_loader": valid_loader,
                "test_loader": test_loader,
                "test_dataset": (processed_test_data, test_label),
                "dev_normal_dataset": dev_normal_data
            })
        for update_type in ["mse_avg"]:
        # for update_type in ["fedprox"]:
        # for update_type in ["mse_avg"]:
            # for model_type in ["autoencoder"]:
            model_auc_comparison = []
            for model_type in ["hybrid"]:
                exp_name = get_experiment_name(model_type)
                score_type = get_score_type(model_type)
                dataset_label = f"N-BaIoT non-IID {network_size} clients"
                run_auc_summary = []
                for run in range(num_runs):
                    set_seeds(run*10000)
                    for client in client_info:
                        client['save_dir'] = os.path.join(f"Checkpoint/{network_size}/{no_Exp}/{run}/ClientModel", scen_name, model_type, update_type, client['device'])
                    global_worse = 0
                    min_val_loss = float("inf")
                    if True:
                        # random.seed(run*10000)
                        
                        # devices_list = config['devices_list']

                        directory = f'Checkpoint/Results/Update/{network_size}/{no_Exp}/Run_{run}/{metric}'
                        if not os.path.exists(directory):
                            os.makedirs(directory)

                        # Check if the file exists and delete its content if it does
                        filename = f'{directory}/{scen_name}_{num_participants}_{model_type}_{update_type}_results.json'
                        open(filename, 'w').close()
                        
                        if get_eval_model_type(model_type) == "hybrid":
                            global_model = build_global_model(model_type)
                            
                            global_aggregator = GlobalAggregator(global_model, update_type=update_type)
                            params = count_trainable_params(global_aggregator.model)
                            
                            # Calculate the minimum length of all clients' datasets
                            min_len = min([len(client['dev_normal_dataset']) for client in client_info])

                            # Sample min_len data points from each client's dataset and create dev_dataset
                            dev_dataset = []
                            for client in client_info:
                                sample_data = client['dev_normal_dataset'].sample(n=min_len)
                                dev_dataset.append(sample_data)
                                # client['dev_normal_dataset'] = client['dev_normal_dataset'].drop(sample_data.index)

                            # Concatenate all the sampled data into a single numpy array
                            dev_dataset = np.concatenate(dev_dataset, axis=0)

                            global_aggregator.create_dev_dataset({"dataset": dev_dataset})
                            
                            # Now all clients' datasets have the same size
                            
                            # indices = np.random.choice(processed_dev_data.shape[0], 200, replace=False)
                            # dev_dataset = np.concatenate([client['dev_normal_dataset'][0] for client in client_info], axis=0)
                            # dev_label = np.concatenate([client['dev_normal_dataset'][1] for client in client_info], axis=0)
                            # global_aggregator.create_dev_dataset({"dataset": dev_dataset, "label": dev_label})
                            
                            # dev_dataset = np.concatenate([client['dev_normal_dataset'][0][indices] for client in client_info], axis=0)
                            # dev_label = np.concatenate([client['dev_normal_dataset'][1][indices] for client in client_info], axis=0)
                            # global_aggregator.create_dev_dataset({"dataset": dev_dataset, "label": dev_label})
                        
                            # global_test_data = np.concatenate([client['test_dataset'][0] for client in client_info], axis=0)
                            # global_test_label = np.concatenate([client['test_dataset'][1] for client in client_info], axis=0)
                            # global_test_dataset = IoTDataset(global_test_data, global_test_label)
                            # global_test_dataloader = DataLoader(
                            #     dataset=global_test_dataset,
                            #     batch_size=batch_size,
                            #     pin_memory=True
                            # )
                            
                            # Start training process
                            results = []
                            client_latent = {}
                            last_round = None
                            last_mean_auc = None
                            last_std_auc = None
                            round_summaries = []
                            for round in range(num_rounds):
                                client_latent[round] = {}
                                dev_dataset = []
                                dev_label = []
                                selected_idx = random.sample([i for i in range(len(client_info))], int(num_participants*len(client_info)))
                                selected_clients = [client_info[i] for i in selected_idx]
                                
                                total_training_samples = sum([len(client['train_loader'].dataset) for client in selected_clients])
                                
                                # for client in client_info:
                                #     # indices = np.random.choice(client['dev_normal_dataset'].shape[0], 50, replace=False)
                                #     n_samples = min(20, len(client['dev_normal_dataset']))
                                #     sample_data = client['dev_normal_dataset'].sample(n=n_samples)
                                #     dev_dataset.append(sample_data)
                                #     client['dev_normal_dataset'] = client['dev_normal_dataset'].drop(sample_data.index)

                                # dev_dataset = np.concatenate(dev_dataset, axis=0)
                                # dev_label = np.concatenate(dev_label, axis=0)
                                # dev_dataset = np.concatenate([client['dev_normal_dataset'] for client in client_info], axis=0)
                                # global_aggregator.create_dev_dataset({"dataset": dev_dataset, "label": dev_label})
                                # global_aggregator.create_dev_dataset({"dataset": dev_dataset})
                                # Choose clients to train
                                # random.seed(round*1234)
                                # num_participants = random.uniform(0,1)
                                
                                client_weights = []
                                round_train_start = time.time()
                                round_global_state = copy.deepcopy(
                                    global_aggregator.model.state_dict()
                                )
                                for i, client in enumerate(selected_clients):
                                    logging.info("Training local model...")
                                    local_model = build_global_model(model_type)
                                    local_model.load_state_dict(round_global_state)
                                    device_trainer = ClientTrainer(model=local_model, \
                                        save_dir=client['save_dir'], epoch=epoch, lr_rate=lr_rate, update_type=update_type)
                                    
                                    device_trainer.run(client["train_loader"], client["valid_loader"])
                                    client_weights.append((copy.deepcopy(device_trainer.model.state_dict()), total_training_samples, len(client["train_loader"].dataset)))
                                    logging.info(f"Client {i} training done!")
                                    
                                # client_weights = random.sample(client_weights, int(num_participants * len(client_weights)))
                                global_aggregator.update(local_models=client_weights)

                                if model_type == "hybrid_ca":
                                    global_aggregator.model.eval()
                                    with torch.no_grad():
                                        dev_tensor = torch.Tensor(global_aggregator.dev_dataset).to(
                                            torch.device("cuda" if torch.cuda.is_available() else "cpu")
                                        )
                                        global_latent, _, _ = global_aggregator.model(dev_tensor)
                                        global_center = global_latent.mean(dim=0)
                                        global_aggregator.model.set_global_center(global_center)
                                train_time_sec = time.time() - round_train_start

                                logging.info(f"Round {round+1}/{num_rounds} - Updated global model - \
                                    Global loss: {global_aggregator.val_loss}")
                                
                                logging.info("Training done! Evaluating...")
                                # evaluate the model in clients
                            
                                evaluator = Evaluator(global_aggregator.model, metric=metric, model_type="hybrid")
                                round_results = {}
                                inference_times = {}
                                
                                for i, client in enumerate(client_info):
                                    logging.info(f"Evaluating client {i} - name: {client['device']}")
                                    eval_start = time.time()
                                    auc_score, test_latent, test_label = evaluator.evaluate(client["test_loader"], client["train_loader"])
                                    inference_times[client['device']] = time.time() - eval_start
                                    round_results[client['device']] = auc_score
                                    # store latent of SAE and SAE_MSEFed
                                    client_latent[round][client['device']] = (test_latent, test_label)
                                client_auc_by_name = dict(round_results)
                                client_auc_scores = list(client_auc_by_name.values())
                                mean_auc = float(np.mean(client_auc_scores))
                                std_auc = float(np.std(client_auc_scores))
                                global_loss = safe_float(global_aggregator.val_loss)
                                last_round = round + 1
                                last_mean_auc = mean_auc
                                last_std_auc = std_auc
                                print_round_report(
                                    exp_name=exp_name,
                                    score_type=score_type,
                                    update_type=update_type,
                                    run=run,
                                    round_num=round + 1,
                                    client_auc_scores=client_auc_by_name,
                                    mean_auc=mean_auc,
                                    std_auc=std_auc,
                                    global_loss=global_loss,
                                    selected_idx=selected_idx,
                                    params=params,
                                    train_time_sec=train_time_sec,
                                    inference_times=inference_times,
                                )
                                round_summary = {
                                    "round": round + 1,
                                    "mean_auc": mean_auc,
                                    "std_auc": std_auc,
                                    "global_loss": global_loss,
                                    "join_clients": selected_idx,
                                }
                                round_summaries.append(round_summary)
                                round_results["mean_auc"] = mean_auc
                                round_results["std_auc"] = std_auc
                                round_results["global_loss"] = global_loss
                                round_results['join_clients'] = selected_idx
                                round_results["score_type"] = score_type
                                round_results["params"] = params
                                round_results["train_time_sec"] = train_time_sec
                                round_results["inference_time_per_client_sec"] = inference_times
                                round_results = {f'round_{round+1}': round_results}
                                
                                # Append to the JSON file
                                with open(filename, 'a') as f:
                                    f.write(json.dumps(round_results) + '\n')
                                
                                if global_aggregator.val_loss < min_val_loss:
                                    min_val_loss = global_aggregator.val_loss
                                    global_worse = 0
                                
                                if global_aggregator.val_loss >= min_val_loss:
                                    global_worse += 1
                                    if global_worse > global_patience:
                                        logging.info("Early stopping in global round!")
                                        break
                            # store latent data of SAE and SAE_MSEFed for all rounds
                            # Define the file path
                            file_path = f'Checkpoint/LatentData/{network_size}/{no_Exp}/Run_{run}/latent_{model_type}_{update_type}.pkl'

                            # Create the directory if it does not exist
                            os.makedirs(os.path.dirname(file_path), exist_ok=True)

                            # Now you can safely write the file
                            with open(file_path, 'wb') as f:
                                pickle.dump(client_latent, f)

                            if round_summaries:
                                best_summary = min(round_summaries, key=lambda item: item["global_loss"])
                                final_summary = round_summaries[-1]
                                print_run_summary(
                                    run=run,
                                    exp_name=exp_name,
                                    score_type=score_type,
                                    update_type=update_type,
                                    best_summary=best_summary,
                                    final_summary=final_summary,
                                )
                                run_auc_summary.append({
                                    "run": run,
                                    "best_round": best_summary["round"],
                                    "best_mean_auc": best_summary["mean_auc"],
                                    "best_std_auc": best_summary["std_auc"],
                                    "best_global_loss": best_summary["global_loss"],
                                    "final_round": final_summary["round"],
                                    "final_mean_auc": final_summary["mean_auc"],
                                    "final_std_auc": final_summary["std_auc"],
                                    "final_global_loss": final_summary["global_loss"],
                                    "mean_auc": final_summary["mean_auc"],
                                    "std_auc": final_summary["std_auc"],
                                })
                            
                        if get_eval_model_type(model_type) == "autoencoder":
                            global_model = build_global_model(model_type)
                            
                            global_aggregator = GlobalAggregator(global_model, update_type=update_type)
                            params = count_trainable_params(global_aggregator.model)
                            # Calculate the minimum length of all clients' datasets
                            min_len = min([len(client['dev_normal_dataset']) for client in client_info])

                            # Sample min_len data points from each client's dataset and create dev_dataset
                            dev_dataset = []
                            for client in client_info:
                                sample_data = client['dev_normal_dataset'].sample(n=min_len)
                                dev_dataset.append(sample_data)
                                # client['dev_normal_dataset'] = client['dev_normal_dataset'].drop(sample_data.index)

                            # Concatenate all the sampled data into a single numpy array
                            dev_dataset = np.concatenate(dev_dataset, axis=0)

                            global_aggregator.create_dev_dataset({"dataset": dev_dataset})
                            
                            # dev_dataset = np.concatenate([client['dev_normal_dataset'][0] for client in client_info], axis=0)
                            # dev_label = np.concatenate([client['dev_normal_dataset'][1] for client in client_info], axis=0)
                            # global_aggregator.create_dev_dataset({"dataset": dev_dataset, "label": dev_label})
                            
                            
                            
                            # global_test_data = np.concatenate([client['test_dataset'][0] for client in client_info], axis=0)
                            # global_test_label = np.concatenate([client['test_dataset'][1] for client in client_info], axis=0)
                            # global_test_dataset = IoTDataset(global_test_data, global_test_label)
                            # global_test_dataloader = DataLoader(
                            #     dataset=global_test_dataset,
                            #     batch_size=batch_size,
                            #     pin_memory=True
                            # )
                            
                            # Start training process
                            results = []
                            last_round = None
                            last_mean_auc = None
                            last_std_auc = None
                            round_summaries = []
                            for round in range(num_rounds):
                                dev_dataset = []
                                dev_label = []
                                dev_dataset = []
                                dev_label = []
                                # selected_idx = random.sample([i for i in range(len(client_info))], int(num_participants*len(client_info)))
                                # selected_clients = [client_info[i] for i in selected_idx]
                                # for client in client_info:
                                #     # indices = np.random.choice(client['dev_normal_dataset'].shape[0], 50, replace=False)
                                #     n_samples = min(20, len(client['dev_normal_dataset']))
                                #     sample_data = client['dev_normal_dataset'].sample(n=n_samples)
                                #     dev_dataset.append(sample_data)
                                #     client['dev_normal_dataset'] = client['dev_normal_dataset'].drop(sample_data.index)
                                
                                # dev_dataset = np.concatenate(dev_dataset, axis=0)
                                # dev_label = np.concatenate(dev_label, axis=0)
                                # dev_dataset = np.concatenate([client['dev_normal_dataset'] for client in client_info], axis=0)
                                # global_aggregator.create_dev_dataset({"dataset": dev_dataset, "label": dev_label})
                                # global_aggregator.create_dev_dataset({"dataset": dev_dataset})
                                
                                # Choose clients to train
                                # random.seed(round*1234)
                                # num_participants = random.uniform(0,1)
                                
                                selected_idx = random.sample([i for i in range(len(client_info))], int(num_participants*len(client_info)))
                                selected_clients = [client_info[i] for i in selected_idx]
                                
                                total_training_samples = sum([len(client['train_loader'].dataset) for client in selected_clients])
                                
                                client_weights = []
                                round_train_start = time.time()
                                round_global_state = copy.deepcopy(
                                    global_aggregator.model.state_dict()
                                )
                                for i, client in enumerate(selected_clients):
                                    logging.info("Training local model...")
                                    local_model = build_global_model(model_type)
                                    local_model.load_state_dict(round_global_state)
                                    device_trainer = ClientTrainer(model=local_model, \
                                        save_dir=client['save_dir'], epoch=epoch, update_type=update_type, lr_rate=lr_rate)
                                    device_trainer.run(client["train_loader"], client["valid_loader"])
                                    # client_weights.append(copy.deepcopy(device_trainer.model.state_dict()))
                                    client_weights.append((copy.deepcopy(device_trainer.model.state_dict()), total_training_samples, len(client["train_loader"].dataset)))
                                    logging.info(f"Client {i} training done!")
                                
                                logging.info(f"Round {round+1}/{num_rounds} - Updating global model")
                                
                                # client_weights = random.sample(client_weights, int(num_participants * len(client_weights)))
                                global_aggregator.update(local_models=client_weights)
                                train_time_sec = time.time() - round_train_start

                                logging.info(f"Round {round+1}/{num_rounds} - Updated global model - \
                                    Global loss: {global_aggregator.val_loss}")
                                
                                logging.info("Training done! Evaluating...")
                                # evaluate the model in clients
                            
                                evaluator = Evaluator(global_aggregator.model, metric=metric, model_type=get_eval_model_type(model_type))
                                round_results = {}
                                inference_times = {}
                                for i, client in enumerate(client_info):
                                    logging.info(f"Evaluating client {i} - name: {client['device']}")
                                    eval_start = time.time()
                                    auc_score = evaluator.evaluate(client["test_loader"], client["train_loader"])
                                    inference_times[client['device']] = time.time() - eval_start
                                    round_results[client['device']] = auc_score
                                client_auc_by_name = dict(round_results)
                                client_auc_scores = list(client_auc_by_name.values())
                                mean_auc = float(np.mean(client_auc_scores))
                                std_auc = float(np.std(client_auc_scores))
                                global_loss = safe_float(global_aggregator.val_loss)
                                last_round = round + 1
                                last_mean_auc = mean_auc
                                last_std_auc = std_auc
                                print_round_report(
                                    exp_name=exp_name,
                                    score_type=score_type,
                                    update_type=update_type,
                                    run=run,
                                    round_num=round + 1,
                                    client_auc_scores=client_auc_by_name,
                                    mean_auc=mean_auc,
                                    std_auc=std_auc,
                                    global_loss=global_loss,
                                    selected_idx=selected_idx,
                                    params=params,
                                    train_time_sec=train_time_sec,
                                    inference_times=inference_times,
                                )
                                round_summary = {
                                    "round": round + 1,
                                    "mean_auc": mean_auc,
                                    "std_auc": std_auc,
                                    "global_loss": global_loss,
                                    "join_clients": selected_idx,
                                }
                                round_summaries.append(round_summary)
                                round_results["mean_auc"] = mean_auc
                                round_results["std_auc"] = std_auc
                                round_results["global_loss"] = global_loss
                                round_results['join_clients'] = selected_idx
                                round_results["score_type"] = score_type
                                round_results["params"] = params
                                round_results["train_time_sec"] = train_time_sec
                                round_results["inference_time_per_client_sec"] = inference_times
                                round_results = {f'round_{round+1}': round_results}

                                # Append to the JSON file
                                with open(filename, 'a') as f:
                                    f.write(json.dumps(round_results) + '\n')

                                if global_aggregator.val_loss < min_val_loss:
                                    min_val_loss = global_aggregator.val_loss
                                    global_worse = 0

                                if global_aggregator.val_loss >= min_val_loss:
                                    global_worse += 1
                                    if global_worse > global_patience:
                                        logging.info("Early stopping in global round!")
                                        break

                            if round_summaries:
                                best_summary = min(round_summaries, key=lambda item: item["global_loss"])
                                final_summary = round_summaries[-1]
                                print_run_summary(
                                    run=run,
                                    exp_name=exp_name,
                                    score_type=score_type,
                                    update_type=update_type,
                                    best_summary=best_summary,
                                    final_summary=final_summary,
                                )
                                run_auc_summary.append({
                                    "run": run,
                                    "best_round": best_summary["round"],
                                    "best_mean_auc": best_summary["mean_auc"],
                                    "best_std_auc": best_summary["std_auc"],
                                    "best_global_loss": best_summary["global_loss"],
                                    "final_round": final_summary["round"],
                                    "final_mean_auc": final_summary["mean_auc"],
                                    "final_std_auc": final_summary["std_auc"],
                                    "final_global_loss": final_summary["global_loss"],
                                    "mean_auc": final_summary["mean_auc"],
                                    "std_auc": final_summary["std_auc"],
                                })

                if run_auc_summary:
                    print_final_summary(
                        exp_name=exp_name,
                        score_type=score_type,
                        update_type=update_type,
                        dataset_label=dataset_label,
                        run_auc_summary=run_auc_summary,
                    )
                    final_mean_aucs = [item["final_mean_auc"] for item in run_auc_summary]
                    final_std_aucs = [item["final_std_auc"] for item in run_auc_summary]
                    model_auc_comparison.append({
                        "model_type": model_type,
                        "model_name": exp_name,
                        "score": get_score_label(score_type),
                        "score_type": score_type,
                        "run_final_mean_aucs": final_mean_aucs,
                        "mean_auc": float(np.mean(final_mean_aucs)),
                        "std_auc": float(np.std(final_mean_aucs)),
                        "avg_client_std": float(np.mean(final_std_aucs)),
                    })

            if model_auc_comparison:
                max_runs = max(len(item["run_final_mean_aucs"]) for item in model_auc_comparison)
                run_headers = [f"Run{i}" for i in range(max_runs)]
                comparison_lines = [
                    "========== Final Model Comparison ==========",
                    f"Update: {update_type}",
                    " | ".join(["Model", "Score", *run_headers, "Mean+/-Std", "Avg Client Std"]),
                ]
                for item in model_auc_comparison:
                    run_values = [f"{value:.6f}" for value in item["run_final_mean_aucs"]]
                    run_values.extend(["-"] * (max_runs - len(run_values)))
                    comparison_lines.append(
                        " | ".join([
                            item["model_name"],
                            item["score"],
                            *run_values,
                            f"{item['mean_auc']:.6f}+/-{item['std_auc']:.6f}",
                            f"{item['avg_client_std']:.6f}",
                        ])
                    )
                comparison_lines.append("==========================================")
                comparison_text = "\n".join(comparison_lines)
                logging.info("\n" + comparison_text)
                print(comparison_text)
                                    
