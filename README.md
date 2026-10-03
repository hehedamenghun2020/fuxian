# Fed-IoTIDS
The implementation of paper: [*FedMSE: Semi-supservised federated learning for IoT network intrusion detection*](https://doi.org/10.1016/j.cose.2025.104337)

This repository includes the prepared data for all subnetworks in the IoT network and the source code for reproducing the experimental results
# Project structure

1. Data/ folder: Store data for experiments
2. Notebook/: Jupyter notebook for prepare data for training and testing. You can download the original dataset from [the official website here](https://archive.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot)

3. src/: Source code for experiments
    - Checkpoint/ --> Store the experimental report in json format and store latent data of SAE in each run
    - Configuration/ --> Store the config file to load the data
    - DataLoader/ --> Module for data preparation and preprocessing
    - Evaluator/ --> Testing phase module
    - Model/ --> Machine learning model definition
    - Trainer/ --> Modules for local training and global aggregation
    - Utils/ --> Utils for calculate the sim score
    - Visualization/ --> Jupyter notebook for visualizing the data and results

# How to run the experiments
1. Configure the data for experimental

```
config_file = f"Configuration/scen2-nba-iot-10clients.json"
```

2. Configure some hyper-parameters for experiment

```
num_participants = 0.5
epoch = 100
num_rounds = 20
lr_rate = 1e-5
shrink_lambda = 10
```

3. Install dependency libraries

```
pip install -r requirements.txt
```

4. Run the experiment

```
python main.py
```

# Citation

If you use this code or find our work useful for your research, please cite our paper:

```
Nguyen, V.T., Beuran, R. (2025). FedMSE: Semi-supervised federated learning approach for IoT network intrusion detection. Computers & Security, 151, 104337. 
https://doi.org/10.1016/j.cose.2025.104337
```

BibTeX:
```bibtex
@article{NGUYEN2025104337,
  title = {FedMSE: Semi-supervised federated learning approach for IoT network intrusion detection},
  journal = {Computers & Security},
  volume = {151},
  pages = {104337},
  year = {2025},
  issn = {0167-4048},
  doi = {https://doi.org/10.1016/j.cose.2025.104337},
  url = {https://www.sciencedirect.com/science/article/pii/S0167404825000264},
  author = {Van Tuan Nguyen and Razvan Beuran},
  keywords = {Internet of things, Intrusion detection system, Machine learning, Federated learning},
  abstract = {This paper proposes a novel federated learning approach for improving IoT network intrusion detection. The rise of IoT has expanded the cyber attack surface, making traditional centralized machine learning methods insufficient due to concerns about data availability, computational resources, transfer costs, and especially privacy preservation. A semi-supervised federated learning model was developed to overcome these issues, combining the Shrink Autoencoder and Centroid one-class classifier (SAE-CEN). This approach enhances the performance of intrusion detection by effectively representing normal network data and accurately identifying anomalies in the decentralized strategy. Additionally, a mean square error-based aggregation algorithm (MSEAvg) was introduced to improve global model performance by prioritizing more accurate local models. The results obtained in our experimental setup, which uses various settings relying on the N-BaIoT dataset and Dirichlet distribution, demonstrate significant improvements in real-world heterogeneous IoT networks in detection accuracy from 93.98 ± 2.90 to 97.30 ± 0.49, reduced learning costs when requiring only 50% of gateways participating in the training process, and robustness in large-scale networks.}
}
```

