This project contains three main training pipelines and one baseline framework for glycemic forecasting using the OhioT1DM dataset:

Included Scripts:
- Pipeline_1_TripleR.py
- Pipeline_2_TripleR.py
- Pipeline_3_TripleR.py
- Triple_R_Base.py (baseline framework without the patient graph / similarity matrix)

Dataset Structure:
To run any of the pipelines, place the OhioT1DM dataset inside the folder ohio_datasets/.
The dataset must include two subfolders:
- Train/    → containing XML files for training patients
- Test/     → containing XML files for test patients

Similarity Matrix:
A CSV file named Similarity_matrix_Ohio.csv containing the patient similarity matrix must also be present in the Dati/ directory.
A sample version of this file is already included for testing purposes.

Cluster Configuration:
If using Pipeline 1 or Pipeline 2, you must also provide a file named Clusters.npy in the Dati/ directory.
This file is a NumPy array containing the cluster index for each patient, for example:
[0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1]
where 0 indicates Cluster 1 and 1 indicates Cluster 2.

NOTE:
Inside Pipeline_1_TripleR.py and Pipeline_2_TripleR.py, you must manually assign the patient indices to the variables Cluster1 and Cluster2.
These same index assignments must also be modified in the script located at:
utils_model/model_funcs.py
(specifically in the variables "cluster1" and "cluster2")

If future development includes more than two clusters, the code must be adapted accordingly. Contact the authors if needed.

Execution Logic (all scripts share the same structure):
1. Access to dataset — converts the dataset to .npy format
2. Loading Data — loads data and initializes variables and hyperparameters
3. Model Train — simulates the federated training loop across clients
4. Metrics saves — saves training metrics and performs inference
5. Plots — visualizes results
