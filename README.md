# Nodality
 
The code for the paper 'Who is driving the conversation? Analysing the nodality of British MPs and journalists on Twitter' is published here.

Tweet IDs of the tweets used in this work can be found in the Data folder, along with their classification labels that indicate the topic that the tweet has been assigned to. Users can classify their own tweets using the code available in Tweet_classification folder. Users can define their own classification rules in 'labeling_functions.py'.
If you use this code, please cite:

Ratner, A., Hancock, B., Dunnmon, J., Sala, F., Pandey, S., and R ́e, C. (2019). Training complex models with multi-task weak supervision. Proceedings of the AAAI Conference on Artificial Intelligence. AAAI Conference on Artificial Intelligence, 33:4763–4771

Castro-Gonzalez, L., Chung, Y.-L., Kirk, H. R., Francis, J., Williams, A. R., Johansson, P., and Bright, J. (2024). Cheap Learning: Maximising Performance of Language Models for Social Data Science Using Minimal Data. ArXiv. arXiv:2401.12295 [cs]

Labelled tweets are used to build information networks (the code for this can be found in Generate_networks). Once the networks are formed, node measures are extracted for each node in the network and a PCA analysis is performed on combinations of node measures. The code in PCA_KMeans is used to perform the PCA analysis and KMeans clustering (to cluster nodes based on their inherent nodalities). The PCA results for the cost-of-living crisis topic are shown in 'Plot_PCA_results_Costofliving.ipynb'.

We use the code in the 'Linear_model' folder to study the relationship between influence share of a group of actors on the rest of the population and the mean active and inherent nodalities of the group.

For any queries please reach out to schakraborty@turing.ac.uk.
