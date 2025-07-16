For modelling Potency, a base model was trained using ChEMBL ligand data mapped with UniPROT protein targets. The IC50 value (in nM) was used as the outputs to train a protein transformer - ligand GNN concatenation model. 

The base model was then fine-tuned using three different approaches:

- simple: Only the last layers (representing the FC head) were unfrozen and the model fine-tuned with the Polaris train data

- partial unfreezing: The FC head as well as the top layers of both the protein and graph embeddings were unfrozen and and the model fine-tuned with the Polaris train data

- gradual unfreezing: The layers are unfrozen in a sequential manner, allowing the model to adjust to the change in the input distribution without dramatic changes to the network. Fine-tuning is performed with Polaris train data.

The predictions from all these models are compared using a validation split and the best model is identified. The best model is used to predict with the Polaris unblinded test data and the following performance was obtained.  

Partial Unfreezing: MAE 0.8286; $R^2$ 0.3128

Gradual Unfreezing: MAE 0.8506; $R^2$ 0.2732



