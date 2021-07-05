#  Learning Tame Hidden Representations for Non-parametric Hawkes Processes.


This repository presents source code and data used in the paper "Learning Tame Hidden Representations for Non-parametric Hawkes Processes.". The paper provides an LSTM-based Hawkes process which can learn hidden representations within a short-length dyadic conversation emotion dynamics. Learning hidden influence patterns in conversations opens the venue for emotional intelligent chatbots where a controller agent can estimate the future consequences of its expressed emotions.. We evaluate our model on two public datasets: (1) Cornell Movie-Dialogues, and (2) Topical Chat. 


## Model Architicture
<img src="fig_arc.PNG" width="800" height="500">

## Model Parameters
<img src="fig_param.PNG" width="300" height="350">

## Requirements

### To install requirements:

```setup
pip install -r requirements.txt
```

## Running the Experiments

To regenerate results from the paper or to rerun the whole experiments, first, unzip the data folder in the same parent directory, then follow the below commands and select according to your machine power either CPU or GPU:

## to test

### on GPU


```
python main.py --mode 'test' --gpu 'yes --synthetic_shift_v 1 --model_name [A or B or C D] --mut_excit [based on model name, chose yes or no]' --increment [based on model name select yes or no] --data_path [path to the targetted dataset] --prediction_length 3
```

### on CPU

```
python main.py --mode 'test' --gpu 'no' --synthetic_shift_v 1 --model_name [A or B or C D] --mut_excit [based on model name, chose yes or no]' --increment [based on model name select yes or no] --data_path [path to the targetted dataset] --prediction_length 3

```
## to train

### on GPU

```
python main.py --mode 'train' --gpu 'yes'  --mut_excit [based on desired shift representation, chose yes or no]' --increment [based on based on desired shift representation select yes or no] --data_path [path to the targetted dataset] --drop_out 0.3 --e_input_dim 2 --t_input_dim 1 e_output_dim 2 --t_output_dim 3 --hidden_dim 256 --mlp_dim 32 --seq_length 6 --t_criterion 'mse' --e_criterion 'cross_el' --l_r 0.00001 --optimizer 'adam' --n_layers 1 --batch_size 1 --epochs 35
```

### on CPU

```
python main.py --mode 'train' --gpu 'no'  --mut_excit [based on desired shift representation, chose yes or no]' --increment [based on based on desired shift representation select yes or no] --data_path [path to the targetted dataset] --drop_out 0.3 --e_input_dim 2 --t_input_dim 1 e_output_dim 2 --t_output_dim 3 --hidden_dim 256 --mlp_dim 32 --seq_length 6 --t_criterion 'mse' --e_criterion 'cross_el' --l_r 0.00001 --optimizer 'adam' --n_layers 1 --batch_size 1 --epochs 35

```


## Results

Our model achieves the following performance on the two datasets with each emotion change representation, below are the obtained results as in the paper for the other partner (e.g., human prediction accuracy and mutual excitation captured due to the chatbot actions) :

###  Empirical Results for Short-term Prediction Accuracy and Excitation Proprieties Captured on the two Datasets and Different Utilized Representation Methods.


Representation| Acc (1) | Acc (2) | Acc avg | Mutual-Excitation|
| ------------------ |---------------- | -------------- | ------------------ |---------------- |
|TopicalChat-A|0.82|0.51|0.67|0.16|
|MovieDialogues-A|0.80|0.35|0.57|0.40|
|TopicalChat-B|0.89|0.62|0.76|0.05|
|MovieDialogues-B|0.82|0.20|0.51|0.45|
|TopicalChat-C|0.85|0.54|0.64|0.06|
|MovieDialogues-C|0.81|0.37|0.59|0.24|
|TopicalChat-D|0.94|0.57|0.76|0.06|
|MovieDialogues-D|0.92|0.76|0.84|0.16|


## Contributing
All contributions welcome! All content in this repository is licensed under the MIT license.




