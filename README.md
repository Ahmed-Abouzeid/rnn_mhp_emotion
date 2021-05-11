#  Neural Emotion Hawkes Process for Oracling Emotion Consequences in online Dialogues.


This repository provides the source code and datasets used in the paper: Neural Emotion Hawkes Process for Oracling Emotion Consequences in online Dialogues. We provide an Artificial Intelligence model which can provide excitation for the other partner during an one-on-one online text-based conversation. We propose a Neural Emotion Hawkes Process (NEHP) for predicting future emotions of the other conversation partner. Moreover, the model is be able to distinguish between variant future consequences of the provided excitation, and selects the optimum behaviour, accordingly. We evaluate our model on three public datasets: (1) Cornell Movie-Dialogues, (2) Topical Chat, and (3) Iemocap dataset. 


## Model Architicture
<img src="2fig.png" width="400" height="600">


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Running the Experiments

To regenerate results from the paper or to rerun the whole experiments, first, unzip the data folder in the same parent directory, then follow the below commands and select according to your machine power either CPU or GPU:

### to test
```
python main.py --mode 'test' --gpu 'yes --synthetic_shift_v 1 --model_name [A or B or C D] --mut_excit [based on model name, chose yes or no]' --increment [based on model name select yes or no] --data_path [path to the targetted dataset] --prediction_length 5
```
```
python main.py --mode 'test' --gpu 'no' --synthetic_shift_v 1 --model_name [A or B or C D] --mut_excit [based on model name, chose yes or no]' --increment [based on model name select yes or no] --data_path [path to the targetted dataset] --prediction_length 5

```
### to train
```
python main.py --mode 'train' --gpu 'yes'  --mut_excit [based on desired shift representation, chose yes or no]' --increment [based on based on desired shift representation select yes or no] --data_path [path to the targetted dataset] --drop_out 0.3 --e_input_dim 2 --t_input_dim 1 e_output_dim 2 --t_output_dim 3 --hidden_dim 256 --mlp_dim 32 --seq_length 6 --t_criterion 'mse' --e_criterion 'cross_el' --l_r 0.00001 --optimizer 'adam' --n_layers 1 --batch_size 1 --epochs 35
```
```
python main.py --mode 'train' --gpu 'no'  --mut_excit [based on desired shift representation, chose yes or no]' --increment [based on based on desired shift representation select yes or no] --data_path [path to the targetted dataset] --drop_out 0.3 --e_input_dim 2 --t_input_dim 1 e_output_dim 2 --t_output_dim 3 --hidden_dim 256 --mlp_dim 32 --seq_length 6 --t_criterion 'mse' --e_criterion 'cross_el' --l_r 0.00001 --optimizer 'adam' --n_layers 1 --batch_size 1 --epochs 35

```


## Results

Our model achieves the following performance on the three datasets with each emotion shift representation, below are the obtained results as in the paper :

###  Empirical Results for Long-term Prediction Accuracy and Excitation Proprieties Captured on the Three Datasets andDifferent Utilized Representation Methods.


Dataset-Model|Acc #1 | Acc #2 | Acc #3 | Self-Excitation|  Mutual-Excitation|  Test Conv # | Train Conv #|
| ------------------ |---------------- | -------------- | ------------------ |---------------- |---------------- |------------------ |---------------- |
Iemocap-A|0.65|0.67|0.59|0.26|0|30|119|
|TopicalChat-A|0.83|0.72|0.64|0.06|0|349|4742|
|MovieDialogues-A|0.80|0.66|0.56|0.35|0|2291|7999|
|Iemocap-B|0.80|0.75|0.63|0|0|30|119|
|TopicalChat-B|0.88|0.80|0.73|0.01|0|349|4742|
|MovieDialogues-B|0.82|0.67|0.58|0.46|0|2291|7999|
|Iemocap-C|0.87|0.56|0.49|0|0|30|119|
|TopicalChat-C|0.85|0.56|0.51|0.01|0.02|349|4742|
|MovieDialogues-C|0.81|0.51|0.45|0.12|0.14|2291|7999|
|Iemocap-D|0.85|0.44|0.38|0.64|0.71|30|119|
|TopicalChat-D|0.94|0.59|0.50|0.33|0.31|349|4742|
|MovieDialogues-D|0.91|0.59|0.52|0.21|0.18|2291|7999|


## Contributing
All contributions welcome! All content in this repository is licensed under the MIT license.



