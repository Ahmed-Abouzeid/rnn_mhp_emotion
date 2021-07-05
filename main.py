from mailbox import mbox
import torch.utils.data
import torch.nn as nn
from model import RNN
from argparse import ArgumentParser
from utils import load_data, final_data, run_stats, draw_features

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-data_path', type=str, default='data/final/topical_chat')
    parser.add_argument('-e_input_dim', type=int, default=2)
    parser.add_argument('-t_input_dim', type=int, default=1)
    parser.add_argument('-hidden_dim', type=int, default=256)
    parser.add_argument('-mlp_dim', type=int, default=32)
    parser.add_argument('-seq_length', type=int, default=6)
    parser.add_argument('-e_output_dim', type=int, default=2)
    parser.add_argument('-t_output_dim', type=int, default=3)
    parser.add_argument('-t_criterion', type=str, default='mse')
    parser.add_argument('-e_criterion', type=str, default='cross_el')
    parser.add_argument('-l_r', type=float, default=.00001)
    parser.add_argument('-train_split_ratio', type=float, default=1)
    parser.add_argument('-loss_scale', type=float, default=.05)
    parser.add_argument("-drop_out", type=float, default=0.3)
    parser.add_argument('-optimizer', type=str, default='adam')
    parser.add_argument('-gpu', type=str, default='no')
    parser.add_argument('-epochs', type=int, default=35)
    parser.add_argument('-n_layers', type=int, default=1)
    parser.add_argument('-batch_size', type=int, default=1)
    parser.add_argument('-prediction_length', type=int, default=3)
    parser.add_argument('-synthetic_shift_v', type=float, default=1)
    parser.add_argument('-mode', type=str, default='test')
    parser.add_argument('-mut_excit', type=bool, default=False)
    parser.add_argument('-increment', type=bool, default=False)
    parser.add_argument('-model_name', type=str, default='A')

    config = parser.parse_args()

    train_samples, test_samples = load_data(config.data_path)

    train_samples, train_conv = final_data(train_samples, config.seq_length, False, config.mut_excit, config.increment)
    test_samples_no_intervention, test_conv = final_data(test_samples, config.seq_length, False, config.mut_excit,  config.increment)

    test_samples_alt, _ = final_data(test_samples, config.seq_length, True, config.mut_excit,  config.increment)

    #draw_features(test_conv)

    run_stats(train_samples, train_conv, test_samples_no_intervention, test_conv, config.mut_excit)
    rnn = RNN(config)
    if torch.cuda.is_available() and config.gpu == 'yes':
        torch.cuda.set_device(0)
        rnn.cuda()
        dev = 'cuda:0'
    else:
        dev = 'cpu'
    rnn.set_device(dev)

    if config.mode == 'train':
        rnn.train_model(train_samples, test_samples_no_intervention)
    else:
        RNN.evaluate_model(test_samples_no_intervention, config.data_path, config.prediction_length, config)
        RNN.evaluate_intervention(test_samples_no_intervention, test_samples_alt, config)
