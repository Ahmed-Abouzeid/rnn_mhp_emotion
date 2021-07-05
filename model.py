import torch
from tqdm import tqdm
import torch.nn as nn
from utils import batch_samples, calc_categ_accur,\
    create_train_validate_sets, plot_learning_curve, \
    input_2_tensor, encoding_space, plot_pred_vs_real, \
    convert_encoding_to_value, convert_to_sequence_shift_value, cal_seq_emo_shift
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import math


class RNN(nn.Module):

    def __init__(self, config):
        """initialize the RNN object"""

        super(RNN, self).__init__()

        self.device = 'cpu'
        self.lstm = nn.LSTM(config.e_input_dim + config.t_input_dim, config.hidden_dim, config.n_layers, batch_first=True)
        self.fc_mlp = nn.Linear(config.hidden_dim, config.mlp_dim)

        self.drop_out = nn.Dropout(config.drop_out)
        self.e_fc = nn.Linear(config.mlp_dim, config.e_output_dim)
        self.t_fc = nn.Linear(config.mlp_dim, config.t_output_dim)

        self.hidden_dim = config.hidden_dim

        self.e_input_dim = config.e_input_dim
        self.t_input_dim = config.t_input_dim

        self.seq_length = config.seq_length
        self.l_r = config.l_r
        self.epochs = config.epochs
        self.batch_size = config.batch_size

        self.e_criterion = RNN.set_criterion(config.e_criterion)
        self.t_criterion = RNN.set_criterion(config.t_criterion)
        self.optimizer = self.set_optimizer(config.optimizer)

        self.train_split_ratio = config.train_split_ratio
        self.loss_scale = config.loss_scale
        self.prediction_length = config.prediction_length
        self.synthetic_shift_v = config.synthetic_shift_v

        self.config = config

    def set_device(self, dev):
        """set the network running device to either gpu or cpu"""

        self.device = dev

    @staticmethod
    def set_criterion(criterion):
        """set objective function for the optimizer"""

        if criterion.lower() == 'nllloss':
            return nn.NLLLoss()
        elif criterion.lower() == 'cross_el':
            return nn.CrossEntropyLoss()
        elif criterion.lower() == 'mae':
            return nn.L1Loss()
        elif criterion.lower() == 'mse':
            return nn.MSELoss()

    def set_optimizer(self, optimizer):
        """set the optimizer for the training process"""

        if optimizer.lower() == 'sgd':
            return torch.optim.SGD(self.parameters(), lr=self.l_r)
        elif optimizer.lower() == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.l_r)
        elif optimizer.lower() == 'adabound':
            return ''  # to be implemented

    def forward(self, e_input, t_input, mode = 'train'):
        """compute a step forward in time in the network for a given sequence, then returns
         the output and hidden state values"""

        lstm_input = torch.cat((e_input, t_input.unsqueeze(-1)), dim=1)
        if mode == 'train':
            h_t, _ = self.lstm(lstm_input.view(1, lstm_input.shape[0], lstm_input.shape[1]).to(self.device))
        else:
            h_t, _ = self.lstm(lstm_input.view(1, lstm_input.shape[0], lstm_input.shape[1]).to('cpu'))
        mlp_output = torch.tanh(self.fc_mlp(h_t[:, -1, :]))
        mlp_output = self.drop_out(mlp_output)
        e_output = self.e_fc(mlp_output)
        t_output = self.t_fc(mlp_output)
        return e_output, t_output

    def init_hidden(self):
        """initialize the hidden state for computation time step 0"""

        return torch.zeros((1, 1, self.hidden_dim))

    def train_model(self, train_samples, test_samples):
        """training the RNN network on sequence samples as tensors"""

        batches = batch_samples(train_samples, self.batch_size, self.config.mut_excit)
        test_batches = batch_samples(test_samples, self.batch_size, self.config.mut_excit)
        train_batches, val_batches = create_train_validate_sets(batches, self.train_split_ratio)
        xs, train_ys, val_ys, test_ys = [], [], [], []

        for e in range(self.epochs):
            e_train_accuracy = []
            t_train_accuracy = []

            total_loss = 0
            sample_counter = 0

            for e_, batch in enumerate(tqdm(train_batches, 'training')):
                e_loss = 0
                t_loss = 0

                for e_x, e_y, t_x, t_y in batch:
                    sample_counter += 1
                    e_output, t_output = self.forward(e_x, t_x, self.config.mode)
                    e_output = e_output[-1].unsqueeze(0)
                    t_output = t_output[-1].unsqueeze(0)

                    e_y = torch.tensor([el.tolist().index(max(el.tolist())) for el in e_y]).to(self.device)
                    t_y = torch.tensor([el.tolist().index(max(el.tolist())) for el in t_y]).to(self.device)
                    e_loss += self.e_criterion(e_output, e_y)
                    t_loss += self.e_criterion(t_output, t_y)
                    e_predicts = e_output[0].tolist().index(max(e_output[0].tolist()))
                    e_g_truthes = e_y.tolist()[0]
                    t_predicts = t_output[0].tolist().index(max(t_output[0].tolist()))
                    t_g_truthes = t_y.tolist()[0]
                    e_train_accuracy.append(calc_categ_accur(e_g_truthes, e_predicts))
                    t_train_accuracy.append(calc_categ_accur(t_g_truthes, t_predicts))

                loss = (e_loss + t_loss) * self.loss_scale
                total_loss += loss
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            e_train_accuracy = round(np.mean(e_train_accuracy), 4)
            t_train_accuracy = round(np.mean(t_train_accuracy), 4)

            total_loss_per_sample = round(total_loss.item() / sample_counter, 4)

            e_test_accr, t_test_acc, test_total_loss = \
                self.validate_test_model(test_batches, 'test')

            print('\n| epoch->', e, ' | ', 'train total loss->', total_loss_per_sample, ' | turn class accr->',
                  e_train_accuracy, ' | emotion class accr->',
                  t_train_accuracy, ' |')

            print(' | test total loss->', test_total_loss, ' | class accr->',
                  e_test_accr, ' | emotion class accr->', t_test_acc, ' |')
            print('-------------------------------------------------------------------------')
            xs.append(e)
            train_ys.append((t_train_accuracy, total_loss_per_sample, 0))
            test_ys.append((t_test_acc, test_total_loss, 0))

        torch.save(self, 'saved_models/'+self.config.data_path.split('/')[-1]+'.pth')
        plot_learning_curve(xs, train_ys, test_ys)

    def validate_test_model(self, validate_batches, data='val'):
        """run validation or test across validating/testing dataset"""

        e_accuracy = []
        t_accuracy = []
        total_loss = 0
        sample_counter = 0
        if data == 'val':
            caption = 'validation'
        else:
            caption = 'testing'

        for batch in tqdm(validate_batches, caption):
            e_loss = 0
            t_loss = 0
            for e_x, e_y, t_x, t_y in batch:
                sample_counter += 1
                e_output, t_output = self.forward(e_x, t_x, self.config.mode)
                e_output = e_output[-1].unsqueeze(0)
                t_output = t_output[-1].unsqueeze(0)
                e_y = torch.tensor([el.tolist().index(max(el.tolist())) for el in e_y]).to(self.device)
                t_y = torch.tensor([el.tolist().index(max(el.tolist())) for el in t_y]).to(self.device)

                e_loss += self.e_criterion(e_output, e_y)
                t_loss += self.e_criterion(t_output, t_y)
                e_predicts = e_output[0].tolist().index(max(e_output[0].tolist()))
                e_g_truthes = e_y.tolist()[0]
                t_predicts = t_output[0].tolist().index(max(t_output[0].tolist()))
                t_g_truthes = t_y.tolist()[0]
                e_accuracy.append(calc_categ_accur(e_g_truthes, e_predicts))
                t_accuracy.append(calc_categ_accur(t_g_truthes, t_predicts))

            loss = (e_loss + t_loss) * self.loss_scale
            total_loss += loss

        e_accuracy = round(np.mean(e_accuracy), 4)
        t_acc = round(np.mean(t_accuracy), 4)

        total_loss_per_sample = round(total_loss.item() / sample_counter, 4)

        return e_accuracy, t_acc, total_loss_per_sample

    @staticmethod
    def generate_predicted_sequence(test_samples, conv_id, config):
        """this method tries to predict the next sequence hopping it matches with the groundtruth one,
        it works with long term prediction as well."""

        single_sequence_cont_var = []
        single_sequence_evnt_id = []
        prediction_seq_evnt_id = []
        prediction_seq_cont_var = []

        model = torch.load('saved_models/'+config.data_path.split('/')[-1] + '_' + config.model_name+'.pth', map_location=torch.device('cpu'))
        model.cpu()
        for e, seq in enumerate(test_samples):
            if seq[0] == conv_id:
                if e < len(test_samples) - 1:
                   if test_samples[e+1][0] != conv_id:
                       single_sequence_evnt_id += seq[1]
                       single_sequence_cont_var += seq[2]
                       break
                   else:
                       single_sequence_evnt_id.append(seq[1][0])
                       single_sequence_cont_var.append(seq[2][0])
                else:
                    single_sequence_evnt_id.append(seq[1][0])
                    single_sequence_cont_var.append(seq[2][0])
        if len(single_sequence_evnt_id) < config.seq_length - 1:
            return None, None, None, None

        whole_tensor_cont_var = torch.FloatTensor(single_sequence_cont_var)
        whole_tensor_evnt_id = input_2_tensor(single_sequence_evnt_id, encoding_space)

        start_index = 0
        end_index = model.seq_length - 1
        while end_index < len(whole_tensor_cont_var):
            if start_index == 0:
                seq_evnt_id = whole_tensor_evnt_id[start_index: end_index].detach().clone()
                seq_cont_var = whole_tensor_cont_var[start_index: end_index].detach().clone()
            else:
                emo_shift_class_value = convert_encoding_to_value(t_output)
                if config.mut_excit:
                    output = convert_to_sequence_shift_value(seq_cont_var[-1], emo_shift_class_value,
                                                             model.synthetic_shift_v)
                else:
                    output = convert_to_sequence_shift_value(seq_cont_var[-2], emo_shift_class_value,
                                                             model.synthetic_shift_v)
                seq_evnt_id = torch.cat((seq_evnt_id[1:], e_output.unsqueeze(0)), dim=0)
                seq_cont_var = torch.cat((seq_cont_var[1:], torch.FloatTensor([output])), dim=0)

            start_index += 1
            end_index += 1
            e_output, t_output = model.forward(seq_evnt_id, seq_cont_var, config.mode)
            e_output = e_output[-1].unsqueeze(0)
            t_output = t_output[-1].unsqueeze(0)

            e_temp_tensor = torch.zeros(2)
            t_temp_tensor = torch.zeros(3)
            e_indx = e_output[0].tolist().index(max(e_output[0].tolist()))
            t_indx = t_output[0].tolist().index(max(t_output[0].tolist()))
            t_temp_tensor[t_indx] = 1
            e_temp_tensor[e_indx] = 1
            e_output = e_temp_tensor
            t_output = t_temp_tensor
            e_predicts = convert_encoding_to_value(e_output)
            t_predicts = convert_encoding_to_value(t_output)
            prediction_seq_evnt_id.append(e_predicts)
            prediction_seq_cont_var.append(t_predicts)

        real_sequence_evnt_id = whole_tensor_evnt_id
        real_sequence_cont_var = whole_tensor_cont_var

        if config.mut_excit:
            real_sequence_cont_var = cal_seq_emo_shift(real_sequence_cont_var[config.seq_length-2:], config.mut_excit)
        else:
            real_sequence_cont_var = cal_seq_emo_shift(real_sequence_cont_var[config.seq_length-3:], config.mut_excit)

        real_sequence_evnt_id = [convert_encoding_to_value(v) for v in real_sequence_evnt_id][config.seq_length-1:]
        classes_truths = []
        for c in real_sequence_cont_var[:config.prediction_length]:
            classes_truths.append(c)

        if len(classes_truths) < config.prediction_length:
            return None, None, None, None
        classes_pred = []
        for c in prediction_seq_cont_var[:config.prediction_length]:
            classes_pred.append(c)

        tp_1 = 0
        fp_1 = 0
        tp_2 = 0
        fp_2 = 0

        for e, g_t in enumerate(classes_truths):
            if e == 0:
                if classes_pred[e] == g_t:
                    tp_1 += 1
                else:
                    fp_1 += 1
            elif e == 2:
                if classes_pred[e] == g_t:
                    tp_2 += 1
                else:
                    fp_2 += 1

        xs_1 = [i+1 for i in range(len(real_sequence_evnt_id))][:config.prediction_length]
        xs_2 = [i+1 for i in range(len(prediction_seq_evnt_id))][:config.prediction_length]
        plot_pred_vs_real(xs_1, xs_2, real_sequence_cont_var[:config.prediction_length],
                          prediction_seq_cont_var[:config.prediction_length], real_sequence_evnt_id[:config.prediction_length],
                          prediction_seq_evnt_id[:config.prediction_length], conv_id)

        return tp_1, fp_1, tp_2, fp_2

    @staticmethod
    def evaluate_model(test_samples, test_files_path, prediction_length, config):
        """evaluate model prediction for long term sequence prediction"""

        reader = open(test_files_path + "/test.csv", 'r')
        lines = reader.readlines()
        start = int(lines[1].split(',')[0])
        end = int(lines[-1].split(',')[0])
        reader.close()
        accs_1 = []
        accs_2 = []

        for _ in tqdm(range(10)):
            exp_accs_1 = []
            exp_accs_2 = []

            for conv_id in range(start, end, 1):
                tps_1, fps_1, tps_2, fps_2 = RNN.generate_predicted_sequence(test_samples, str(conv_id), config)
                if tps_1 is not None and fps_1 is not None:
                    exp_accs_1.append(tps_1 / (tps_1 + fps_1))
                if tps_2 is not None and fps_2 is not None:
                    exp_accs_2.append(tps_2 / (tps_2 + fps_2))

            accs_1.append(sum(exp_accs_1) / len(exp_accs_1))
            accs_2.append(sum(exp_accs_2) / len(exp_accs_2))

        print('Avg Long Term Acc: ', round(sum(accs_1) / len(accs_1), 4), 'On Future Turn: 1')
        print('Avg Long Term Acc: ', round(sum(accs_2) / len(accs_2), 4), 'On Future Turn: 2')

    @staticmethod
    def evaluate_intervention(test_samples_h0, test_samples_h1, config):
        """runs a t-test on two hypothetical sequence of emotion of ne conversation partner to evaluate if
        swapping (alternative H) emotion has significant effect on the excitation of emotion of the other conversation
        partner"""

        print('******************************************')
        print('EVALUATING INTERVENTION CAPABILITIES')
        print('******************************************')

        sig_perc = []
        alt_was_better_perc = []
        for i in range(6):
            all_ids = set()
            for sample in test_samples_h0:
                all_ids.add(sample[0])
            counter = 0
            alternative_better = 0
            for conv_id in all_ids:
                p_value, conseq_basic, conseq_alter = RNN.run_t_test(test_samples_h0, test_samples_h1, conv_id, config)
                if p_value == -1:
                    print('Number of Samples is Less than 10 for the T-test or p-value cannot be measured')
                    continue
                elif p_value < .05:
                    counter += 1
                    if conseq_alter > conseq_basic:
                        alternative_better += 1
            sig_perc.append(counter / len(all_ids))
            if counter != 0:
                alt_was_better_perc.append(alternative_better / counter)
            else:
                alt_was_better_perc.append(0)
        print('Statistical Significance Found Over Data Samples With Perc:', round(np.mean(sig_perc), 2))
        print('Total Number of Conversation:', len(all_ids))
        if counter > 0:
            print('Perc That Alternative Consequence was Better:', round(np.mean(alt_was_better_perc), 2))
        else:
            print('Perc That Alternative Consequence was Better:', 0)

    @staticmethod
    def run_t_test(h_0_samples, h_1_samples, conv_id, config):
        """running the actual T-test"""

        a = []
        b = []
        all_samples = [h_0_samples, h_1_samples]
        model = torch.load('saved_models/' + config.data_path.split('/')[-1] + '_' + config.model_name+'.pth', map_location=torch.device('cpu'))
        model.cpu()

        for enum, test_samples in enumerate(all_samples):
            single_sequence_cont_var = []
            single_sequence_evnt_id = []
            prediction_seq_evnt_id = []
            prediction_seq_cont_var = []
            for e, seq in enumerate(test_samples):
                if seq[0] == conv_id:
                    if e < len(test_samples) - 1:
                        if test_samples[e + 1][0] != conv_id:
                            single_sequence_evnt_id += seq[1]
                            single_sequence_cont_var += seq[2]
                            break
                        else:
                            single_sequence_evnt_id.append(seq[1][0])
                            single_sequence_cont_var.append(seq[2][0])
                    else:
                        single_sequence_evnt_id.append(seq[1][0])
                        single_sequence_cont_var.append(seq[2][0])

            if len(single_sequence_evnt_id) < config.seq_length - 1:
                return -1, -1, -1
            whole_tensor_cont_var = torch.FloatTensor(single_sequence_cont_var)
            whole_tensor_evnt_id = input_2_tensor(single_sequence_evnt_id, encoding_space)

            start_index = 0
            end_index = model.seq_length - 1
            while end_index < len(whole_tensor_cont_var):
                if start_index == 0:
                    seq_evnt_id = whole_tensor_evnt_id[start_index: end_index].detach().clone()
                    seq_cont_var = whole_tensor_cont_var[start_index: end_index].detach().clone()
                else:
                    emo_shift_class_value = convert_encoding_to_value(t_output)
                    if config.mut_excit:
                        output = convert_to_sequence_shift_value(seq_cont_var[-1], emo_shift_class_value,
                                                                 model.synthetic_shift_v)
                    else:
                        output = convert_to_sequence_shift_value(seq_cont_var[-2], emo_shift_class_value,
                                                                 model.synthetic_shift_v)
                    seq_evnt_id = torch.cat((seq_evnt_id[1:], e_output.unsqueeze(0)), dim=0)
                    seq_cont_var = torch.cat((seq_cont_var[1:], torch.FloatTensor([output])), dim=0)

                start_index += 1
                end_index += 1
                e_output, t_output = model.forward(seq_evnt_id, seq_cont_var, config.mode)
                e_output = e_output[-1].unsqueeze(0)
                t_output = t_output[-1].unsqueeze(0)

                e_temp_tensor = torch.zeros(2)
                t_temp_tensor = torch.zeros(3)
                e_indx = e_output[0].tolist().index(max(e_output[0].tolist()))
                t_indx = t_output[0].tolist().index(max(t_output[0].tolist()))
                t_temp_tensor[t_indx] = 1
                e_temp_tensor[e_indx] = 1
                e_output = e_temp_tensor
                t_output = t_temp_tensor
                e_predicts = convert_encoding_to_value(e_output)
                t_predicts = convert_encoding_to_value(t_output)
                prediction_seq_evnt_id.append(e_predicts)
                prediction_seq_cont_var.append(t_predicts)

            P = []
            for e, emo_class in enumerate(prediction_seq_cont_var):
                if e % 2 == 0: # even indices mean that we are evaluating mutual excitation on human sequence, 
                    # odd means we evaluate self excitation on chatbot. 
                    # The permutation of emotions is done only in the feature vector for the chatbot sequence.
                    P.append(emo_class)
            if enum == 0:
                a = P[:2]
            else:
                b = P[:2]

        if len(a) >= 1:
            tStat, pValue = stats.ttest_ind(a, b, equal_var=False)
            sns.kdeplot(a, shade=True, label='Consequence(i), M= ' + str(round(np.mean(a), 2)), linestyle="--", color='black')
            sns.kdeplot(b, shade=True, label='Consequence(j), M= ' + str(round(np.mean(b), 2)), color='maroon')
            plt.xlabel('Avg Emotion Shift')
            plt.title('P-value='+str(round(pValue, 4)))

            plt.legend()
            if pValue < .05:
                plt.savefig('results/' + str(pValue) + '_' + conv_id + '.png')
            plt.close()
            if not math.isnan(pValue):
                return pValue, np.mean(a), np.mean(b)
            else:
                return -1, -1, -1
        else:
            return -1, -1, -1
