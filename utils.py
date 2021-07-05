import math
import os
import re
import math
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import collections
from matplotlib.colors import LinearSegmentedColormap

encoding_space__ = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
encoding_space_ = [0, 1, 2, 3, 4, 5, 6]
encoding_space = [0, 1]
encoding_space_shifts = [-1, 0, 1]


def create_synthetic(destination_folder):
    """create synthetic data for emotional conversation mimics a customer satisfaction scenario"""

    c_emotion = [0, -1, -2, -3, -4, 2, 3, 4, 7]  # mimics customer speaker emotions: neutral, afraid, sad, angry,
    # unsatisfied, grateful, happy, excited, statisfied
    a_emotion = [0, 1, 2, 3, 5, 6]  # mimics chatbot agent emotions: neutral, understanding, grateful, happy,
    # determined, curious
    #  now we set some constraints to avoid any non-logical conversation such as human is angry then chatbot is happy
    # however, we will let some logical non-smart acts by the chatbot so we make sure all possible real-life scenarios
    # would be added in the synthetic data
    emotion_constraint = [[-1, 5], [-1, 3], [-1, 2], [-2, 2], [-2, 3], [-2, 5], [-3, 2], [-3, 2],
                          [-3, 3], [0, 5], [2, 0], [2, 1], [2, 5], [4, 5], [4, 1],
                          [6, 7], [3, 7], [7, 1], [5, -2], [5, -3]]

    emo_context_corr_constrains = [[-3, 6, 0], [-3, 6, 2], [-3, 6, 3], [-3, 6, 4], [-3, 5, -3], [-2, 6, 4],
                                   [-1, 0, 3], [-1, 0, 4], [-1, 1, 3], [-1, 1, 4], [-1, 2, 4], [-1, 5, 4],
                                   [-1, 5, 3], [-2, 0, 3], [-2, 0, 4], [-2, 1, 3], [-2, 1, 4], [-2, 2, 4], [-2, 5, 4],
                                   [-2, 5, 3], [-3, 0, 3], [-3, 0, 4], [-3, 1, 3], [-3, 1, 4], [-3, 2, 4], [-3, 5, 4],
                                   [-3, 5, 3]]
    number_of_turns_in_seq = 12
    num_of_conversations = 30000
    sequences = {}
    counter = 0
    for i in tqdm(list(range(num_of_conversations)), 'creating synthetic conversations'):
        sequence_length = number_of_turns_in_seq
        while sequence_length > 0:
            while True:
                c = random.choice(c_emotion)
                a = random.choice(a_emotion)
                # while True:
                #      c_1 = random.choice(c_event_types)
                #      if c_1 != 4:
                #          break

                seq_sample = [c, a]
                # if random.random() > .5:
                #     random.shuffle(seq_sample)

                if i in sequences.keys():
                    sequences[i] += seq_sample
                else:
                    sequences.update({i: seq_sample})

                sequence_length -= 2
                if sequence_length == 0:
                    # sequences[i] += [random.choice([4, 5])]
                    is_good_seq, satisfied_conv_count = is_good_sequence(sequences[i], emotion_constraint,
                                                                         emo_context_corr_constrains,
                                                                         c_emotion, a_emotion)
                    if is_good_seq:
                        break
                    else:
                        sequence_length = number_of_turns_in_seq
                        sequences[i] = []

        counter += satisfied_conv_count

    print('convs perc with satisfied outcome:', counter/num_of_conversations)
    prepare_final_data_synthetic(sequences, destination_folder)
    return sequences


def prepare_final_data_synthetic(sequences, destination_folder):
    """prepare the csv files before running the model and its pre-processing (embedding/ multip seq replication)"""

    train_destination_end_path = destination_folder + '/train.csv'
    test_destination_end_path = destination_folder + '/test.csv'
    f_w = open(train_destination_end_path, 'a')
    f_w_tst = open(test_destination_end_path, 'a')
    f_w.write('seq_id, cont_var1, event_id, cont_var2\n')
    f_w_tst.write('seq_id, cont_var1, event_id, cont_var2\n')

    for seq_id, conv_id in enumerate(sequences.keys()):
        for e, emo in enumerate(sequences[conv_id]):
            if e % 2 == 0:
                evnt_id = 0
            else:
                evnt_id = 1
            cont_var1 = 0
            cont_var2 = emo
            if seq_id < 27000:
                f_w.write(str(seq_id) + ',' + str(cont_var1) + ',' + str(evnt_id) + ',' + str(cont_var2) + '\n')
            else:
                f_w_tst.write(str(seq_id) + ',' + str(cont_var1) + ',' + str(evnt_id) + ',' + str(cont_var2) + '\n')

    f_w.close()
    f_w_tst.close()


def is_good_sequence(sequence, emotion_constraint, emo_context_corr_constrains, c_emotion, a_emotion):
    """helper function that evaluate a generate sequence according to teh given sequence constraints"""

    satisfied_convs_count = 0
    x = len(sequence) - 1
    y = len(sequence) - 2
    #excited = 0
    unsatisfied_index = -1
    satisfied = False
    if (2 in sequence or 3 in sequence or 4 in sequence) and (1 not in sequence or 5 not in sequence):
        return False, 0
    for e, emo in enumerate(sequence):
        if e == 0 and emo in [7, -4]:
            return False, 0
        if emo == -4:
            unsatisfied_index = e

        if unsatisfied_index!= -1 and unsatisfied_index+2 == e and emo in [4, 3, 2]:
            return False, 0

        if emo == 7:
            satisfied = True

        if (emo == -3 or emo == -4) and satisfied:
            return False, 0

        if e < x:
            if [emo, sequence[e + 1]] in emotion_constraint or (emo in c_emotion and sequence[e + 1] in c_emotion) \
                    or (emo in a_emotion and sequence[e + 1] in a_emotion):
                return False, 0

        if e < y:
            if [emo, sequence[e + 1], sequence[e + 2]] in emo_context_corr_constrains:
                return False, 0

    if random.random() >= .3:
        if 7 not in sequence:
            satisfied_convs_count = 0
            return False, satisfied_convs_count
        else:
            satisfied_convs_count = 1

    return True, satisfied_convs_count


def read_types(conversations):
    """helper function to interpret emotional ids"""

    dict = {0: 'neutral', 1: 'understanding', 2: 'grateful', 3: 'happy', 4: 'excited', 5: 'determined'
        , -1: 'afraid', -2: 'sad', -3: 'angry', 6: 'curious', 7: 'satisfied', -4: 'unsatisfied'}

    for conv_id in conversations.keys():
        for emo in conversations[conv_id]:
            print(dict[emo])
        print('----------------------------')


def load_data(data_folder):
    """function that loads final data format for training the model"""

    test_data = []
    train_data = []
    for fil in os.listdir(data_folder):
        end_path = data_folder + '/' + fil
        if fil == 'test.csv':
            f_r = open(end_path, 'r')
            test_data = f_r.readlines()
            f_r.close()
        elif fil == 'train.csv':
            f_r = open(end_path, 'r')
            train_data = f_r.readlines()
            f_r.close()

    return train_data[1:], test_data[1:]  # we start from index 1 to avoid header


def final_data(data, seq_length, intervention=False, mut_excit = False, increment = False):
    """function that clean final data for training the model, by returning a clean seq list per sample, use
    max_length to include only certain sequences with certain length as maximum"""

    prev_id = None
    feats = []
    all_evnt_ids = []
    evnt_ids = []
    conv_vals = []
    conv_ids = []
    all_conv_ids = []
    for i in data:
        if i.split(',')[0] == prev_id or prev_id is None:
            prev_id = i.split(',')[0]
            conv_vals.append(float(i.split(',')[3].strip('\n')))
            evnt_ids.append(int(i.split(',')[2]))
            conv_ids.append(i.split(',')[0])
        else:
            if len(conv_vals) % 2 == 0:  # to maintain only odd length conversations for balancing groung truthes during learning
                conv_vals = conv_vals[:-1]
                evnt_ids = evnt_ids[:-1]
                conv_ids = conv_ids[:-1]

            seq_w_emo_shifts = to_feature_emo_shift(conv_vals, intervention, mut_excit, increment)
            feats.append(seq_w_emo_shifts)
            all_evnt_ids.append(evnt_ids)
            all_conv_ids.append(conv_ids)
            conv_ids = []
            evnt_ids = []
            conv_vals = []
            prev_id = i.split(',')[0]
            conv_vals.append(float(i.split(',')[3].strip('\n')))
            evnt_ids.append(int(i.split(',')[2]))
            conv_ids.append(i.split(',')[0])

    featured_data = list(zip(all_conv_ids, all_evnt_ids, feats))

    ready_data = []
    for conv in featured_data:
        start_index = 0
        ending = seq_length
        while ending <= len(conv[0]):
            event_type_seq = conv[1][start_index: ending]
            cont_var2_seq = conv[2][start_index: ending]
            ready_data.append([conv[0][0], event_type_seq, cont_var2_seq])
            start_index += 1
            ending += 1

    return ready_data, featured_data


def prepare_raw_data_scenarioSA(target_data_fdr, destination_data_fdr):
    """prepare the raw dataset format into a final formal and create csv files for, used for scenarioSA data"""

    for e_fil, fil in enumerate(os.listdir(target_data_fdr)):
        target_end_path = target_data_fdr + '/' + fil

        train_destination_end_path = destination_data_fdr + '/train.csv'
        test_destination_end_path = destination_data_fdr + '/test.csv'

        f_r = open(target_end_path, 'r')
        f_w = open(train_destination_end_path, 'a')
        f_w_tst = open(test_destination_end_path, 'a')
        if e_fil == 0:
            f_w.write('seq_id, cont_var1, event_id, cont_var2\n')
            f_w_tst.write('seq_id, cont_var1, event_id, cont_var2\n')

        for e_line, l in enumerate(f_r.readlines()):
            if len(l.split(' ')) < 3:
                continue
            seq_id = fil.split('.')[0]
            evnt_id = l.split(':')[0].strip()

            if evnt_id == 'A':
                evnt_id = 0
            else:
                evnt_id = 1

            cont_var1 = 0

            # cont_var2 = l.split(':')[1].split(' ')[-1].strip()
            cont_var2 = l.split(':')[1:][-1].split(' ')[-1].strip()

            print(seq_id, evnt_id, cont_var1, float(cont_var2))

            if e_fil < 2000:
                f_w.write(str(seq_id) + ',' + str(cont_var1) + ',' + str(evnt_id) + ',' + str(cont_var2) + '\n')
            else:
                f_w_tst.write(str(seq_id) + ',' + str(cont_var1) + ',' + str(evnt_id) + ',' + str(cont_var2) + '\n')

        f_r.close()
        f_w.close()
        f_w_tst.close()


def prepare_raw_data_iemocap(target_data_fdr, destination_data_fdr):
    """prepare the raw dataset format into a final formal and create csv files for, used for iemocap data"""

    for e_fil, fil in enumerate(os.listdir(target_data_fdr+'/sessions_transcripts')):
        target_end_path = target_data_fdr+'/sessions_transcripts'+ '/' + fil

        train_destination_end_path = destination_data_fdr + '/train.csv'
        test_destination_end_path = destination_data_fdr + '/test.csv'

        f_r = open(target_end_path, 'r')
        f_w = open(train_destination_end_path, 'a')
        f_w_tst = open(test_destination_end_path, 'a')
        if e_fil == 0:
            f_w.write('seq_id, cont_var1, event_id, cont_var2\n')
            f_w_tst.write('seq_id, cont_var1, event_id, cont_var2\n')

        for e_line, l in enumerate(f_r.readlines()):
            seq_id = e_fil
            if e_line % 2 == 0:
                evnt_id = 0
            else:
                evnt_id = 1

            cont_var1 = 0

            # cont_var2 = l.split(':')[1].split(' ')[-1].strip()
            turn_name = l.split(' ')[0].strip('\n')
            cont_var2 = None
            if len(turn_name) > 12:
                cont_var2 = get_iemocap_emotion(target_data_fdr+'/ground_truths'+ '/' + fil, turn_name)
                if cont_var2 == 'neu' or cont_var2 == 'Neutral':
                    cont_var2 = 0
                elif cont_var2 == 'fru' or cont_var2 == 'Frustration':
                    cont_var2 = -1
                elif cont_var2 == 'sad' or cont_var2 == 'Sadness':
                    cont_var2 = -3
                elif cont_var2 == 'ang' or cont_var2 == 'Anger':
                    cont_var2 = -4
                elif cont_var2 == 'exc' or cont_var2 == 'Excited':
                    cont_var2 = 3
                elif cont_var2 == 'hap' or cont_var2 == 'Happiness':
                    cont_var2 = 2
                elif cont_var2 == 'sur' or cont_var2 == 'Surprise':
                    cont_var2 = 1
                elif cont_var2 == 'fea' or cont_var2 == 'Fear':
                    cont_var2 = -2
                elif cont_var2 == 'dis' or cont_var2 == 'Disgust':
                    cont_var2 = -5

            if cont_var2 is not None:
                print(seq_id, evnt_id, cont_var1, float(cont_var2))
                if e_fil < 120:
                    f_w.write(str(seq_id) + ',' + str(cont_var1) + ',' + str(evnt_id) + ',' + str(cont_var2) + '\n')
                else:
                    f_w_tst.write(str(seq_id) + ',' + str(cont_var1) + ',' + str(evnt_id) + ',' + str(cont_var2) + '\n')

        f_r.close()
        f_w.close()
        f_w_tst.close()


def get_iemocap_emotion(session_file_name, turn_name):
    """gets the assigned evaluated iemocap emotion for a given turn name in a given session dialogue"""

    r = open(session_file_name, 'r')
    lines = r.readlines()
    for e, l in enumerate(lines):
        emo_info = l.split('\t')
        if len(emo_info) == 4 and emo_info[1] == turn_name:
            if emo_info[2] in ['xxx', 'oth']:
                emo = lines[e+1].split('\t')[1].strip(';')
                if emo.split(';')[0] != 'Other':
                    return emo.split(';')[0]
                else:
                    emo = lines[e + 2].split('\t')[1].strip(';')
                    if emo.split(';')[0] != 'Other':
                        return emo.split(';')[0]
                    else:
                        emo = lines[e + 3].split('\t')[1].strip(';')
                        return emo.split(';')[0]
            else:
                return emo_info[2]


def prepare_raw_data_topicalchat(target_data_path, destination_data_fdr):
    """this function process the raw json file for the topical chat dataset to extract sequential events of both
    conversation turn taker and the associated sentiment value of such turn"""

    f_w_train = open(destination_data_fdr + '/' + 'train.csv', 'a')
    f_w_test = open(destination_data_fdr + '/' + 'test.csv', 'a')
    f_w_train.write('seq_id, cont_var1, event_id, cont_var2\n')
    f_w_test.write('seq_id, cont_var1, event_id, cont_var2\n')

    with open(target_data_path) as f:
        data = json.load(f)
        for e, item in enumerate(data.items()):
            seq_id = e
            if len(item[-1]['content']) == 21:
                for e_, msg in enumerate(item[-1]['content']):
                    cont_var1 = 0
                    if e_ % 2 == 0:
                        evnt_id = 0
                    else:
                        evnt_id = 1

                    if msg['sentiment'].lower() == 'angry':
                        cont_var2 = -4
                    elif msg['sentiment'].lower() == 'disgusted':
                        cont_var2 = -3
                    elif msg['sentiment'].lower() == 'fearful':
                        cont_var2 = -2
                    elif msg['sentiment'].lower() == 'sad':
                        cont_var2 = -1
                    elif msg['sentiment'].lower() == 'neutral':
                        cont_var2 = 0
                    elif msg['sentiment'].lower() == 'curious to dive deeper':
                        cont_var2 = 1
                    elif msg['sentiment'].lower() == 'surprised':
                        cont_var2 = 2
                    elif msg['sentiment'].lower() == 'happy':
                        cont_var2 = 3
                    if e < 8000:
                        f_w_train.write(
                            str(seq_id) + ',' + str(cont_var1) + ',' + str(evnt_id) + ',' + str(cont_var2) + '\n')
                    else:
                        f_w_test.write(
                            str(seq_id) + ',' + str(cont_var1) + ',' + str(evnt_id) + ',' + str(cont_var2) + '\n')

    f_w_train.close()
    f_w_test.close()


def input_2_tensor(input_seq, encoding_space):
    """convert regular sequence of values to one hot encoding tensor"""

    input_tensor = torch.zeros((len(input_seq), len(encoding_space)), dtype=torch.float)
    for e, element in enumerate(input_seq):
        element_tensor = torch.zeros(len(encoding_space), dtype=torch.float)
        index = encoding_space.index(element)
        element_tensor[index] = 1
        input_tensor[e] = element_tensor

    return input_tensor


def batch_samples(samples, batch_size, mut_excit):
    """helper function to divid a list into n smaller chunks, where n is calculated from batch:size
    parameter, each sample in all chunks will have input-output format for the ground truth,
    the output is always one step ahead of the input"""

    n = len(samples) // batch_size
    batches = generate_sequences_x_y(samples, batch_size, mut_excit)
    if n != len(samples) / batch_size:  # handling last batch when list size is not a multiple of n and batch size
        picked_elements = []
        while len(picked_elements) < batch_size:
            picked_elements.append(random.choice(samples))

        batches += generate_sequences_x_y(picked_elements, batch_size)

    return batches


def scale_samples(train_samples, test_samples):
    """scale the continuies variable before passing the samples for batching and for training. 
       Could be useful to overcome the sensetivity of the LSTM twards the feature numeric scale and would improve the predeiction results """

    train_scaled = []
    test_scaled = []
    for sample in train_samples:
        vals = []
        for num_val in sample[2]:
            vals.append(num_val)

        #train_scaled.append(torch.tanh(torch.tensor(vals, dtype=float)).tolist())
    for sample in test_samples:
        vals = []

        for num_val in sample[2]:
            vals.append(num_val)
        #test_scaled.append(torch.tanh(torch.tensor(vals, dtype=float)).tolist())

    scaler = StandardScaler()
    scaler.fit(torch.tensor(vals).reshape(-1, 1))

    train_scaled_samples = []
    for e, sample in enumerate(train_samples):
        event_seq = sample[1]
        conv_id = sample[0]
        scaled_cont_var_seq = scaler.transform(torch.tensor(sample[2]).reshape(-1, 1))
        scaled_cont_var_seq = torch.tensor(scaled_cont_var_seq).flatten().tolist()
        #scaled_cont_var_seq = train_scaled[e]
        train_scaled_samples.append([conv_id, event_seq, scaled_cont_var_seq])

    test_scaled_samples = []
    for e, sample in enumerate(test_samples):
        event_seq = sample[1]
        conv_id = sample[0]
        scaled_cont_var_seq = scaler.transform(torch.tensor(sample[2]).reshape(-1, 1))
        scaled_cont_var_seq = torch.tensor(scaled_cont_var_seq).flatten().tolist()
        #scaled_cont_var_seq = test_scaled[e]
        test_scaled_samples.append([conv_id, event_seq, scaled_cont_var_seq])

    return train_scaled_samples, test_scaled_samples, scaler  # we return the scaler object to use it again in the model when inverse is required


def create_train_validate_sets(batches, split_ratio):
    """this function splits the samples in the batches into two sets, one for training, and another for validation"""

    training_set_end_idx = int(len(batches) * split_ratio)
    training_batches = batches[: training_set_end_idx]
    validation_batches = batches[training_set_end_idx:]

    return training_batches, validation_batches


def get_max_seq_length(samples):
    """helper function to get the maximum seq length for padding preparation"""

    seq_lengths = []
    for seq in samples:
        seq_lengths.append(len(seq[0]))

    return max(seq_lengths)


def append_cont_var_paddings(cont_var_seq, max_length):
    """append padding values to the continues variable values so it matches padding with event sequence values"""

    if len(cont_var_seq) < max_length:
        temp_list = [0 for _ in range(max_length - len(cont_var_seq))]

        return torch.FloatTensor(cont_var_seq.tolist() + temp_list)
    else:
        return cont_var_seq


def generate_sequences_x_y(samples, batch_size, mut_excit):
    """generate the input-output pairs"""

    batches = []
    batch = []
    if mut_excit:
        for element in tqdm(samples, 'generate batches'):
            batch.append([input_2_tensor(element[1][:-1], encoding_space),
                              input_2_tensor([element[1][-1]], encoding_space),
                          torch.FloatTensor(element[2][:-1]),
                          input_2_tensor([calc_emotion_shift(element[2][-2], element[2][-1])], encoding_space_shifts)])

            if len(batch) == batch_size:
                batches.append(batch)
                batch = []
    else:
        for element in tqdm(samples, 'generate batches'):
            batch.append([input_2_tensor(element[1][:-1], encoding_space),
                          input_2_tensor([element[1][-1]], encoding_space),
                          torch.FloatTensor(element[2][:-1]),
                          input_2_tensor([calc_emotion_shift(element[2][-3], element[2][-1])], encoding_space_shifts)])

            if len(batch) == batch_size:
                batches.append(batch)
                batch = []

    return batches


def calc_categ_accur(g_truth, predicts):
    """two lists must be same in size, one for ground:truth and another for model predictions,
    we keep the function works over lists even if it is a many to one prediction, that is for reusability on
    many to many sequence models"""

    true_counter = 0
    false_counter = 0

    if predicts == g_truth:
        true_counter += 1
    else:
        false_counter += 1

    return true_counter / (true_counter + false_counter)


def calc_mae(g_truth, predicts, scaler):
    """calculating the Mean Absolute Error,
    two lists must be same in size, one for ground:truth and another for model predictions"""

    error = 0
    if scaler:
        g_truth = scaler.inverse_transform(g_truth.reshape(-1, 1)).flatten().tolist()
        predicts = scaler.inverse_transform(predicts.cpu().detach().reshape(-1, 1)).flatten().tolist()
    else:
        g_truth = g_truth.tolist()
        predicts = predicts.tolist()
    for e, result in enumerate(predicts):  # to eliminate padded values from the evaluation process
        error += abs(result - g_truth[e])

    return error / len(g_truth)


def to_feature_incremental(real_numbers_seq):
    """this helper function transforms a sequence of real numbers into a feauture-oriented sequence
    by standardizing the entries into differences between each current entry and the one before, this is suitable
    when the continues variable in question is incremental over time"""

    seq = np.array([real_numbers_seq[0]] + real_numbers_seq)
    seq = np.diff(seq)

    return seq


def calc_emotion_shift(prev_emo, current_emo):
    """helper function to set an emotion shift class for a previous sequence of shifts"""

    if prev_emo == current_emo:
        return 0
    elif current_emo > prev_emo:
        return 1
    elif current_emo < prev_emo:
        return -1


def cal_seq_emo_shift(real_sequence_cont_var, mut_excit):
    """calculates the emotion shifts as class values over a sequence"""

    shifts = []
    if mut_excit:
        for e, _ in enumerate(real_sequence_cont_var):
            if e < len(real_sequence_cont_var)-1:
                shifts.append(calc_emotion_shift(real_sequence_cont_var[e], real_sequence_cont_var[e+1]))
    else:
        for e, _ in enumerate(real_sequence_cont_var):
            if e < len(real_sequence_cont_var)-2:
                shifts.append(calc_emotion_shift(real_sequence_cont_var[e], real_sequence_cont_var[e+2]))
    return shifts


def calc_emotion_feat(prev_emo, current_emo):
    """helper function to calculate the difference or the shift between two emotions"""

    if prev_emo == current_emo:
        return 0
    else:
        x = current_emo - prev_emo
        return x


def to_feature_emo_shift(real_numbers_seq, intervention, mut_excit = False, increment = False):
    """this helper function transforms a sequence of real numbers into a feauture-oriented sequence,
    the features can be represented as emotion shifts for each individual turn taker in a dialogue"""

    event_1_seq = []
    event_2_seq = []
    indexer = 0
    emotion_shifts = []
    while indexer <= len(real_numbers_seq) - 1:
        event_1_seq.append(real_numbers_seq[indexer])
        indexer += 1
        if indexer > len(real_numbers_seq) - 1:
            break
        event_2_seq.append(real_numbers_seq[indexer])
        indexer += 1
    
    if intervention:
        random.shuffle(event_2_seq) # if we evaulate for intevention and excitation so we permute chatbot sequence (event_seq_2)

    if not mut_excit:
        seq_1 = [event_1_seq[0]]
        for e, val in enumerate(event_1_seq):
            if e < len(event_1_seq) - 1:
                if increment:
                    seq_1.append(calc_emotion_feat(seq_1[-1], event_1_seq[e+1]))
                else:
                    seq_1.append(calc_emotion_feat(event_1_seq[e], event_1_seq[e+1]))

        seq_2  = [event_2_seq[0]]
        for e, val in enumerate(event_2_seq):
            if e < len(event_2_seq) - 1:
                if increment:
                    seq_2.append(calc_emotion_feat(seq_2[-1], event_2_seq[e+1]))
                else:
                    seq_2.append(calc_emotion_feat(event_2_seq[e], event_2_seq[e+1]))

        seq_1_2 = [0 for _ in range(len(seq_1) + len(seq_2))]

        indiv_indexer = 0
        whole_indexer = 0

        while whole_indexer < len(seq_1_2):
            if indiv_indexer < len(seq_1):
                seq_1_2[whole_indexer] = seq_1[indiv_indexer]
                whole_indexer += 1
            if indiv_indexer < len(seq_2):
                seq_1_2[whole_indexer] = seq_2[indiv_indexer]
                whole_indexer += 1
                indiv_indexer += 1
        return seq_1_2

    else:
        seq_1_2 = [0 for _ in range(len(event_1_seq) + len(event_2_seq))]

        indiv_indexer = 0
        whole_indexer = 0

        while whole_indexer < len(seq_1_2):
            if indiv_indexer < len(event_1_seq):
                seq_1_2[whole_indexer] = event_1_seq[indiv_indexer]
                whole_indexer += 1
            if indiv_indexer < len(event_2_seq):
                seq_1_2[whole_indexer] = event_2_seq[indiv_indexer]
                whole_indexer += 1
                indiv_indexer += 1
        seq_1_2_mutation = [seq_1_2[0]]
        for e, val in enumerate(seq_1_2):
            if e < len(seq_1_2) - 1:
                if increment:
                    seq_1_2_mutation.append(calc_emotion_feat(seq_1_2_mutation[-1], seq_1_2[e+1]))
                else:
                    seq_1_2_mutation.append(calc_emotion_feat(seq_1_2[e], seq_1_2[e+1]))

        return seq_1_2_mutation


def plot_learning_curve(xs, train_ys, test_ys):
    """plotting the learning curve with regard to model accuracy, loss"""

    plt.ylabel('Emotion Shift Accuracy')
    ys_train = [y[0] for y in train_ys]
    ys_test = [y[0] for y in test_ys]
    plt.plot(xs, ys_train, label='train', color='royalblue')
    plt.plot(xs, ys_test, label='test', color='green')
    plt.xlabel('Epochs')
    plt.yticks([0, .1, .2, .3, .4,  .5, .6, .7, .8, .9, 1])
    plt.xticks([x for x in xs if x % 2 == 0])
    plt.grid()
    plt.legend()
    plt.savefig('results/acc.png')
    plt.close()

    plt.ylabel('Total Loss')
    ys_train = [y[1] for y in train_ys]
    ys_test = [y[1] for y in test_ys]
    plt.plot(xs, ys_train, label='train', color='royalblue')
    plt.plot(xs, ys_test, label='test', color='green')
    plt.xlabel('Epochs')
    plt.xticks([x for x in xs if x % 2 == 0])
    plt.grid()
    plt.legend()
    plt.savefig('results/loss.png')
    plt.close()


def plot_pred_vs_real(xs_1, xs_2, real_ys, predicts_ys, real_evnt_ids, predicted_evnt_ids, conv_id):
    """plotting the trajectory of a learned continues variable and its evolving over time, while predicted events id
    parameter will be used to compare the predicted event types trajectories over time steps"""

    A_x, B_x, A_y, B_y = split_conv_parties(xs_1, real_ys)
    A_x_simu, B_x_simu, A_y_simu, B_y_simu = split_conv_parties(xs_2, predicts_ys, xs_2[0] % 2 != 0)
    plt.scatter(A_x, A_y, label='A: real', marker = 'x', color='royalblue')
    plt.scatter(A_x_simu, A_y_simu, s=80, facecolors='none', edgecolors='lightsteelblue', label='A: simu')
    plt.ylabel('Emotion Shift')
    plt.xlabel('Conversation Turn Index')
    plt.yticks([-1, 0, 1])
    plt.locator_params(axis='x', nbins=20)
    plt.legend()
    plt.grid()
    plt.savefig('results/'+conv_id+'_A.png')
    plt.close()

    plt.scatter(B_x, B_y, label='B: real', marker = 'x', color='royalblue')
    plt.scatter(B_x_simu, B_y_simu, s=80, facecolors='none', edgecolors='lightsteelblue', label='B: simu')
    plt.ylabel('Emotion Shift')
    plt.xlabel('Conversation Turn Index')
    plt.yticks([-1, 0, 1])
    plt.locator_params(axis='x', nbins=20)
    plt.legend()
    plt.grid()
    plt.savefig('results/'+conv_id+'_B.png')
    plt.close()


def split_conv_parties(whole_x, whole_y, A_start=None):
    """helper function to split a sequence from a conversation into two seqs to show each user sentimental
    evolving alone"""

    if A_start is None:
        A_x = [x for x in whole_x if x % 2 != 0]
        B_x = [x for x in whole_x if x % 2 == 0]

        A_y, B_y = [], []
        for e, y in enumerate(whole_y):
            if e % 2 == 0:
                A_y.append(y)
            else:
                B_y.append(y)

    else:
        if A_start == True:
            A_y, B_y, A_x, B_x = [], [], [], []
            for e, x in enumerate(whole_x):
                if e % 2 == 0:
                    A_x.append(x)
                else:
                    B_x.append(x)
            for e, y in enumerate(whole_y):
                if e % 2 == 0:
                    A_y.append(y)
                else:
                    B_y.append(y)
        else:
            A_y, B_y, A_x, B_x = [], [], [], []
            for e, x in enumerate(whole_x):
                if e % 2 != 0:
                    A_x.append(x)
                else:
                    B_x.append(x)

            for e, y in enumerate(whole_y):
                if e % 2 != 0:
                    A_y.append(y)
                else:
                    B_y.append(y)

    return A_x, B_x, A_y, B_y


def stepping_tanh():
    """a helper function to step with .1 through the tanh function range"""

    mx = 1
    i = -1
    steps = []
    while True:
        steps.append(i)
        i += .1
        if round(i, 1) > mx:
            break

    return steps


def prepare_raw_data_movie_lines(target_data_fdr, destination_data_fdr):
    """prepare the raw dataset format into a final formal and create csv files for, used for movie lines
     data"""

    reader = open(target_data_fdr+'\\annotated.txt', 'r')
    lines = reader.readlines()
    train_destination_end_path = destination_data_fdr + '/train.csv'
    test_destination_end_path = destination_data_fdr + '/test.csv'
    f_w = open(train_destination_end_path, 'a')
    f_w_tst = open(test_destination_end_path, 'a')
    f_w.write('seq_id, cont_var1, event_id, cont_var2\n')
    f_w_tst.write('seq_id, cont_var1, event_id, cont_var2\n')
    prev_speaker = None
    turns = []
    repeat_num = 0
    for e, line in enumerate(lines):
        seq_id = line.split('+++!+++')[0]
        speaker = line.split('+++!+++')[1]
        sentence = line.split('+++!+++')[2]
        annotation_labels = re.findall('LABEL_\d*', str(line.split('+++!+++')[3].split(':')[1].split(',')[:-1]))
        annotation_scores = re.findall('\d.\d*', str(line.split('+++!+++')[3].split(':')[2].split(',')))
        if prev_speaker is not None and prev_speaker == speaker and prev_seq_id == seq_id:
            annotation_labels = prev_labels + annotation_labels
            annotation_scores = prev_scores + annotation_scores
            sentence = prev_sentence + '. ' + sentence
            repeat_num += 1
            turns.pop(e-repeat_num)
        prev_speaker = speaker
        prev_labels = annotation_labels
        prev_scores = annotation_scores
        prev_sentence = sentence
        prev_seq_id = seq_id

        turns.append([seq_id, speaker, sentence, annotation_labels, annotation_scores])
    prev_seq_id = None
    row_tracer = 0
    turn_counter = 0
    no_emo_counter = 0
    emotions = []
    for seq_id, _, sentence, annotation_labels, annotation_scores in turns:
        turn_counter += 1
        cont_var2, no_emotion_counter = compute_emotion(annotation_labels, annotation_scores)
        emotions.append(cont_var2)
        no_emo_counter += no_emotion_counter
        cont_var1 = 0
        if prev_seq_id is None or prev_seq_id != seq_id:
            row_tracer = 0
        if row_tracer % 2 == 0:
            evnt_id = 0
        else:
            evnt_id = 1

        row_tracer += 1
        prev_seq_id = seq_id
        # if int(seq_id) < 8000:
        #     f_w.write(str(seq_id) + ',' + str(cont_var1) + ',' + str(evnt_id) + ',' + str(cont_var2) + '\n')
        # else:
        #     f_w_tst.write(str(seq_id) + ',' + str(cont_var1) + ',' + str(evnt_id) + ',' + str(cont_var2) + '\n')

    reader.close()
    f_w.close()
    f_w_tst.close()

    print('perc of no emotion classified over all turns: ', no_emo_counter/turn_counter, 'as ',no_emo_counter,
          ' was missing')
    return emotions


def compute_emotion(labels, scores):
    """helper function that transforms labels with highest score to emotional values"""

    no_label_counter = 0
    label_mapping = {'LABEL_0':7,'LABEL_1':9,'LABEL_2':-5,'LABEL_3':-4,'LABEL_4':10
    ,'LABEL_5':13,'LABEL_6':-2,'LABEL_7':3,'LABEL_8':12,'LABEL_9':-3,'LABEL_10':-6
    ,'LABEL_11':-7,'LABEL_12':-12,'LABEL_13':11,'LABEL_14':-8,'LABEL_15':4,'LABEL_16':-11
    ,'LABEL_17':8,'LABEL_18':14,'LABEL_19':-13,'LABEL_20':5,'LABEL_21':6,'LABEL_22':1
    ,'LABEL_23':2,'LABEL_24':-10,'LABEL_25':-9,'LABEL_26':-1,'LABEL_27':0}
    if scores:
        selected_index = scores.index(max(scores))
        selected_emo_label = labels[selected_index]
        return label_mapping[selected_emo_label], no_label_counter
    else:
        no_label_counter += 1
        return 0, no_label_counter


def estimate_emotion_influence(emotions_shifts):
    """this important function estimates how stronger the excitation between two emotions shifts
    by calculating the frequency of occurrence of one emotion shift following another"""

    emotion_shift_lookup = np.unique(emotions_shifts)
    A = np.zeros((len(emotion_shift_lookup), len(emotion_shift_lookup)), dtype=float)
    for index, e in enumerate(emotion_shift_lookup):
        #all_accu = count_freq_elem(emotions_shifts, e)
        mutual_excitation = get_mutual_excitation_probs(e, emotions_shifts)
        #print(mutual_excitation_probs.values())
        A[index] = list(mutual_excitation.values())
        #if e > -50 and e < 50:
            #print(e, A[index])
        print(e, mutual_excitation)
    #influencer_index = np.where(emotion_shift_lookup == 24)
    #influenced_index = np.where(emotion_shift_lookup == 6)
    #print('influencer index:', influencer_index)
    #print(A)


def get_mutual_excitation_probs(element, my_list):
    """helper function to estimate influence between a passed element and all other
    elements in a passed list"""

    counts = {}
    emotion_shift_lookup = np.unique(my_list)
    for e_shift in emotion_shift_lookup:
        counts[e_shift] = 0
    for e, elem in enumerate(my_list):
        if elem == element:
            if e < len(my_list) -1:
                counts[my_list[e+1]] += 1

    for key in counts.keys():
        counts[key] = counts[key]

    return counts


def count_freq_elem(my_list, element):
    """helper function to calc the frequency of an element in a list"""

    counter = 0
    for elem in my_list:
        if elem == element:
            counter += 1
    return counter


def convert_encoding_to_value(encoding):
    """converts the encoding to the real class value"""

    indx = encoding.tolist().index(max(encoding.tolist()))

    if len(encoding.tolist()) > 2:
        if indx == 0:
            predicts = -1
        elif indx == 1:
            predicts = 0
        elif indx == 2:
            predicts = 1
        return predicts
    else:
        return indx


def convert_to_sequence_shift_value(turn_recent_seq_value, emo_shift_class_value, synthetic_shift_value):
    """converts a class value to normal sequence shift value to match the format of the LSTM input. Used for processing
    sequence values to match LSTM input for long term prediction"""

    turn_recent_seq_value = float(turn_recent_seq_value)
    synthetic_shift = None
    if emo_shift_class_value == 0:
        synthetic_shift = turn_recent_seq_value
    elif emo_shift_class_value == 1:
        synthetic_shift = turn_recent_seq_value + synthetic_shift_value
    elif emo_shift_class_value == -1:
        synthetic_shift = turn_recent_seq_value - synthetic_shift_value

    return synthetic_shift


def run_stats(train_samples, train_convs, test_samples, test_convs, mut_excit):
    """runs a statistical report on the datasets, the samples argument will be the training or test samples.
    On the other hand, the convs argument will be the conversations before sliding them into samples. Stats will be
    calculated from both types of data"""

    num_train_convs = len(train_convs)
    num_test_convs = len(test_convs)
    num_train_samples = len(train_samples)
    num_test_samples = len(test_samples)

    train_conv_lengths = []
    train_shifts = []
    for conv in train_convs:
        train_conv_lengths.append(len(conv[0]))
        train_shifts.append(np.mean(conv[2]))

    test_conv_lengths = []
    test_shifts = []
    for e, conv in enumerate(test_convs):
        test_conv_lengths.append(len(conv[0]))
        test_shifts.append(np.mean(conv[2]))

    avg_train_conv_length = np.mean(train_conv_lengths)
    avg_test_conv_length = np.mean(test_conv_lengths)

    avg_train_shifts = np.mean(train_shifts)
    avg_test_shifts = np.mean(test_shifts)
    std_train_shifts = np.std(train_shifts)
    std_test_shifts = np.std(test_shifts)

    train_classes = []
    for seq in generate_sequences_x_y(train_samples, 1, mut_excit):
        train_classes.append(convert_encoding_to_value(seq[0][3][0]))

    test_classes = []
    for seq in generate_sequences_x_y(test_samples, 1, mut_excit):
        test_classes.append(convert_encoding_to_value(seq[0][3][0]))

    train_counter = collections.Counter(train_classes)
    test_counter = collections.Counter(test_classes)

    train_class_freq = {-1: train_counter[-1]/num_train_samples, 0:train_counter[0]/num_train_samples,
                        1:train_counter[1]/num_train_samples}
    test_class_freq = {-1: test_counter[-1]/num_test_samples, 0:test_counter[0]/num_test_samples,
                       1:test_counter[1]/num_test_samples}

    print('train convs num: ', num_train_convs)
    print('test convs num: ', num_test_convs)
    print('train samples num: ', num_train_samples)
    print('test samples num: ', num_test_samples)
    print('avg train convs length: ', avg_train_conv_length)
    print('avg test convs length: ', avg_test_conv_length)
    print('avg train conv emotion shift: ', avg_train_shifts)
    print('avg test conv emotion shift: ', avg_test_shifts)
    print('std train conv emotion shift: ', std_train_shifts)
    print('std test conv emotion shift: ', std_test_shifts)
    print('train samples class freq: ', train_class_freq)
    print('test samples class freq: ', test_class_freq)


def draw_features(train_convs):
    """draws the emotion shifts for each model, shows how a conversation as an example evolves in term of emotion shifts
    where the latter is represented in a certain way"""

    shifts = []
    for conv in train_convs:
        if conv[0][0] == '8000':
            shifts = conv[2][:10]
            break
    turns = list(range(len(shifts)))
    N = len(turns)
    data = np.ones((N, N)) * np.nan
    ref_shift_indices = [-1 for _ in range(len(shifts))]
    for e, s in enumerate(shifts):
        ref_shift_indices[e] = sorted(shifts).index(s)

    for t in turns:
        data[ref_shift_indices[t]][t] = shifts[t]

    new_data = np.zeros((N, N))
    for e, row in enumerate(reversed(data)):
        new_data[e] = row

    fig, ax = plt.subplots(1, 1, tight_layout=True)
    # make color map
    dic = {'red': ((0., 1, 0),
                   (0.66, 1, 1),
                   (0.89, 1, 1),
                   (1, 0.5, 0.5)),
           'green': ((0., 1, 0),
                     (0.375, 1, 1),
                     (0.64, 1, 1),
                     (0.91, 0, 0),
                     (1, 0, 0)),
           'blue': ((0., 1, 1),
                    (0.34, 1, 1),
                    (0.65, 0, 0),
                    (1, 0, 0))}

    # set the'bad' values (nan) to be white and transparent
    cmap = LinearSegmentedColormap('custom_cmap', dic)
    cmap.set_bad('white')
    # draw the grid
    for x in range(N + 1):
        ax.axhline(x, lw=0.1, color='k')
        ax.axvline(x, lw=0.1, color='k')
    # draw the boxes
    img = ax.imshow(new_data, interpolation='none', cmap=cmap, extent=[0, N, 0, N])
    fig.colorbar(img)
    print(shifts)
    print(ref_shift_indices)
    ax.axis('off')
    plt.show()
    exit()



