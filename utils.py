"""
Util functions for:
- Building vocabulary (word2idx & idx2word)
- Pre-processing text data (punctuation, tokenize, etc.)
- Filter Answers (during evaluation, e.g. K=2, 1000, etc.)
"""
import string
import numpy as np
import pickle
import os
import errno
import matplotlib.pyplot as plt

# Pre-trained VGG-11 weights
PATH_VGG_WEIGHTS = '/home/axe/Projects/Pre_Trained_Models/vgg11_bn-6002323d.pth'


def pad_sequences(seq, max_len):
    """
    Pads a sequence, given max length
    :param seq: list (int tokens)
    :param max_len: pad to max length
    :return: list (padded sequence)
    """
    padded = np.zeros((max_len,), np.int64)
    if len(seq) > max_len:
        padded[:] = seq[:max_len]
    else:
        padded[:len(seq)] = seq
    return padded


def sort_batch(images, questions, answers, ques_seq_lens):
    """
    Sort data (desc.) based on sequence lengths of samples in the batch
    (needed for pad_packed_sequence)
    """
    # question --> (batch_size, sequence_length)

    ques_seq_lens, idx = ques_seq_lens.sort(dim=0, descending=True)
    questions = questions[idx]
    answers = answers[idx]
    images = images[idx]

    return images, questions, answers, ques_seq_lens


# def preprocess_text(text):
#     """
#     Given comma-separated text, removes punctuations & converts to lowercase.
#
#     :param text: string of comma-separated words (sentence)
#     :return: list of preprocessed word tokens
#
#     Example:
#
#     >>> x = 'Man sleeping next to a cat on a bed.'
#     >>> preprocess_text(x)
#     ['man', 'sleeping', 'next', 'to', 'a', 'cat', 'on', 'a', 'bed']
#     """
#     # Comma-separated word tokens
#     text_token_list = text.strip().split(',')
#     text = ' '.join(text_token_list)
#
#     # Remove punctuations
#     table = str.maketrans('', '', string.punctuation)
#     words = text.strip().split()
#     words = [w.translate(table) for w in words]
#
#     # Set to lowercase & drop empty strings
#     words = [word.lower() for word in words if word != '' and word != 's']
#
#     return words

def preprocess_text(input_string, nlp):
    doc = nlp(input_string)
    tokens = [tk.text for tk in doc]
    return tokens


def build_vocab(data, min_word_count=1):
    """
    Given the VQA Dataset, builds vocabulary for the questions

    :param list data: img_name \t question \t answer
    :param int min_word_count: min. word count threshold for including word in the vocab
    :returns: index2word, word2index & max sequence length
    """
    word_count = {}
    max_sequence_length = 0

    # Build a set of unique words
    for sample in data:
        question = sample.split('\t')[1].strip()

        # tokenization with spaCy
        wds = preprocess_text(question)

        # Add words to frequency dict
        for word in wds:
            if word not in word_count:
                word_count[word] = 0
            else:
                word_count[word] += 1

        # Update the max length sequence
        if len(wds) > max_sequence_length:
            max_sequence_length = len(wds)

    # Build word to index mapping
    helper_tokens = {'<PAD>': 0, '<UNKNOWN>': 1, 's': 2, '/s':3}
    num_helper_tokens = len(helper_tokens)
    # TODO simplify this part
    word_list = list(word_count.keys())

    vocab_tokens = {}
    vocab_idx = num_helper_tokens

    for word in word_list:
        # If word meets the count threshold, add it to vocab & increment idx
        if word_count[word] >= min_word_count:
            vocab_tokens[word] = vocab_idx
            vocab_idx += 1

    word2idx = {**helper_tokens, **vocab_tokens}

    # Conversely index to word mapping
    idx2word = {idx: word for word, idx in word2idx.items()}

    return word2idx, idx2word, max_sequence_length


def build_answer(data, K):
    """
    Given the VQA Dataset, builds label-index dicts by \n
    calculating the K most frequent answers from the dataset.

    :param list data: img_name \t question \t answer
    :param int K: num. of most frequent answers
    :returns: label2idx & idx2label
    """
    # Compute the answer frequency
    answer_frequency = {}

    for sample in data:
        answer = sample.split('\t')[2].strip()

        if answer in answer_frequency:
            answer_frequency[answer] += 1
        else:
            answer_frequency[answer] = 1

    # Filter the top-K most frequent answers
    top_k_answers = sorted(answer_frequency.items(), reverse=True, key=lambda kv: kv[1])[:K]
    top_k_answers = [ans for ans, cnt in top_k_answers]

    # Add a dummy UNKNOWN answer for labels that aren't selected in the top-K list
    top_k_answers = ['UNKNOWN'] + top_k_answers

    # Build the label2idx & idx2label mapping
    label2idx = {answer: idx for idx, answer in enumerate(top_k_answers)}
    idx2label = {idx: answer for idx, answer in enumerate(top_k_answers)}

    return label2idx, idx2label


def save_vocab(train_file, vocab_file_path, min_word_count, K):
    """
    Given training dataset file (txt), builds vocabulary from training set
    and saves to `vocab_file_path`. \n

    Effectively, the question vocab is built by filtering out words below the
    `min_word_count` threshold. The answer vocab is generated by selecting the top-K
    most frequent answers in the training set. \n

    We also define a special tag for unknown words & answer labels
    (for words filtered out from the set).

    :param train_file: path to file containing the triplet <img_name, question, answer>
    :param vocab_file_path: path to save the vocabulary
    :param min_word_count: min. word count for including in vocab
    :param K: `K` most frequent answers to be selected
    :return: None
    """
    # Extract <image_filename question answer> samples
    with open(train_file, 'r') as f:
        train_data = f.read().strip().split('\n')

    # Build vocab (word-index dicts, max_seq_len, label-idx dicts)
    word2idx, idx2word, max_seq_length = build_vocab(train_data, min_word_count)
    label2idx, idx2label = build_answer(train_data, K)

    print('Vocab Size: {} \nMax Sequence Length: {}\n'.format(len(word2idx), max_seq_length))

    vocab = {'word2idx': word2idx, 'idx2word': idx2word,
             'label2idx': label2idx, 'idx2label': idx2label,
             'max_seq_length': max_seq_length}

    # Save vocab to disk
    with open(vocab_file_path, 'wb') as handle:
        pickle.dump(vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print('Saving vocab data at {}'.format(vocab_file_path))


def load_vocab(vocab_file):
    """
    Load vocabulary pickle file form disk.

    :param str vocab_file: path to vocab file (.pkl)
    :return: {word2idx, idx2word, max_seq_length}
    :rtype: dict
    """
    # If vocab previously created, load from disk
    if os.path.exists(vocab_file):
        with open(vocab_file, 'rb') as handle:
            vocab = pickle.load(handle)

            print('Loading vocab data from {}'.format(vocab_file))
            print('Vocab data: {}\n'.format(list(vocab.keys())))
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), vocab_file)

    return vocab


# For filtering samples - Evaluation (K=2 Yes/No, K=1000 open-ended)
def filter_samples_by_label(file_path, labels):
    """
    Filters out samples that don't contain answers in the labels list

    :param file_path: path to dataset file (.txt)
    :param labels: answer labels
    :type labels: list

    :return: filtered list of samples from the data file
    """
    # Convert to HashSet: O(1) lookup
    labels = set(labels)

    with open(file_path, 'r') as file_in:
        data = []

        line = file_in.readline()

        while line:
            answer = line.strip().split('\t')[2]

            if answer in labels:
                data.append(line)

            line = file_in.readline()

        return data


def plot_data(dataloader, idx2word, idx2label, num_plots=4):
    """
    For plotting input data (after preprocessing with dataloader). \n
    Helper for sanity check.
    """
    for i, data in enumerate(dataloader):

        # Read dataset, select one random sample from the mini-batch
        batch_size = len(data['label'])
        idx = np.random.choice(batch_size)
        ques = data['question'][idx]
        label = data['label'][idx]
        img = data['image'][idx]

        # Convert question tokens to words & answer class index to label
        ques_str = ' '.join([idx2word[word_idx] for word_idx in ques.tolist()])
        ans_str = ' '.join(idx2label[label.tolist()])

        # Plot Data
        plt.imshow(img.permute(1, 2, 0))
        plt.text(0, 0, ques_str, bbox=dict(fill=True, facecolor='white', edgecolor='red', linewidth=2))
        plt.text(220, 220, ans_str, bbox=dict(fill=True, facecolor='white', edgecolor='blue', linewidth=2))
        plt.show()

        i += 1

        if i >= num_plots:
            break


def print_and_log(msg, log_file):
    """
    :param msg: Message to be printed & logged
    :param file log_file: log file
    :return: None
    """
    log_file.write(msg + '\n')
    log_file.flush()

    print(msg)


def str2bool(v):
    v = v.lower()
    assert v == 'true' or v == 'false'
    return v.lower() == 'true'


def int_min_two(k):
    k = int(k)
    assert k >= 2 and type(k) == int, 'Ensure k >= 2'
    return k
