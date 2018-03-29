import functools

# import dynet as dy
import torch, torch.nn as nn
from torch.autograd import Variable
import numpy as np

import trees

START = "<START>"
STOP = "<STOP>"
UNK = "<UNK>"

def augment(scores, oracle_index):
    # assert isinstance(scores, dy.Expression)
    assert isinstance(scores, Variable)
    # shape = scores.dim()[0]
    shape = scores.size()
    assert len(shape) == 1
    # increment = np.ones(shape)
    increment = torch.ones(shape)
    increment[oracle_index] = 0
    # return scores + dy.inputVector(increment)
    return scores + to_variable(increment)

def to_variable(t):
    if torch.cuda.is_available():
        t = t.cuda()
    return Variable(t)


# class Feedforward(object):
#     def __init__(self, model, input_dim, hidden_dims, output_dim):
#         self.spec = locals()
#         self.spec.pop("self")
#         self.spec.pop("model")

#         self.model = model.add_subcollection("Feedforward")

#         self.weights = []
#         self.biases = []
#         dims = [input_dim] + hidden_dims + [output_dim]
#         for prev_dim, next_dim in zip(dims, dims[1:]):
#             self.weights.append(self.model.add_parameters((next_dim, prev_dim)))
#             self.biases.append(self.model.add_parameters(next_dim))

class Feedforward(nn.Sequential):

    def __init__(self, input_dim, hidden_dims, output_dim):
        dims = [input_dim] + hidden_dims + [output_dim]
        layers = []
        for i, (prev_dim, next_dim) in enumerate(zip(dims, dims[1:])):
            layers.append(nn.Linear(prev_dim, next_dim))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        super(Feedforward, self).__init__(*layers)

#     def param_collection(self):
#         return self.model

#     @classmethod
#     def from_spec(cls, spec, model):
#         return cls(model, **spec)

#     def __call__(self, x):
#         for i, (weight, bias) in enumerate((zip(self.weights, self.biases)):
#             weight = dy.parameter(weight)
#             bias = dy.parameter(bias)
#             x = dy.affine_transform([bias, weight, x])
#             if i < len(self.weights) - 1:
#                 x = dy.rectify(x)
#         return x

# class TopDownParser(object):
class TopDownParser(nn.Module):
    def __init__(
            self,
            tag_vocab,
            word_vocab,
            label_vocab,
            tag_embedding_dim,
            word_embedding_dim,
            lstm_layers,
            lstm_dim,
            label_hidden_dim,
            split_hidden_dim,
            dropout,
    ):
        # self.spec = locals()
        # self.spec.pop("self")
        # self.spec.pop("model")
        super(TopDownParser, self).__init__()

        # self.model = model.add_subcollection("Parser")
        self.tag_vocab = tag_vocab
        self.word_vocab = word_vocab
        self.label_vocab = label_vocab
        self.lstm_dim = lstm_dim

        # self.tag_embeddings = self.model.add_lookup_parameters(
        #     (tag_vocab.size, tag_embedding_dim))
        # self.word_embeddings = self.model.add_lookup_parameters(
        #     (word_vocab.size, word_embedding_dim))
        self.tag_embeddings = nn.Embedding(tag_vocab.size, tag_embedding_dim)
        self.word_embeddings = nn.Embedding(word_vocab.size, word_embedding_dim)

        # self.lstm = dy.BiRNNBuilder(
        #     lstm_layers,
        #     tag_embedding_dim + word_embedding_dim,
        #     2 * lstm_dim,
        #     self.model,
        #     dy.VanillaLSTMBuilder)
        self.lstm = nn.LSTM(
            input_size = tag_embedding_dim + word_embedding_dim,
            hidden_size = lstm_dim,
            num_layers = lstm_layers,
            dropout = dropout,
            bidirectional = True)

        # self.f_label = Feedforward(
        #     self.model, 2 * lstm_dim, [label_hidden_dim], label_vocab.size)
        # self.f_split = Feedforward(
        #     self.model, 2 * lstm_dim, [split_hidden_dim], 1)
        self.f_label = Feedforward(
            2 * lstm_dim, [label_hidden_dim], label_vocab.size)
        self.f_split = Feedforward(
            2 * lstm_dim, [split_hidden_dim], 1)

        # self.dropout = dropout

    # def param_collection(self):
    #     return self.model

    # @classmethod
    # def from_spec(cls, spec, model):
    #     return cls(model, **spec)

    # def parse(self, sentence, gold=None, explore=True):
    def parse(self, sentences, gold_trees=None, explore=True):
        # is_train = gold is not None
        is_train = gold_trees is not None
        assert is_train == self.training

        # if is_train:
        #     self.lstm.set_dropout(self.dropout)
        # else:
        #     self.lstm.disable_dropout()

        # embeddings = []
        max_sen_len = max([len(sentence) for sentence in sentences])
        indexes = torch.LongTensor(2, max_sen_len + 2, len(sentences))
        indexes[0].fill_(self.tag_vocab.index(STOP))
        indexes[1].fill_(self.word_vocab.index(STOP))
        for s, sentence in enumerate(sentences):
            for w, (tag, word) in enumerate([(START, START)] + sentence + [(STOP, STOP)]):
                # tag_embedding = self.tag_embeddings[self.tag_vocab.index(tag)]
                indexes[0][w][s] = self.tag_vocab.index(tag)
                if word not in (START, STOP):
                    count = self.word_vocab.count(word)
                    if not count or (is_train and np.random.rand() < 1 / (1 + count)):
                        word = UNK
                # word_embedding = self.word_embeddings[self.word_vocab.index(word)]
                indexes[1][w][s] = self.word_vocab.index(word)
                # embeddings.append(dy.concatenate([tag_embedding, word_embedding]))
  
        self.sentences = sentences
        self.gold_trees = gold_trees
        self.explore = explore
        if torch.cuda.is_available():
            indexes = indexes.cuda()        
        losses = self(to_variable(indexes))
        return self.trees, losses

    def forward(self, indexes):

        embeddings = torch.cat([self.tag_embeddings(indexes[0]), 
                                self.word_embeddings(indexes[1])],
                                -1)
        # lstm_outputs = self.lstm.transduce(embeddings)
        lstm_outputs, _ = self.lstm(embeddings)

        self.trees = []
        losses = []
        for i in range(len(self.sentences)):
            tree, loss = self.greedy_tree(
                            self.training, 
                            lstm_outputs[:,i,:], 
                            self.sentences[i],
                            self.gold_trees[i] if self.gold_trees is not None else None, 
                            self.explore)
            self.trees.append(tree)
            losses.append(loss)
        return torch.cat(losses)


    def greedy_tree(self, is_train, lstm_outputs, sentence, gold, explore):

        @functools.lru_cache(maxsize=None)
        def get_span_encoding(left, right):
            forward = (
                lstm_outputs[right][:self.lstm_dim] -
                lstm_outputs[left][:self.lstm_dim])
            backward = (
                lstm_outputs[left + 1][self.lstm_dim:] -
                lstm_outputs[right + 1][self.lstm_dim:])
            # return dy.concatenate([forward, backward])
            return torch.cat([forward, backward])

        def helper(left, right):
            assert 0 <= left < right <= len(sentence)

            label_scores = self.f_label(get_span_encoding(left, right))

            if is_train:
                oracle_label = gold.oracle_label(left, right)
                oracle_label_index = self.label_vocab.index(oracle_label)
                label_scores = augment(label_scores, oracle_label_index)

            # label_scores_np = label_scores.npvalue()
            label_scores_np = label_scores.data.cpu().numpy()
            argmax_label_index = int(
                label_scores_np.argmax() if right - left < len(sentence) else
                label_scores_np[1:].argmax() + 1)
            argmax_label = self.label_vocab.value(argmax_label_index)

            if is_train:
                label = argmax_label if explore else oracle_label
                label_loss = (
                    label_scores[argmax_label_index] -
                    label_scores[oracle_label_index]
                    # if argmax_label != oracle_label else dy.zeros(1))
                    if argmax_label != oracle_label else to_variable(torch.zeros(1)))
            else:
                label = argmax_label
                label_loss = label_scores[argmax_label_index]

            if right - left == 1:
                tag, word = sentence[left]
                tree = trees.LeafParseNode(left, tag, word)
                if label:
                    tree = trees.InternalParseNode(label, [tree])
                return [tree], label_loss

            left_encodings = []
            right_encodings = []
            for split in range(left + 1, right):
                left_encodings.append(get_span_encoding(left, split))
                right_encodings.append(get_span_encoding(split, right))
            # left_scores = self.f_split(dy.concatenate_to_batch(left_encodings))
            # right_scores = self.f_split(dy.concatenate_to_batch(right_encodings))
            left_scores = self.f_split(torch.stack(left_encodings))
            right_scores = self.f_split(torch.stack(right_encodings))
            split_scores = left_scores + right_scores
            # split_scores = dy.reshape(split_scores, (len(left_encodings),))
            split_scores = split_scores.view(len(left_encodings))

            if is_train:
                oracle_splits = gold.oracle_splits(left, right)
                oracle_split = min(oracle_splits)
                oracle_split_index = oracle_split - (left + 1)
                split_scores = augment(split_scores, oracle_split_index)

            # split_scores_np = split_scores.npvalue()
            split_scores_np = split_scores.data.cpu().numpy()
            argmax_split_index = int(split_scores_np.argmax())
            argmax_split = argmax_split_index + (left + 1)

            if is_train:
                split = argmax_split if explore else oracle_split
                split_loss = (
                    split_scores[argmax_split_index] -
                    split_scores[oracle_split_index]
                    # if argmax_split != oracle_split else dy.zeros(1))
                    if argmax_split != oracle_split else to_variable(torch.zeros(1)))
            else:
                split = argmax_split
                split_loss = split_scores[argmax_split_index]

            left_trees, left_loss = helper(left, split)
            right_trees, right_loss = helper(split, right)

            children = left_trees + right_trees
            if label:
                children = [trees.InternalParseNode(label, children)]

            return children, label_loss + split_loss + left_loss + right_loss

        children, loss = helper(0, len(sentence))
        assert len(children) == 1
        tree = children[0]
        if is_train and not explore:
            assert gold.convert().linearize() == tree.convert().linearize()
        return tree, loss

# class ChartParser(object):
#     def __init__(
#             self,
#             model,
#             tag_vocab,
#             word_vocab,
#             label_vocab,
#             tag_embedding_dim,
#             word_embedding_dim,
#             lstm_layers,
#             lstm_dim,
#             label_hidden_dim,
#             dropout,
#     ):
#         self.spec = locals()
#         self.spec.pop("self")
#         self.spec.pop("model")

#         self.model = model.add_subcollection("Parser")
#         self.tag_vocab = tag_vocab
#         self.word_vocab = word_vocab
#         self.label_vocab = label_vocab
#         self.lstm_dim = lstm_dim

#         self.tag_embeddings = self.model.add_lookup_parameters(
#             (tag_vocab.size, tag_embedding_dim))
#         self.word_embeddings = self.model.add_lookup_parameters(
#             (word_vocab.size, word_embedding_dim))

#         self.lstm = dy.BiRNNBuilder(
#             lstm_layers,
#             tag_embedding_dim + word_embedding_dim,
#             2 * lstm_dim,
#             self.model,
#             dy.VanillaLSTMBuilder)

#         self.f_label = Feedforward(
#             self.model, 2 * lstm_dim, [label_hidden_dim], label_vocab.size - 1)

#         self.dropout = dropout

#     def param_collection(self):
#         return self.model

#     @classmethod
#     def from_spec(cls, spec, model):
#         return cls(model, **spec)

#     def parse(self, sentence, gold=None):
#         is_train = gold is not None

#         if is_train:
#             self.lstm.set_dropout(self.dropout)
#         else:
#             self.lstm.disable_dropout()

#         embeddings = []
#         for tag, word in [(START, START)] + sentence + [(STOP, STOP)]:
#             tag_embedding = self.tag_embeddings[self.tag_vocab.index(tag)]
#             if word not in (START, STOP):
#                 count = self.word_vocab.count(word)
#                 if not count or (is_train and np.random.rand() < 1 / (1 + count)):
#                     word = UNK
#             word_embedding = self.word_embeddings[self.word_vocab.index(word)]
#             embeddings.append(dy.concatenate([tag_embedding, word_embedding]))

#         lstm_outputs = self.lstm.transduce(embeddings)

#         @functools.lru_cache(maxsize=None)
#         def get_span_encoding(left, right):
#             forward = (
#                 lstm_outputs[right][:self.lstm_dim] -
#                 lstm_outputs[left][:self.lstm_dim])
#             backward = (
#                 lstm_outputs[left + 1][self.lstm_dim:] -
#                 lstm_outputs[right + 1][self.lstm_dim:])
#             return dy.concatenate([forward, backward])

#         @functools.lru_cache(maxsize=None)
#         def get_label_scores(left, right):
#             non_empty_label_scores = self.f_label(get_span_encoding(left, right))
#             return dy.concatenate([dy.zeros(1), non_empty_label_scores])

#         def helper(force_gold):
#             if force_gold:
#                 assert is_train

#             chart = {}

#             for length in range(1, len(sentence) + 1):
#                 for left in range(0, len(sentence) + 1 - length):
#                     right = left + length

#                     label_scores = get_label_scores(left, right)

#                     if is_train:
#                         oracle_label = gold.oracle_label(left, right)
#                         oracle_label_index = self.label_vocab.index(oracle_label)

#                     if force_gold:
#                         label = oracle_label
#                         label_score = label_scores[oracle_label_index]
#                     else:
#                         if is_train:
#                             label_scores = augment(label_scores, oracle_label_index)
#                         label_scores_np = label_scores.npvalue()
#                         argmax_label_index = int(
#                             label_scores_np.argmax() if length < len(sentence) else
#                             label_scores_np[1:].argmax() + 1)
#                         argmax_label = self.label_vocab.value(argmax_label_index)
#                         label = argmax_label
#                         label_score = label_scores[argmax_label_index]

#                     if length == 1:
#                         tag, word = sentence[left]
#                         tree = trees.LeafParseNode(left, tag, word)
#                         if label:
#                             tree = trees.InternalParseNode(label, [tree])
#                         chart[left, right] = [tree], label_score
#                         continue

#                     if force_gold:
#                         oracle_splits = gold.oracle_splits(left, right)
#                         oracle_split = min(oracle_splits)
#                         best_split = oracle_split
#                     else:
#                         best_split = max(
#                             range(left + 1, right),
#                             key=lambda split:
#                                 chart[left, split][1].value() +
#                                 chart[split, right][1].value())

#                     left_trees, left_score = chart[left, best_split]
#                     right_trees, right_score = chart[best_split, right]

#                     children = left_trees + right_trees
#                     if label:
#                         children = [trees.InternalParseNode(label, children)]

#                     chart[left, right] = (
#                         children, label_score + left_score + right_score)

#             children, score = chart[0, len(sentence)]
#             assert len(children) == 1
#             return children[0], score

#         tree, score = helper(False)
#         if is_train:
#             oracle_tree, oracle_score = helper(True)
#             assert oracle_tree.convert().linearize() == gold.convert().linearize()
#             correct = tree.convert().linearize() == gold.convert().linearize()
#             loss = dy.zeros(1) if correct else score - oracle_score
#             return tree, loss
#         else:
#             return tree, score
