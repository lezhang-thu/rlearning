# -*- coding: utf-8 -*-
from __future__ import absolute_import
import pickle
import argparse
import numpy as np

from os.path import join

import math
import collections

num_actions = 14
virtual_loss = 3
c_puct = 1
num_virtual_threads = 4
num_simulations = 5
global_dtype = tf.float32

BATCH_SIZE = 128


class Node:
    def __init__(
        self,
        action,
        decoder_hidden,
        terminal,
        parent=None,
        Nsa=None,
        Wsa=None,
        Qsa=None,
        Psa=None
    ):
        #state: None is an internal node, not None is a leaf
        self.state = {
            'action': action,
            'decoder_hidden': decoder_hidden,
            'terminal': terminal
        }
        if parent is not None:
            self.parent_info = {
                'parent': parent,
                'statistics': {
                    'Nsa': Nsa,
                    'Wsa': Wsa,
                    'Qsa': Qsa,
                    'Psa': Psa
                }
            }
        else:
            self.parent_info = None
        self.children = None


# Trim voc and pairs
# pairs = trimRareWords(voc, pairs, MIN_COUNT)
from cider_scorer2 import CiderScorer2
refs_ls = []
for pair in pairs:
    output_sentence = pair[1]
    refs_ls.append([output_sentence])

cider_scorer2 = CiderScorer2(refs_ls, n=4, sigma=6.0)


class MCTS:
    def __init__(
        self,
        source_line,
        target_line,
        voc,
        encoder,
        decoder,
        c_puct,
        num_virtual_threads=0,
        virtual_loss=0
    ):
        self.info = {
            'source_line': source_line,
            'target_line': target_line,
            'voc': voc
        }

        # Set dropout layers to eval mode
        encoder.eval()
        decoder.eval()
        self.nw = {'encoder': encoder, 'decoder': decoder}
        encoder_outputs, encoder_hidden = self._init_state(
            source_line, voc, encoder
        )
        self.info['encoder_outputs'] = encoder_outputs  #global

        self.root = Node(
            action=SOS_token,
            decoder_hidden=encoder_hidden[:decoder.n_layers],
            terminal=False
        )

        self.param = {
            'c_puct': c_puct,
            'num_virtual_threads': num_virtual_threads,
            'virtual_loss': virtual_loss
        }

    def _init_encoder_outputs(self, source_line, voc, encoder):
        indexes_batch = [indexesFromSentence(voc, sentence)]
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)

        input_batch = input_batch.to(device)
        lengths = lengths.to(device)

        encoder_outputs, encoder_hidden = encoder(input_batch, lengths)

        return encoder_outputs, encoder_hidden

    def _get_evaluation(self, test):
        #normalize before return
        return cider_scorer2.compute_score(test, [self.target_line]) / 5.0 - 2.0

    def _get_action_prob(self, decoder_input, decoder_hidden):
        decoder_output, decoder_hidden = self.nw['decoder'](
            decoder_input, decoder_hidden, self.info['encoder_outputs']
        )
        return decoder_output, decoder_hidden

    def simulate_many_times(self, num_times):
        assert num_times > 1

        for _ in range(num_times):
            self._simulate_once()

    def _simulate_once(self):
        leaf_ls = []
        for _ in range(self.param['num_virtual_threads']):
            leaf_ls.append(self._select())

        value_ls = self._rollout_ls(leaf_ls)

        for i, leaf in enumerate(leaf_ls):
            self._backup(leaf, value_ls[i])

        leaf_set = set(leaf_ls)
        self._expand_ls(leaf_set)
        for leaf in leaf_set:
            if not leaf.state['terminal'][0]:
                leaf.state = None

    def _select(self):
        s = self.root
        action_seq = []
        while s.state is None:
            visit_total = 0

            action_seq.append(-1)  #update

            for c in s.children:
                visit_total += c.parent_info['statistics']['Nsa']
            max_child = None
            for i, c in enumerate(s.children):
                if max_child is None or (
                    max_child.parent_info['statistics']['Qsa'] +
                    self.param['c_puct'] * max_child.parent_info['statistics']
                    ['Psa'] * math.sqrt(visit_total) /
                    (1 + max_child.parent_info['statistics']['Nsa'])
                ) < (
                    c.parent_info['statistics']['Qsa'] +
                    self.param['c_puct'] * c.parent_info['statistics']['Psa'] *
                    math.sqrt(visit_total) /
                    (1 + c.parent_info['statistics']['Nsa'])
                ):
                    max_child = c
                    action_seq[-1] = i  #update
            s = max_child
            #'the rollout statistics are updated as if it has lost n_vl games'
            s.parent_info['statistics']['Nsa'] += self.param['virtual_loss']
            s.parent_info['statistics']['Wsa'] -= self.param['virtual_loss']
            s.parent_info['statistics']['Qsa'] = s.parent_info['statistics'][
                'Wsa'
            ] / s.parent_info['statistics']['Nsa']

        return {'leaf': s, 'action_seq': action_seq}

    def _rollout_ls(self, leaf_ls):
        #initialize
        init_slots = []
        for _ in range(4):
            init_slots.append([None] * len(leaf_ls))
        decoder_input_ls, decoder_hidden_ls, terminal_ls, action_seq_ls = init_slots

        #notice the np.copy
        for i, leaf in enumerate(leaf_ls):
            decoder_input_ls[i], decoder_hidden_ls[i], terminal_ls[
                i
            ], action_seq_ls[i] = (
                torch.ones(1, 1, device=device, dtype=torch.long
                          ) * leaf_ls[i]['leaf'].state['action'],
                leaf_ls[i]['leaf'].state['decoder_hidden'],
                leaf_ls[i]['leaf'].state['terminal'], leaf_ls[i]['action_seq']
            )
        index_ls = [_ for _ in range(len(leaf_ls))]

        final_action_ls = [None] * len(leaf_ls)

        while len(decoder_input_ls) > 0:
            #purge
            init_slots = []
            for _ in range(5):
                init_slots.append(list())
            purge_input_ls, purge_hidden_ls, purge_terminal_ls, purge_action_ls, purge_index = init_slots

            for i, terminal in enumerate(terminal_ls):
                if not terminal:
                    purge_input_ls.append(decoder_input_ls[i])
                    purge_hidden_ls.append(decoder_hidden_ls[i])
                    purge_terminal_ls.append(terminal_ls[i])
                    purge_action_ls.append(action_seq_ls[i])

                    purge_index.append(index_ls[i])
                else:
                    final_action_ls[index_ls[i]] = action_seq_ls[i]

            if len(purge_input_ls) == 0:
                break

            #compute action distributions
            action_ls = [None] * len(purge_input_ls)
            for i in range(len(purge_input_ls) // BATCH_SIZE + 1):
                start, end = (
                    i * BATCH_SIZE,
                    min((i + 1) * BATCH_SIZE, len(purge_input_ls))
                )

                #                 import sys
                #                 inp = input('continue? (y/n)')
                #                 if inp == 'n':
                #                     raise KeyboardInterrupt
                action_prob_ls, purge_hidden_ls[
                    start:end
                ] = self._get_action_prob(
                    torch.cat(purge_input_ls[start:end]),
                    torch.cat(purge_hidden_ls[start:end])
                )
                purge_input_ls[start:end] = torch.multinomial(
                    action_prob_ls, num_samples=1
                )
                action_ls[start:end] = [
                    purge_input_ls[k][0].item()
                    for k in range(len(action_prob_ls))
                ]

            for i in range(action_ls):
                if action_ls[i] == EOS_token:
                    purge_terminal_ls[i] = True
                else:
                    purge_action_ls.append(action_ls[i])

            decoder_input_ls, decoder_hidden_ls, terminal_ls = (
                purge_input_ls, purge_hidden_ls, purge_terminal_ls
            )
            index_ls = purge_index
            action_seq_ls = purge_action_ls

        value_ls = [None] * len(leaf_ls)
        for i in range(len(leaf_ls)):
            action_seq = final_action_ls[i]
            test = ' '.join(self.info['voc'].index2word(k) for k in action_seq)
            value_ls[i] = self._get_evaluation(test)

        return value_ls

    def _expand_ls(self, leaf_ls):
        init_slots = []
        for _ in range(3):
            init_slots.append(list())
        purge_leaf_ls, purge_input_ls, purge_hidden_ls = init_slots

        for leaf in leaf_ls:
            if not leaf.state['terminal']:
                purge_leaf_ls.append(leaf)
                purge_input_ls.append(
                    torch.ones(1, 1, device=device, dtype=torch.long) *
                    leaf.state['action']
                )
                purge_hidden_ls.append(leaf.state['decoder_hidden'])

        if len(purge_leaf_ls) == 0:
            return

        action_prob_ls = [None] * len(purge_leaf_ls)
        for i in range(len(purge_img_ls) // BATCH_SIZE + 1):
            start, end = (
                i * BATCH_SIZE, min((i + 1) * BATCH_SIZE, len(purge_img_ls))
            )
            action_prob_ls[start:end
                          ], purge_hidden_ls[start:end] = self._get_action_prob(
                              torch.cat(purge_input_ls[start:end]),
                              torch.cat(purge_hidden_ls[start:end])
                          )

        for i, leaf in enumerate(purge_leaf_ls):
            leaf.children = list()
            for k in range(num_actions):
                leaf.children.append(
                    Node(
                        action=k,
                        decoder_hidden=purge_hidden_ls[k],
                        terminal=False,
                        parent=leaf,
                        Nsa=0,
                        Wsa=0,
                        Qsa=0,
                        Psa=action_prob_ls[i][k].item()
                    )
                )
            leaf.children[EOS_token].state['terminal'] = True

    def _backup(self, leaf, value):
        s = leaf
        while s.parent_info is not None:
            s.parent_info['statistics']['Nsa'
                                       ] += (-self.param['virtual_loss'] + 1)
            s.parent_info['statistics'
                         ]['Wsa'] += (self.param['virtual_loss'] + value)
            s.parent_info['statistics']['Qsa'] = s.parent_info['statistics'][
                'Wsa'
            ] / s.parent_info['statistics']['Nsa']
            s = s.parent_info['parent']

    def reach_terminal(self):
        if self.root.state is not None and self.root.state['terminal']:
            return True
        return False

    def act_one_step(self):
        #select the action with maximum visit count
        if self.root.state is not None:  #a leaf node
            return -1

        #root is an internal node
        max_child = None
        action = -1
        for i, c in enumerate(self.root.children):
            if max_child is None or max_child.parent_info['statistics'][
                'Nsa'
            ] < c.parent_info['statistics']['Nsa']:
                max_child = c
                action = i
        self.root = max_child
        self.root.parent_info = None
        return action

    def debug_print(self):
        if self.root.state is not None:
            print('reach a leaf')
        else:
            for i, c in enumerate(self.root.children):
                print(
                    'Nsa {}, Wsa {}, Qsa {}, Psa {}'.format(
                        c.parent_info['statistics']['Nsa'],
                        c.parent_info['statistics']['Wsa'],
                        c.parent_info['statistics']['Qsa'],
                        c.parent_info['statistics']['Psa']
                    )
                )
                print('above info for child {}'.format(i))
            print('end of the printing this time ...')


def train(
    input_variable,
    lengths,
    target_variable,
    mask,
    max_target_len,
    encoder,
    decoder,
    encoder_optimizer,
    decoder_optimizer,
    batch_size,
    clip,
    max_length=MAX_LENGTH
):

    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Initialize variables
    loss = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Forward batch of sequences through decoder one time step at a time
    for t in range(max_target_len):
        decoder_output, decoder_hidden = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        # Teacher forcing: next input is current target
        decoder_input = target_variable[t].view(1, -1)
        # Calculate and accumulate loss
        mask_loss, _ = maskNLLLoss(decoder_output, target_variable[t], mask[t])
        loss += mask_loss

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    def trainIters(
        voc, training_data, encoder, decoder, encoder_optimizer,
        decoder_optimizer, n_iteration, batch_size, clip
    ):

        # Load batches for each iteration
        training_batches = [
            batch2TrainData(
                voc, [random.choice(training_data) for _ in range(batch_size)]
            ) for _ in range(n_iteration)
        ]

        for iteration in range(n_iteration):
            training_batch = training_batches[iteration]
            # Extract fields from batch
            input_variable, lengths, target_variable, mask, max_target_len = training_batch

            # Run a training iteration with batch
            train(
                input_variable, lengths, target_variable, mask, max_target_len,
                encoder, decoder, encoder_optimizer, decoder_optimizer,
                batch_size, clip
            )


import random

loadFilename = 'loadFilename'
checkpoint = torch.load(loadFilename)
# If loading a model trained on GPU to CPU
#checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))

encoder_sd = checkpoint['en']
decoder_sd = checkpoint['de']
encoder_optimizer_sd = checkpoint['en_opt']
decoder_optimizer_sd = checkpoint['de_opt']
embedding_sd = checkpoint['embedding']

# Initialize word embeddings
embedding = nn.Embedding(voc.num_words, hidden_size)
embedding.load_state_dict(embedding_sd)

# Initialize encoder & decoder models
encoder = EncoderRNN(hidden_size, embedding, encoder_n_layers, dropout)
decoder = LuongAttnDecoderRNN(
    attn_model, embedding, hidden_size, voc.num_words, decoder_n_layers, dropout
)

encoder.load_state_dict(encoder_sd)
decoder.load_state_dict(decoder_sd)

# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)

# Ensure dropout layers are in train mode
encoder.train()
decoder.train()

encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(
    decoder.parameters(), lr=learning_rate * decoder_learning_ratio
)
encoder_optimizer.load_state_dict(encoder_optimizer_sd)
decoder_optimizer.load_state_dict(decoder_optimizer_sd)

player_train = {
    'embedding': embedding,
    'encoder': encoder,
    'decoder': decoder,
    'encoder_optimizer': encoder_optimizer,
    'decoder_optimizer': decoder_optimizer
}

player_generate = dict()
player_generate['embedding'] = nn.Embedding(voc.num_words, hidden_size)

player_generate['encoder'] = EncoderRNN(
    hidden_size, player_generate['embedding'], encoder_n_layers, dropout
)
player_generate['decoder'] = LuongAttnDecoderRNN(
    attn_model, player_generate['embedding'], hidden_size, voc.num_words,
    decoder_n_layers, dropout
)
# Set dropout layers to eval mode
player_genearte['encoder'].eval()
player_generate['decoder'].eval()

while True:
    player_generate['embedding'].load_state_dict(
        player_train['embedding'].state_dict()
    )
    player_generate['encoder'].load_state_dict(
        player_train['encoder'].state_dict()
    )
    player_generate['decoder'].load_state_dict(
        player_train['decoder'].state_dict()
    )

    while True:
        #player_generate generates training data ...
        training_data = list()
        num_self_play = 1000
        for _ in range(num_self_play):
            pair = random.choice(pairs)
            mcts_tree = MCTS(
                pair[0], pair[1], voc, player_generate['encoder'],
                player_generate['decoder'], c_puct, num_virtual_threads,
                virtual_loss
            )
            action_seq = list()
            while not mcts_tree.reach_terminal():
                mcts_tree.simulate_many_times(num_simulations)
                #        mcts_tree.debug_print()
                action_seq.append(mcts_tree.act_one_step())
            target = ' '.join(
                self.info['voc'].index2word(k) for k in action_seq
            )
            training_data.append(pair[0], target)

        #player_train is trained over training data ...
        trainIters(
            voc, training_data, encoder, decoder, encoder_optimizer,
            decoder_optimizer, n_iteration, batch_size, clip
        )

        #evaluate player_generate and play_train
