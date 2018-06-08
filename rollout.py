# -*- coding:utf-8 -*-

import copy

import torch
import torch.nn.functional as F
from torch.distributions import Categorical


class Rollout:
    """Roll-out policy"""

    def __init__(self, max_sentence_length, corpus, parent):
        self.embed = corpus
        self.lstm = None
        self.max_sentence_length = max_sentence_length
        self.output_linear = None
        # self.lstm = torch.nn.LSTM(input_size=corpus.embed_size,
        #                           hidden_size=parent.input_encoding_size,
        #                           num_layers=parent.num_layers,
        #                           batch_first=True,
        #                           dropout=float(parent.dropout))
        #
        # self.output_linear = torch.nn.Linear(parent.input_encoding_size, corpus.vocab_size)

    def reward2(self, generated, image_features, hidden, monte_carlo_count, evaluator, steps=1):
        assert monte_carlo_count % steps == 0, "Monte Carlo Count can't be divided by Steps"
        monte_carlo_count //= steps

        with torch.no_grad():
            batch_size = generated.size(0)
            result = torch.zeros(batch_size).cuda()
            remaining = self.max_sentence_length - generated.shape[1]
            h, c = hidden
            h, c = (h.unsqueeze(2).repeat(1, 1, monte_carlo_count, 1).view(h.shape[0], -1, h.shape[-1]),
                    c.unsqueeze(2).repeat(1, 1, monte_carlo_count, 1).view(c.shape[0], -1, c.shape[-1]))
            generated = generated.unsqueeze(1).repeat(1, monte_carlo_count, 1, 1).view(
                generated.shape[0] * monte_carlo_count, generated.shape[1], -1)
            image_features = image_features.unsqueeze(1).repeat(1, monte_carlo_count, 1).view(-1,
                                                                                              image_features.shape[-1])
            for _ in range(steps):
                hidden = (h, c)
                inputs = generated[:, -1].unsqueeze(1)
                current_generated = generated
                for i in range(remaining):
                    _, hidden = self.lstm(inputs, hidden)
                    outputs = self.output_linear(hidden[0]).squeeze(0)
                    outputs = F.softmax(outputs, -1)
                    predicted = outputs.multinomial(1)
                    # m = Categorical(outputs)
                    # predicted = m.sample()
                    # embed the next inputs, unsqueeze is required cause of shape (batch_size, 1, embedding_size)
                    inputs = self.embed.word_embeddings_from_indices(predicted.view(-1).cpu().data.numpy()).unsqueeze(
                        1).cuda()
                    current_generated = torch.cat([current_generated, inputs], dim=1)
                reward = evaluator(image_features,
                                   # image_features.repeat(monte_carlo_count, 1),
                                   current_generated)
                # reward = reward.view(batch_size, monte_carlo_count, -1).sum(1)
                # reward = reward[::monte_carlo_count].sum(1)
                # reward = torch.stack([reward[i::batch_size].sum() for i in range(monte_carlo_count)])
                # t = reward
                # reward = torch.stack([reward[i::batch_size].sum() for i in range(batch_size)])
                # reward = reward.view(-1, batch_size).sum(0)
                # reward = reward.view(monte_carlo_count, -1).sum(0)
                reward = reward.view(-1, monte_carlo_count).sum(1)
                # reward = reward.view(batch_size, -1).sum(1)
                result += reward
                result /= monte_carlo_count
            return result

    def reward(self, generated, image_features, hidden, monte_carlo_count, evaluator, steps=1):
        with torch.no_grad():
            batch_size = generated.size(0)

            result = torch.zeros(batch_size, 1).cuda()
            remaining = self.max_sentence_length - generated.shape[1]
            original_hidden = hidden

            for j in range(monte_carlo_count):
                current_generated = generated
                hidden = original_hidden
                inputs = generated[:, -1].view(batch_size, 1, -1)

                for i in range(remaining):
                    _, hidden = self.lstm(inputs, hidden)
                    outputs = self.output_linear(hidden[0]).squeeze(0)
                    outputs = F.softmax(outputs, -1)
                    predicted = outputs.multinomial(1)
                    # embed the next inputs, unsqueeze is required cause of shape (batch_size, 1, embedding_size)
                    inputs = self.embed.word_embeddings_from_indices(
                        predicted.view(-1).cpu().data.numpy()).unsqueeze(1).cuda()
                    current_generated = torch.cat([current_generated, inputs], dim=1)

                    if self.embed.word_from_index(predicted[0, 0].item()) == self.embed.END_SYMBOL:
                        if self.max_sentence_length - current_generated.shape[1] > 0:
                            pad = torch.stack([self.embed.word_embedding(self.embed.PAD)] * (self.max_sentence_length - current_generated.shape[1])).cuda()
                            current_generated = torch.cat([current_generated, pad.unsqueeze(0)], dim=1)
                        break

                reward = evaluator(image_features, current_generated)
                reward = reward.view(batch_size, -1)
                result += reward
            result /= monte_carlo_count
        return result

    def update(self, original_model):
        self.lstm = copy.deepcopy(original_model.lstm)
        self.lstm.flatten_parameters()
        self.output_linear = copy.deepcopy(original_model.output_linear)
