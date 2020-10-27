import numpy as np


class Score():
    def __init__(self, metric, save_minimum=True):
        self.metric = metric
        self.save_minimum = save_minimum
        if save_minimum:
            init_score = np.inf
        else:
            init_score = -1 * np.inf

        self.best_score = init_score
        self.best_epoch = -1
        self.scores = []

    def append(self, score, epoch):
        if self.save_minimum:
            is_best = score[self.metric] < self.best_score
        else:
            is_best = score[self.metric] > self.best_score

        if is_best:
            self.best_score = score[self.metric]
            assert self.best_epoch < epoch
            self.best_epoch = epoch

        self.scores.append(score)

    def is_best(self, epoch):
        return epoch == self.best_epoch

    def state_dict(self):
        return {
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'scores': self.scores
        }

    def load_state_dict(self, state_dict):
        self.best_score = state_dict['best_score']
        self.best_epoch = state_dict['best_epoch']
        self.scores = state_dict['scores']
