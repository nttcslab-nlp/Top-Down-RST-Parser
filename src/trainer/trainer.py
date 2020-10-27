import sys
import torch
from tqdm import tqdm
from pathlib import Path
from nltk import Tree
import json

from trainer.score import Score
from trainer.checkpointer import Checkpointer
from evaluate.rsteval import rst_parseval


class Trainer():
    def __init__(self, config, model, optimizer, scheduler, train_iter, valid_iter, fields):
        self._epochs = config.epochs
        self._disable_tqdm = config.disable_tqdm
        self._log_file = config.log_file
        self._serialization_dir = Path(config.serialization_dir)
        self._checkpointer = Checkpointer(self._serialization_dir,
                                          config.keep_all_serialized_models,
                                          config.model_name)

        save_minimum = not config.maximize_metric
        self._score = Score(config.metric, save_minimum=save_minimum)

        self._train_iter = train_iter
        self._valid_iter = valid_iter

        self._model = model
        self._optimizer = optimizer
        self._max_grad_norm = config.grad_clipping
        self._scheduler = scheduler

        self._fields = fields
        self._config = config

        self._start_epoch = 1

        # call at end of "__init__"
        if self._checkpointer.get_latast_checkpoint() is not None:
            self.load_checkpoint()

    def load_checkpoint(self):
        checkpoint_path = self._checkpointer.get_latast_checkpoint()
        device = torch.device('cpu') if self._config.cpu else torch.device('cuda:0')
        checkpoint = self._checkpointer.restore(checkpoint_path, device)
        self._start_epoch = checkpoint['epoch'] + 1
        self._model.load_state_dict(checkpoint['model'])
        self._optimizer.load_state_dict(checkpoint['optim'])
        self._scheduler.load_state_dict(checkpoint['sched'])
        # self._train_iter.load_state_dict(checkpoint['iter'])
        self._score.load_state_dict(checkpoint['score'])
        print('train from checkpoint: {}'.format(checkpoint_path), file=sys.stderr)

    def run(self):
        for epoch in range(self._start_epoch, self._epochs+1):
            train_loss = self.train(self._model, self._train_iter, self._optimizer,
                                    self._max_grad_norm, self._disable_tqdm)
            valid_score = self.valid(self._model, self._valid_iter, self._disable_tqdm)
            self._scheduler.step()

            self._score.append(valid_score, epoch)
            self._save(epoch, self._score.is_best(epoch))

            scores = {
                'train/loss': train_loss,
                'valid/score': valid_score,
            }
            self._report(epoch, scores)

        return

    @staticmethod
    def train(model, _iter, optimizer, max_grad_norm, disable_tqdm=False):
        total_loss = 0
        total_norm = 0
        model.train()
        for batch in tqdm(_iter, desc='training', ncols=128, disable=disable_tqdm):
            optimizer.zero_grad()
            output_dict = model(batch)
            loss = output_dict["loss"]
            if loss.item() == 0:
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            loss = output_dict["loss"]
            total_loss += loss.item() * len(batch)
            total_norm += len(batch)

        return {'loss': total_loss / total_norm}

    @staticmethod
    def valid(model, _iter, disable_tqdm=False):
        model.eval()
        gold_trees = []
        pred_trees = []
        for batch in tqdm(_iter, desc='validation', ncols=128, disable=disable_tqdm):
            with torch.no_grad():
                gold_trees.extend(batch.tree)
                batch.tree = None
                output_dict = model(batch)
                pred_trees.extend(output_dict['tree'])

        pred_trees = [Tree.fromstring(tree) for tree in pred_trees]
        gold_trees = [Tree.fromstring(tree.linearize()) for tree in gold_trees]
        score_dict = {}
        for eval_type in ['span', 'ns', 'relation', 'full']:
            score_dict[eval_type] = rst_parseval(pred_trees, gold_trees, eval_type)

        return score_dict

    def _save(self, epoch, is_best):
        model_state = {
            "epoch": epoch,
            "model": self._model.state_dict(),
            "optim": self._optimizer.state_dict(),
            "sched": self._scheduler.state_dict(),
            "iter": self._train_iter.state_dict(),
            "score": self._score.state_dict(),
            "fields": self._fields,
            "config": self._config,
        }
        self._checkpointer.save(epoch, model_state, is_best)
        return

    def get_best_model_path(self):
        return self._checkpointer.get_best_model_path()

    def _report(self, epoch, scores):
        json_text = json.dumps({'epoch': epoch, 'scores': scores}, indent=4)
        print(json_text, file=sys.stderr)

        with open(self._serialization_dir / self._log_file, "a") as f:
            print(json_text, file=f)

        return
