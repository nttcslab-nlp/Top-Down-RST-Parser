import os
import sys
import torch
import glob


class Checkpointer():
    def __init__(self,
                 serialization_dir,
                 keep_all_serialized_models,
                 prefix):
        self._serialization_dir = serialization_dir
        os.makedirs(serialization_dir, exist_ok=True)
        self._keep_all_serialized_models = keep_all_serialized_models
        self._best_model_path = None
        self.prefix = prefix

    def save(self, epoch, model_state, is_best):
        model_path = os.path.join(
            self._serialization_dir,
            "{}_state_epoch_{}.th".format(self.prefix, epoch)
        )
        torch.save(model_state, model_path)
        print('save model:', model_path, file=sys.stderr)

        if not self._keep_all_serialized_models:
            previous_model_path = os.path.join(
                self._serialization_dir,
                "{}_state_epoch_{}.th".format(self.prefix, epoch-1)
            )
            if os.path.isfile(previous_model_path):
                os.remove(previous_model_path)

        if is_best:
            model_path = os.path.join(
                self._serialization_dir,
                "{}_best_state_epoch_{}.th".format(self.prefix, epoch)
            )
            torch.save(model_state, model_path)
            print('save best model:', model_path, file=sys.stderr)

            previous_best_model_path = self._best_model_path
            if previous_best_model_path is not None and \
               os.path.isfile(previous_best_model_path):
                os.remove(previous_best_model_path)

            self._best_model_path = model_path

        return

    @classmethod
    def restore(cls, model_path, device):
        model_state = torch.load(model_path, device)
        return model_state

    def get_best_model_path(self):
        return self._best_model_path

    def get_latast_checkpoint(self):
        model_path = glob.glob(os.path.join(
            self._serialization_dir,
            "{}_state_epoch_{}.th".format(self.prefix, '*')))

        if len(model_path) == 0:
            return None
        if len(model_path) == 1:
            return model_path[0]
        else:
            # extract epoch number
            epochs = [int(p.split('_')[-1].split('.th')[0]) for p in model_path]
            latest_model_path = zip(epochs, model_path)[-1]
            return latest_model_path
