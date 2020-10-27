import h5py
import json
import torch


class HDF():
    def __init__(self, hdf_file, embed_size=None):
        self.hdf = self.open_hdf(hdf_file)
        self.stoi = json.loads(self.hdf.get('sentence_to_index')[0])

    def open_hdf(self, fname):
        return h5py.File(fname, 'r')

    def stov(self, sentence, device):
        key = self.stoi[sentence]
        vector = torch.tensor(self.hdf.get(key).value, device=device)
        return vector

    @staticmethod
    def save_hdf(hdf_path, s2v):
        assert isinstance(s2v, dict)
        hdf5_file = h5py.File(hdf_path, 'w')
        sentence_index_dataset = hdf5_file.create_dataset(
            'sentence_to_index', (1,), dtype=h5py.special_dtype(vlen=str))
        stoi = {s: str(i) for i, s in enumerate(s2v.keys())}
        sentence_index_dataset[0] = json.dumps(stoi)
        for sentence, idx in stoi.items():
            vector = s2v[sentence]
            if isinstance(vector, torch.Tensor):
                vector = vector.detach().cpu().numpy()
            hdf5_file.create_dataset(idx,
                                     vector.shape, dtype='float32',
                                     data=vector)
        hdf5_file.close()

    # @staticmethod
    # def vectors(hdf_file, vocab, dim):
    #     vectors = torch.Tensor(len(vocab), dim, dtype=torch.float)
    #     with h5py.File(hdf_file, 'r') as h5py_file:
    #         stoi = json.loads(h5py_file.get('sentence_to_index')[0])
    #         for i, sentence in enumerate(vocab.itos):
    #             if sentence == '<pad>':
    #                 vectors[i] = torch.Tensor.zero_(vectors[i])
    #             else:
    #                 key = stoi[sentence]
    #                 vectors[i] = torch.tensor(h5py_file.get(key).value.reshape(-1))
    #     return vectors
