import math
import functools

import torch
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate

from skempi import SkempiDataset, TestDataset, SARSDataset


DEFAULT_PAD_VALUES = {
    'aa': 21, 
    'chain_nb': -1, 
    'chain_id': ' ', 
}


class PaddingCollate(object):

    def __init__(self, mode='train', length_ref_key='aa', pad_values=DEFAULT_PAD_VALUES):
        super().__init__()
        self.length_ref_key = length_ref_key
        self.pad_values = pad_values
        self.mode = mode

    @staticmethod
    def _pad_last(x, n, value=0):
        if isinstance(x, torch.Tensor):
            assert x.size(0) <= n
            if x.size(0) == n:
                return x
            pad_size = [n - x.size(0)] + list(x.shape[1:])
            pad = torch.full(pad_size, fill_value=value).to(x)
            return torch.cat([x, pad], dim=0)
        elif isinstance(x, list):
            pad = [value] * (n - len(x))
            return x + pad
        else:
            return x

    @staticmethod
    def _get_common_keys(list_of_dict):
        keys = set(list_of_dict[0].keys())
        for d in list_of_dict[1:]:
            keys = keys.intersection(d.keys())
        return keys

    def _get_pad_value(self, key):
        if key not in self.pad_values:
            return 0
        return self.pad_values[key]

    def __call__(self, data_list):
        if self.mode != 'test':
            max_length = max([data[self.length_ref_key].size(0) for data in data_list])
            max_length = math.ceil(max_length / 8) * 8
            keys = self._get_common_keys(data_list)

            data_list_padded = []
            for data in data_list:
                data_padded = {k: self._pad_last(v, max_length, value=self._get_pad_value(k)) for k, v in data.items() if k in keys}
                data_list_padded.append(data_padded)

            batch = default_collate(data_list_padded)
            return batch
        
        data_list_wt = [data[0] for data in data_list]
        max_length = max([data[self.length_ref_key].size(0) for data in data_list_wt])
        max_length = math.ceil(max_length / 8) * 8
        keys = self._get_common_keys(data_list_wt)

        data_list_wt_padded = []
        for data in data_list_wt:
            data_padded = {k: self._pad_last(v, max_length, value=self._get_pad_value(k)) for k, v in data.items() if k in keys}
            data_list_wt_padded.append(data_padded)

        data_list_mt = [data[1] for data in data_list]
        max_length = max([data[self.length_ref_key].size(0) for data in data_list_mt])
        max_length = math.ceil(max_length / 8) * 8
        keys = self._get_common_keys(data_list_mt)

        data_list_mt_padded = []
        for data in data_list_mt:
            data_padded = {k: self._pad_last(v, max_length, value=self._get_pad_value(k)) for k, v in data.items() if k in keys}
            data_list_mt_padded.append(data_padded)

        batch = default_collate([data_list_wt_padded, data_list_mt_padded])
        return batch


def inf_iterator(iterable):
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


class SkempiDatasetManager(object):
    def __init__(self, config, num_cvfolds, num_workers=4):
        super().__init__()
        self.config = config
        self.num_cvfolds = num_cvfolds
        self.num_workers = num_workers
        
        self.train_loaders = []
        self.val_loaders = []
        
        for fold in range(num_cvfolds):
            train_loader, val_loader = self.init_loaders(fold)
            self.train_loaders.append(train_loader)
            self.val_loaders.append(val_loader)
        self.pretrain_loader = self.pretrain_loaders()

    def pretrain_loaders(self):
        dataset = functools.partial(
            SkempiDataset,
            csv_path = "../data/SKEMPI_v2/skempi_v2.csv",
            pdb_dir = "../data/SKEMPI_v2/PDBs",
            cache_dir = "../data/SKEMPI_v2_aug",
            config = self.config,
            num_cvfolds = self.num_cvfolds,
            cvfold_index = 0,
            split_seed = self.config.split_seed,
            patch_size = self.config.patch_size
        )
        train_dataset = dataset(split='aug')

        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            collate_fn=PaddingCollate(), 
            num_workers=self.num_workers
        )
        train_iterator = inf_iterator(train_loader)
        
        print('Pre-Train %d'% (len(train_dataset)))
        return train_iterator

    def init_loaders(self, fold):
        dataset = functools.partial(
            SkempiDataset,
            csv_path = "../data/SKEMPI_v2/skempi_v2.csv",
            pdb_dir = "../data/SKEMPI_v2/PDBs",
            cache_dir = "../data/SKEMPI_v2_cache",
            config = self.config,
            num_cvfolds = self.num_cvfolds,
            cvfold_index = fold,
            split_seed = self.config.split_seed,
            patch_size = self.config.patch_size
        )
        val_dataset = dataset(split='val')

        if self.config.ab_mode == -1:
            train_dataset = dataset(split='train')  
            train_cplx = set([e['complex'] for e in train_dataset.entries])
            val_cplx = set([e['complex'] for e in val_dataset.entries])
            leakage = train_cplx.intersection(val_cplx)
            assert len(leakage) == 0, f'data leakage {leakage}'
        else:
            train_dataset = dataset(split='all')

        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            collate_fn=PaddingCollate(), 
            num_workers=self.num_workers
        )
        train_iterator = inf_iterator(train_loader)
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            collate_fn=PaddingCollate(), 
            num_workers=self.num_workers
        )

        if self.config.ab_mode == -1:
            print('Fold %d: Train %d, Val %d, All %d' % (fold, len(train_dataset), len(val_dataset), len(train_dataset) + len(val_dataset)))
        else:
            print('Fold %d: Train %d, Val %d, All %d' % (fold, len(train_dataset), len(val_dataset), len(train_dataset)))
        return train_iterator, val_loader

    def get_train_loader(self, fold):
        return self.train_loaders[fold]

    def get_val_loader(self, fold):
        return self.val_loaders[fold]
    

def get_test_data(config):
    dataset = TestDataset(config, "../data/test_data")
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, collate_fn=PaddingCollate())

    return loader


def get_SARS_data(args, config):
    dataset = SARSDataset(args, mutations=config.mutations)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=PaddingCollate())

    return loader