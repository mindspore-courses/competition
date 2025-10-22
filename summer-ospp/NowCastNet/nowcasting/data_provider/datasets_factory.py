import os

import numpy as np
from nowcasting.data_provider.dataset import Dataset

import mindspore.dataset as ds


class RadarData:
    def __init__(self, data_params, module_name='generation'):
        self.data_params = data_params
        self.module_name = module_name
        case_list = os.listdir(os.path.join(self.data_params.dataset_path, "test"))
        self.case_list = [os.path.join(self.data_params.dataset_path, "test", x) for x in case_list]

    def __len__(self):
        return len(self.case_list)

    def __getitem__(self, item):
        data = np.load(self.case_list[item])
        if self.module_name == 'generation':
            inp, evo = data['ori'], data['evo'] / 128
            data_ = inp[0, :self.data_params.input_length], evo[0], inp[0, self.data_params.input_length:]
        else:
            data_ = data['ori'][0, :self.data_params.input_length + self.data_params.gen_oc]
        return data_

class NowcastDataset(Dataset):
    def __init__(self, dataset_generator, module_name='generation', distribute=False, num_workers=1, shuffle=False):
        super(NowcastDataset, self).__init__(dataset_generator, distribute, num_workers, shuffle)
        self.module_name = module_name

    def create_dataset(self, batch_size):
        ds.config.set_prefetch_size(1)
        if self.module_name == 'generation':
            dataset = ds.GeneratorDataset(self.dataset_generator,
                                            ['inputs', 'evo', 'labels'],
                                            shuffle=self.shuffle,
                                            num_parallel_workers=self.num_workers)
        else:
            dataset = ds.GeneratorDataset(self.dataset_generator,
                                ['inputs'],
                                shuffle=self.shuffle,
                                num_parallel_workers=self.num_workers)
        if self.distribute:
            distributed_sampler_train = ds.DistributedSampler(self.rank_size, self.rank_id)
            dataset.use_sampler(distributed_sampler_train)
        dataset_batch = dataset.batch(batch_size=batch_size, drop_remainder=True,
                                      num_parallel_workers=self.num_workers)
        return dataset_batch