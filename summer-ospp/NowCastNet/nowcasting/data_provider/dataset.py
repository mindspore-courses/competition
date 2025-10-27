import mindspore.dataset as ds
from mindspore.communication import get_rank, get_group_size

class Dataset:
    def __init__(self,
                 dataset_generator, distribute=False, num_workers=1, shuffle=True):
        self.distribute = distribute
        self.num_workers = num_workers
        self.dataset_generator = dataset_generator
        self.shuffle = shuffle

        if distribute:
            self.rank_id = get_rank()
            self.rank_size = get_group_size()

    def create_dataset(self, batch_size):
        ds.config.set_prefetch_size(1)
        dataset = ds.GeneratorDataset(self.dataset_generator,
                                      ['inputs', 'labels'],
                                      shuffle=self.shuffle,
                                      num_parallel_workers=self.num_workers)
        if self.distribute:
            distributed_sampler_train = ds.DistributedSampler(self.rank_size, self.rank_id)
            dataset.use_sampler(distributed_sampler_train)

        dataset_batch = dataset.batch(batch_size=batch_size, drop_remainder=True,
                                      num_parallel_workers=self.num_workers)
        return dataset_batch
