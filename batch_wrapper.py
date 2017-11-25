# coding: utf-8
class batch_dataset:
    
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.data_num = dataset.shape[0]
        self.batch_num = int(self.data_num / batch_size)
        self.iter = 0
    
    def next_batch(self):
        if(self.iter == self.batch_num):
            self.iter = 0
        batch = self.dataset[self.iter*self.batch_size:(self.iter + 1)*self.batch_size]
        self.iter += 1
        return batch