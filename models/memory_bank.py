from typing import Optional, Tuple

import torch

class TorchMemoryBank:
    def __init__(self, num_classes, max_size):
        self.num_classes = int(num_classes)
        self.max_size = int(max_size)        
        self.device = torch.device('cpu')
        self.bank: Optional[torch.Tensor] = None
        self.feature_shape: Optional[Tuple[int, ...]] = None        
        self.ptr = torch.zeros(self.num_classes, dtype=torch.long, device=self.device)
        self.count = torch.zeros(self.num_classes, dtype=torch.long, device=self.device)
        
    def _init_bank(self, sample_shape):
        self.feature_shape = tuple(sample_shape)
        self.bank = torch.zeros((self.num_classes, self.max_size, *self.feature_shape), device=self.device)

    def add(self, samples, labels):
        assert samples.device == self.device, f"Device mismatch : samples({samples.device}) vs {self.__class__.__name__}({self.device})"
        assert labels.device == self.device, f"Device mismatch : labels({labels.device}) vs {self.__class__.__name__}({self.device})"
        
        if self.bank is None:
            sample_shape = samples.shape[1:]
            self._init_bank(sample_shape)
        
        B = labels.shape[0]
        for batch_idx in range(B):
            curr_label = labels[batch_idx].item()
            curr_ptr = self.ptr[curr_label].item()
            self.bank[curr_label, curr_ptr] = samples[batch_idx]
            self.ptr[curr_label] = (curr_ptr + 1) % self.max_size
            if self.count[curr_label] < self.max_size:
                self.count[curr_label] += 1

    def sample(self, labels, num_samples):
        assert self.feature_shape is not None and self.bank is not None, f"Bank is not initialized yet."
        B = labels.shape[0]
        sample_indices = torch.empty((B, num_samples), dtype=torch.long, device=self.device)
        for b in range(B):
            curr_label = int(labels[b])
            valid = int(self.count[curr_label])
            if valid <= 0:
                sample_indices[b] = torch.zeros(num_samples, dtype=torch.long, device=self.device)
            else:
                sample_indices[b] = torch.randint(valid, (num_samples, ), dtype=torch.long, device=self.device)
        
        res = self.bank[labels.unsqueeze(1), sample_indices]    # (B, num_samples, ...)
        return res
    
    
    def to(self, device):
        # assert isinstance(device, torch.device), f"device is not in torch.device datatype. ({type(device)})"
        self.device = torch.device(device)
        if self.bank is not None:
            self.bank = self.bank.to(device)
        self.ptr = self.ptr.to(device)
        self.count = self.count.to(device)
        
        return self


if __name__ == '__main__':
    B = 30
    shape = (10, 64)   
    num_classes = 3
    device = torch.device('cuda')
    
    samples = torch.randn((B, *shape), device=device)
    labels = torch.randint(num_classes, (B,), device=device)
    print(samples.shape, labels)
    
    A = TorchMemoryBank(num_classes, max_size=10)
    A.to(device)
    print(A.device)
    
    A.add(samples, labels)
    
    sample_labels = torch.tensor([0,1,2], dtype=torch.long, device=device)
    for _ in range(1):
        curr_sample = A.sample(sample_labels, 40)
        print(curr_sample.shape)