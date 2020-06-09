import numpy as np
import nlp
import transformers
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import os

# NEW
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

# NEW
def init_process(rank, size, backend='gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)

class IMDBDataset:
    def __init__(self, part):
        self.dataset = nlp.load_dataset('imdb')['train']
        self.tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
    
    def __getitem__(self, idx):
        review = self.dataset[idx]
        label = torch.tensor(review['label'])
        text = torch.tensor(self.tokenizer.encode(review['text']))
        # The default GPT2 token length is 1024. The IMBD text review corpus is pretty long, and
        # the GPT2 BPE tokenizer is pretty verbose, so we exceed this character limit in ~3% of
        # cases. Since this is simple benchmark we are ignoring this problem (ConstantPad1d
        # just clips the last few out words out).
        text = nn.ConstantPad1d((1, 1024 - text.shape[0] - 1), 0)(text)
        return {'text': text, 'label': label}
    
    def __len__(self):
        return self.dataset.num_rows


class IMDBSentimentClassificationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gpt2_config = transformers.GPT2Config()
        self.gpt2_model = transformers.GPT2Model(self.gpt2_config)
        self.head = nn.Sequential(*[
            nn.Linear(768, 2**6),
            nn.Linear(2**6, 2**4),
            nn.Linear(2**4, 2),
            nn.LogSoftmax(dim=0)
        ])
    
    def forward(self, tokens):
        hidden_states, _ = self.gpt2_model(tokens)
        final_hidden_state = hidden_states[:, -1, :]
        out = self.head(final_hidden_state)
        return out


def get_dataloader(rank, world_size):
    dataset = IMDBDataset('train')    
    
    # NEW
    sampler = DistributedSampler(dataset, rank=rank, num_replicas=world_size, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=4, sampler=sampler)
    
    return dataloader

def get_model():
    return IMDBSentimentClassificationModel()

def train(rank, num_epochs, world_size):
    # NEW
    init_process(rank, world_size)
    print(f"Rank {rank}/{world_size} training process initialized.\n")
        
    # NEW
    # Since this is a single-instance multi-GPU training script, it's important
    # that only one process handle downloading of the data, to avoid race conditions
    # implicit in having multiple processes attempt to write to the same file
    # simultaneously.
    if rank == 0:
        nlp.load_dataset('imdb')
        transformers.GPT2Tokenizer.from_pretrained('gpt2')
    dist.barrier()
    print(f"Rank {rank}/{world_size} training process passed data download barrier.\n")
    
    model = get_model()
    model.cuda(rank)
    model.train()

    # NEW
    model = DistributedDataParallel(model, device_ids=[rank])
    
    dataloader = get_dataloader(rank, world_size)

    loss_fn = nn.NLLLoss()
    optimizer = Adam(model.parameters())
    
    writer = SummaryWriter(f'/spell/tensorboards/model_2')

    for epoch in range(1, num_epochs + 1):
        losses = []

        for idx, batch in enumerate(dataloader):
            tokens, labels = batch['text'], batch['label']
            tokens = tokens.cuda(rank)
            labels = labels.cuda(rank)

            model.zero_grad()
            y_pred = model(tokens)
            
            loss = loss_fn(y_pred, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if idx % 10 == 0:
                print(
                    f'Finished epoch {epoch}, rank {rank}/{world_size}, batch {idx}. '
                    f'Loss: {loss:.3f}.\n'
                )
            if rank == 0:
                writer.add_scalar('training loss', loss)
            losses.append(loss)

        print(
            f'Finished epoch {epoch}, rank {rank}/{world_size}. '
            f'Avg Loss: {np.mean(losses)}; Median Loss: {np.min(losses)}.\n'
        )
        
        if rank == 0:
            if not os.path.exists('/spell/checkpoints/'):
                os.mkdir('/spell/checkpoints/')
            torch.save(model.state_dict(), f'/spell/checkpoints/model_{epoch}.pth')

# NEW
NUM_EPOCHS = 20
WORLD_SIZE = torch.cuda.device_count()
def main():
    mp.spawn(train,
        args=(NUM_EPOCHS, WORLD_SIZE),
        nprocs=WORLD_SIZE,
        join=True)

if __name__=="__main__":
    main()
