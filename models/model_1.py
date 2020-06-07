import nlp
import transformers
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2Model
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

NUM_EPOCHS = 20

class IMDBDataset:
    def __init__(self, part):
        # for simplicity, load all three dataset partitions ({train, test, unsupervised})
        # but only expose the relevant one
        self._raw_dataset = nlp.load_dataset('imdb')
        self.dataset = self._raw_dataset['train']
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
        self.flip = lambda tensor: torch.transpose(tensor, *(1, 0))
    
    def forward(self, tokens):
        hidden_states, _ = self.gpt2_model(tokens)
        final_hidden_state = hidden_states[-1]
        out = self.head(final_hidden_state)
        out = self.flip(out)
        return out

def get_dataloader():
    dataset = IMDBDataset('train')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    return dataloader

def get_model():
    return IMDBSentimentClassificationModel()

def train():
    model = get_model()
    model.cuda()
    model.train()

    dataloader = get_dataloader()

    loss_fn = nn.NLLLoss()
    optimizer = Adam(model.parameters())
    
    writer = SummaryWriter(f'/spell/tensorboards/model_1')

    for epoch in range(1, NUM_EPOCHS + 1):
        losses = []

        for idx, batch in enumerate(dataloader):
            tokens, labels = batch['text'], batch['label']
            tokens = tokens.cuda()
            labels = labels.cuda()

            model.zero_grad()
            y_pred = model(tokens)
            
            loss = loss_fn(y_pred, labels)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if idx % 10 == 0:
                print(f"epoch {epoch}, batch {idx} training loss: {losses[-1]}")

        print(
            f'Finished epoch {epoch}. '
            f'Avg Loss: {np.mean(losses)}; Median Loss: {np.min(losses)}.\n'
        )
        
        checkpoints_dir = "/spell/checkpoints/"
        if not os.path.exists(checkpoints_dir):
            os.mkdir(checkpoints_dir)        
        torch.save(model.state_dict(), f"/spell/checkpoints/model_{epoch}.pth")

train()
