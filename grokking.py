import random
import numpy as np
import math
import torch
import torch.nn as nn
from torch.nn import Transformer, Embedding
from torch.nn import TransformerEncoder, TransformerEncoderLayer


def gen_train_test(frac_train, num, seed=0):
    # Generate train and test split
    pairs = [[i, j, num] for i in range(num) for j in range(num)]
    random.seed(seed)
    random.shuffle(pairs)
    div = int(frac_train*len(pairs))
    return pairs[:div], pairs[div:]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 10):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class SimpleFormer(nn.Module):
    def __init__(self, d_vocab, d_model=128, nhead=4, num_encoder_layers=1,
                 num_decoder_layers=1, dropout=0.0):
        super().__init__()
        self.embed = Embedding(d_vocab + 1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.transformer = Transformer(d_model, nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers)
        self.unembed = nn.Linear(d_model, d_vocab)
    
    def forward(self, src, tgt):
        src = self.embed(src)
        src = self.pos_encoder(src)
        tgt = self.embed(tgt)
        tgt = self.pos_encoder(tgt)
        # tgt_mask = torch.ones(tgt.shape[0], tgt.shape[0]).bool().cuda()
        # out = self.transformer(src, tgt, tgt_mask=tgt_mask)
        out = self.transformer(src, tgt)
        out = self.unembed(out)
        return out

class GPT(nn.Module):
    def __init__(self, d_vocab, d_model=128, nhead=4, num_layers=6, dropout=0.0):
        super().__init__()
        self.embed = torch.nn.Embedding(d_vocab + 1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*4, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers, norm=None)
        self.unembed = torch.nn.Linear(d_model, d_vocab)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.embed(src)
        src = self.pos_encoder(src)
        out = self.transformer_encoder(src)
        out = self.unembed(out)
        return out


class AccelNet(nn.Module):
    def __init__(self, d_vocab, d_model=128, dropout=0.0):
        super().__init__()
        self.embed = torch.nn.Embedding(d_vocab + 1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.dropout = nn.Dropout(p=dropout)
        self.vector_layer = torch.nn.Linear(d_model, d_model)
        self.accel_layer = torch.nn.Linear(d_model, d_model)
        self.unembed = torch.nn.Linear(d_model, d_vocab)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.embed(src)
        src = self.pos_encoder(src)
        src = src.transpose(0, 1)
        x = src
        v = self.vector_layer(x)
        a = self.accel_layer(x)
        out = self.unembed(self.dropout(x + v + a))
        out = out.transpose(0, 1)
        return out

class LieNet(nn.Module):
    def __init__(self, d_model=128, n_head=4):
        super().__init__()
        self.map = nn.Linear(d_model, d_model)
        self.relu = nn.ReLU()
        self.blacket_product = nn.Linear(d_model*2, d_model)

    def forward(self, src):
        context = torch.zeros_like(src[0])
        ret = []
        src = self.relu(self.map(src))

        def blacket(a, b):
            return self.relu(self.blacket_product(torch.cat([a, b], dim = 1)))

        for i in range(src.shape[0] - 1):
            tmp = blacket(src[i], src[i + 1])
            context += src[i] + tmp
            if i == src.shape[0] - 1:
                a = src[0]
                b = src[1]
                c = src[2]
                jacobi_identity = (blacket(a, blacket(b, c)) +
                                   blacket(b, blacket(c, a)) +
                                   blacket(c, blacket(a, b)))
                context += jacobi_identity
            ret.append(context)
        return torch.stack(ret)


class LieClassify(nn.Module):
    def __init__(self, d_vocab, d_model=128, dropout=0.0):
        super().__init__()
        self.embed = torch.nn.Embedding(d_vocab + 1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.map = LieNet()
        self.unembed = torch.nn.Linear(d_model, d_vocab)

    def forward(self, src):
        src = self.embed(src)
        src = self.pos_encoder(src)
        out = self.map(src)
        out = self.unembed(out)
        return out


def main():
    frac_train = 0.3 # default 0.3
    p = 113
    seed = 0

    train, test = gen_train_test(frac_train, p, seed)
    train_result = [[p, (i+j)%p] for i, j, _ in train]
    test_result =  [[p, (i+j)%p] for i, j, _ in test]

    # model = SimpleFormer(p, d_model=128, nhead=8, num_encoder_layers=1, num_decoder_layers=3, dropout=0.00)
    # default d_model=128, nhead=4, num_layers=1, d__mlp=n_layers * 4
    # model = GPT(p, d_model=128, nhead=4, num_layers=3, dropout=0.00)
    # model = AccelNet(p, d_model=128, dropout=0.1)
    model = LieClassify(p, d_model=128, dropout=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(step/10, 1))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
    train = torch.tensor(train)
    train_result = torch.tensor(train_result)
    test = torch.tensor(test)
    test_result = torch.tensor(test_result)

    model = model.cuda()
    train = train.cuda()
    train_result = train_result.cuda()
    test = test.cuda()
    test_result = test_result.cuda()

    import tqdm
    from torch.utils.tensorboard import SummaryWriter
    n_epochs = 20000
    save_interval = 1000
    loss = nn.CrossEntropyLoss()
    writer = SummaryWriter()

    with tqdm.tqdm(range(n_epochs)) as p_bar:
        for epoch in p_bar:
            model.train()
            optimizer.zero_grad()
            # ret = model(train.t(), train_result.t())
            ret = model(train.t())
            # ret = ret[0]
            ret = ret[-1, :, :]
            # loss_val = loss(ret, train_result[:, 1].view(-1))
            loss_val = loss(ret, train_result[:, 1])
            loss_val.backward()
            optimizer.step()
            p_bar.set_description(f"Epoch {epoch + 1}/{n_epochs}, loss={loss_val.item():.4f}")
            writer.add_scalar('Loss/train', loss_val.item(), epoch)
            writer.add_scalar('Accuracy/train', (ret.argmax(dim=-1) == train_result[:, 1]).sum() / len(train_result), epoch)
            with torch.no_grad():
                model.eval()
                # ret = model(test.t(), test_result[:,0].view(-1, 1).t())
                ret = model(test.t())
                # ret = ret.permute(1, 0, 2).view(-1, p)
                ret = ret[-1, :, :]
                loss_val = loss(ret, test_result[:, 1])
                writer.add_scalar('Loss/test', loss_val.item(), epoch)
                for i in range(len(test_result)):
                    print('Test', f'({test[i, 0]} + {test[i, 1]}) % {p} = {ret[i].argmax(dim=-1)} == {test_result[i, 1]}', epoch)
                    if i >= 10:
                        break
                writer.add_scalar('Accuracy/test', (ret.argmax(dim=-1) == test_result[:, 1]).sum() / len(test_result), epoch)
            if epoch % save_interval == 0:
                save_dict = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'train_loss': loss_val.item(),
                    'test_loss': loss_val.item(),
                    'epoch': epoch,
                }
                torch.save(save_dict, f"save/results_{epoch}.pth")
    save_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'train_loss': loss_val.item(),
        'test_loss': loss_val.item(),
        'epoch': epoch,
    }
    torch.save(save_dict, f"save/results_{epoch}.pth")

if __name__ == '__main__':
    model = GPT(123, d_model=128, nhead=8, num_layers=3, dropout=0.00)
    input = torch.zeros(3, 6284).long()
    model(input)
    main()