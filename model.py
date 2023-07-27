import enum
from os import write
import torch
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer
from torch.utils.tensorboard import SummaryWriter
import math
import tqdm

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # seq, embedding -> seq, batch, embedding
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class RubiksSolver(torch.nn.Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_layers: int = 6):
        super().__init__()
        d_vocab = 6 + 1
        self.embed = torch.nn.Embedding(d_vocab, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_len=500)
        self.transformer = Transformer(d_model, nhead, num_layers)
        self.unembed = torch.nn.Linear(d_model, d_vocab)
        self.d_model = d_model

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src = self.embed(src)
        src = self.pos_encoder(src)
        tgt = self.embed(tgt)
        tgt = self.pos_encoder(tgt)
        # tgt_key_padding_mask = (tgt == -100).all(dim=-1)
        # tgt_key_padding_mask = tgt_key_padding_mask.permute(1, 0)
        # out = self.transformer(src, tgt, tgt_key_padding_mask=tgt_key_padding_mask)
        out = self.transformer(src, tgt)
        out = self.unembed(out)
        return out

class GPTRubiksSolver(torch.nn.Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_layers: int = 6):
        super().__init__()
        d_vocab = 6 + 1
        self.embed = torch.nn.Embedding(d_vocab, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_len=500)
        encoder_layer = TransformerEncoderLayer(d_model, nhead)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        self.unembed = torch.nn.Linear(d_model, d_vocab)
        self.d_model = d_model

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.embed(src)
        src = self.pos_encoder(src)
        out = self.transformer_encoder(src)
        out = self.unembed(out)
        return out

def train(model, train_data_loader, test_data_loader, n_epochs=100):
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
    writer = SummaryWriter()

    with tqdm.tqdm(range(n_epochs)) as epoch_pbar:
        for epoch in epoch_pbar:
            epoch_pbar.set_description(f"Epoch {epoch + 1}/{n_epochs}")
            model.train()
            with tqdm.tqdm(enumerate(train_data_loader),
                           total=len(train_data_loader),
                           leave=False) as pbar:
                total_loss = 0
                total_acc = 0
                for i, (src, tgt) in pbar:
                    optimizer.zero_grad()
                    src = src.cuda()
                    # batch, face -> seq, batch
                    src = src.t()
                    tgt = tgt.cuda()
                    # batch, seq, faces -> batch, seq
                    tgt = tgt.view(tgt.size(0), -1)
                    out = model(src, tgt.t())
                    # Outputs: (Seq Length, Batch, face_class)
                    loss = loss_fn(out.permute(1, 0, 2).reshape(-1, out.size(2)),
                                   tgt.view(-1))
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    total_acc += (out.argmax(dim=-1).t() == tgt).all(dim=-1).float().mean().item()
                    pbar.set_postfix({'loss': loss.item()})
                writer.add_scalar("Loss/train", total_loss / len(train_data_loader), epoch)
                writer.add_scalar("Accuracy/train", total_acc / len(train_data_loader), epoch)
                writer.add_scalar("Learning rate", optimizer.param_groups[0]['lr'], epoch)
            if epoch % 10 == 0:
                model.eval()
                loss = 0
                acc = 0
                with torch.no_grad(), tqdm.tqdm(enumerate(test_data_loader),
                                                total=len(test_data_loader),
                                                leave=False) as test_pbar:
                    test_pbar.set_description(f"Test")
                    for i, (src, tgt) in test_pbar:
                        src = src.cuda()
                        src = src.t()
                        tgt = tgt.cuda()
                        tgt = tgt.view(tgt.size(0), -1)
                        out = model(src, tgt.t())
                        loss += loss_fn(out.permute(1, 0, 2).reshape(-1, out.size(2)),
                                        tgt.view(-1))
                        acc += (out.argmax(dim=-1).t() == tgt).all(dim=-1).float().mean().item()
                        test_pbar.set_postfix({'loss': loss.item() / (i + 1)})
                    writer.add_scalar("Loss/test", loss / len(test_data_loader), epoch)
                    writer.add_scalar("Accuracy/test", acc / len(test_data_loader), epoch)
        
if __name__=="__main__":
    p = PositionalEncoding(d_model=512, max_len=11*24)
    print("pe.shape", p.pe.shape)
    model = RubiksSolver()
    src = torch.randint(low=1, high=7, size=(32, 24))
    # batch, face -> seq, batch
    src = src.t()
    tgt = torch.randint(low=1, high=7, size=(32, 10, 24))
    # batch, seq, faces -> seq, batch
    tgt = tgt.view(tgt.size(0), -1).t()
    out = model(src, tgt)

    model = GPTRubiksSolver()
    print(src.shape)
    out = model(src)
    print(out.shape)

    model = RubiksSolver()
    from data_loader import StateLoader
    train_data_loader, test_data_loader = StateLoader(train_rate=0.8, batch_size=32, size=50000)
    train(model.cuda(), train_data_loader, test_data_loader)