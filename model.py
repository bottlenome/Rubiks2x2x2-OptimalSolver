import enum
import torch
from torch.nn import Transformer
from torch.utils.tensorboard import SummaryWriter
import math
import tqdm

class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=12):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class RubiksSolver(torch.nn.Module):
    def __init__(self, d_model: int = 512, nhead: int = 8, num_layers: int = 6):
        super().__init__()
        self.transformer = Transformer(d_model, nhead, num_layers)
        self.initial_embedding = torch.nn.Linear(24, d_model)
        self.final_deembedding = torch.nn.Linear(d_model, 24)
        self.pos_encoder_src = PositionalEncoding(d_model, max_len=1)
        self.pos_encoder_tgt = PositionalEncoding(d_model, max_len=11)
        self.d_model = d_model

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src = self.initial_embedding(src)
        src = self.pos_encoder_src(src)
        """"
        embed_tgt = torch.zeros(tgt.shape[0], tgt.shape[1], self.d_model)
        for i in range(tgt.shape[0]):
            embed_tgt[i] = self.initial_embedding(tgt[i])
        embed_tgt = self.pos_encoder_tgt(embed_tgt)
        """
        embed_tgt = self.initial_embedding(tgt)
        out = self.transformer(src, embed_tgt)
        out = self.final_deembedding(out)
        return out

def train(model, train_data_loader, test_data_loader, n_epochs=100):
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    writer = SummaryWriter()

    with tqdm.tqdm(range(n_epochs)) as epoch_pbar:
        for epoch in epoch_pbar:
            epoch_pbar.set_description(f"Epoch {epoch + 1}/{n_epochs}")
            model.train()
            with tqdm.tqdm(enumerate(train_data_loader),
                           total=len(train_data_loader),
                           leave=False) as pbar:
                total_loss = 0
                for i, (src, tgt) in pbar:
                    optimizer.zero_grad()
                    src = src.cuda()
                    tgt = tgt.cuda()
                    # Reshape data to (Seq Length, Batch Size)
                    tgt = tgt.permute(1, 0, 2)
                    out = model(src, tgt)
                    # Outputs: (Seq Length, Batch, faces)
                    loss = loss_fn(out, tgt)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    pbar.set_postfix({'loss': loss.item()})
                writer.add_scalar("Loss/train", total_loss / len(train_data_loader), epoch)
                writer.add_scalar("Learning rate", optimizer.param_groups[0]['lr'], epoch)
            if epoch % 10 == 0:
                model.eval()
                loss = 0
                with torch.no_grad(), tqdm.tqdm(enumerate(test_data_loader),
                                                total=len(test_data_loader),
                                                leave=False) as test_pbar:
                    test_pbar.set_description(f"Test")
                    for i, (src, tgt) in test_pbar:
                        src = src.cuda()
                        tgt = tgt.cuda()
                        tgt = tgt.permute(1, 0, 2)
                        out = model(src, tgt)
                        loss += loss_fn(out, tgt)
                        test_pbar.set_postfix({'loss': loss.item() / (i + 1)})
                    writer.add_scalar("Loss/test", loss / len(test_data_loader), epoch)
        
if __name__=="__main__":
    model = RubiksSolver()
    src = torch.rand(1, 32, 24)
    tgt = torch.rand(11, 32, 24)
    out = model(src, tgt)
    print(out.shape)
    from data_loader import StateLoader
    train_data_loader, test_data_loader = StateLoader(train_rate=0.998, batch_size=1024)
    train(model.cuda(), train_data_loader, test_data_loader)