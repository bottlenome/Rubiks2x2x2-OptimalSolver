import torch
import torch.nn as nn
import data_loader
import math
import sys


def is_colab():
    module_list = sys.modules
    return 'google.colab' in module_list


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 15):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, 1, d_model)
        pe[:, 0, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class LieNet(nn.Module):
    def __init__(self, d_model=128):
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
        
        for i in range(src.shape[0]):
            # consider src[-1] is solved state
            if i == 0:
                x = torch.zeros_like(src[0])
                y = src[0]
            else:
                x = src[i - 1]
                y = src[i]
            context += x + blacket(x, y)
            if i == src.shape[0] - 2:
                a = src[0]
                b = src[1]
                c = src[2]
                jacobi_identity = (blacket(a, blacket(b, c)) +
                                   blacket(b, blacket(c, a)) +
                                   blacket(c, blacket(a, b)))
                context += jacobi_identity
            ret.append(context.clone())
        return torch.stack(ret)


class CubeLieNumNet(nn.Module):
    def __init__(self, d_face=24, d_color=6, d_model=128, n_layers=1, dropout=0.0):
        super().__init__()
        self.embed = torch.nn.Embedding(d_color + 1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.compress = torch.nn.Linear(d_face * d_model, d_model)
        self.map = nn.Sequential()
        for i in range(n_layers):
            self.map.add_module(f"lie{i}", LieNet(d_model))
        self.calc_num = torch.nn.Linear(d_model, 1)
        self.relu = nn.ReLU()
        self.d_face = d_face
        self.d_color = d_color
        self.d_model = d_model

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        n_seq = len(src)
        src = self.embed(src)
        src = self.pos_encoder(src)
        src = src.view(-1, self.d_face * self.d_model)
        # compress by total faces
        src = self.relu(self.compress(src))
        out = src.view(n_seq, - 1, self.d_model)
        for i in range(len(self.map)):
            out = self.map[i](out)
        out = self.calc_num(out)
        out = out.view(n_seq, -1)
        return out[:-1]

class GPTNumNet(nn.Module):
    def __init__(self, d_face=24, d_color=6, d_model=128, n_head=4, n_layers=6, dropout=0.0):
        super().__init__()
        self.embed = torch.nn.Embedding(d_color + 1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.compress = torch.nn.Linear(d_face * d_model, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=n_head, dim_feedforward=d_model*4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, norm=None)
        self.calc_num = torch.nn.Linear(d_model, 1)
        self.d_face = d_face
        self.d_model = d_model

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        n_seq = len(src)
        src = self.embed(src)
        src = self.pos_encoder(src)
        src = src.view(-1, self.d_face * self.d_model)
        src = self.compress(src)
        out = self.transformer_encoder(src)
        out = self.calc_num(out)
        out = out.view(n_seq, -1)
        return out[:-1]

def main(d_model=128, n_layers=3, n_epochs=100000, n_data=512000, batch_size=512, model="lie", debug_print=False):
    if model == "GPT":
        model = GPTNumNet(d_model=d_model, n_layers=n_layers)
    else:
        model = CubeLieNumNet(d_model=d_model, n_layers=n_layers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
    train_loader, test_loader = data_loader.NumLoader(train_rate=0.9, batch_size=batch_size, size=n_data)

    model = model.cuda()
    loss_fn = torch.nn.MSELoss()

    if is_colab():
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
    from torch.utils.tensorboard import SummaryWriter
    n_epochs = n_epochs
    save_interval = 1000
    writer = SummaryWriter()

    with tqdm(range(n_epochs)) as pbar:
        for epoch in pbar:
            pbar.set_description(f"Epoch {epoch + 1}/{n_epochs}")
            model.train()
            with tqdm(enumerate(train_loader),
                      total=len(train_loader),
                      leave=False) as pbar_batch:
                total_loss = 0
                total_acc = 0
                for i, (src, tgt) in pbar_batch:
                    optimizer.zero_grad()
                    src = src.cuda()
                    # batch, seq, face -> seq, batch, face
                    src = src.permute(1, 0, 2)
                    tgt = tgt.cuda()
                    out = model(src)
                    loss = loss_fn(out.t(), tgt.type(torch.float))
                    loss.backward()
                    total_acc += (torch.round(out.t()) == tgt).sum().item() / (out.t().shape[0] * out.t().shape[1])
                    optimizer.step()
                    total_loss += loss.item()
                    pbar_batch.set_postfix(loss=total_loss / (i + 1))
                    if i % 1000 == 0 and debug_print:
                        for j in range(5):
                                print(torch.round(out.t()[j]), tgt[j])
                writer.add_scalar("Loss/train", total_loss / len(train_loader), epoch)
                writer.add_scalar("Accuracy/train", total_acc / len(train_loader), epoch)
            # eval
            model.eval()
            loss = 0
            accu = 0
            with torch.no_grad(), tqdm(enumerate(test_loader),
                                       total=len(test_loader),
                                       leave=False) as test_pbar:
                test_pbar.set_description(f"Test")
                for i, (src, tgt) in test_pbar:
                    src = src.cuda()
                    src = src.permute(1, 0, 2)
                    tgt = tgt.cuda()
                    out = model(src)
                    loss += loss_fn(out.t(), tgt.type(torch.float))
                    accu += (torch.round(out.t()) == tgt).sum().item() / (out.t().shape[0] * out.t().shape[1])
                    test_pbar.set_postfix({'loss': loss.item() / (i + 1)})
                writer.add_scalar("Loss/test", loss / len(test_loader), epoch)
                writer.add_scalar("Accuracy/test", accu / len(test_loader), epoch)
            if epoch % save_interval == 0:
                save_dict = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                }
                torch.save(save_dict, f"save/results_{epoch}.pth")
            # scheduler.step()
    save_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }
    torch.save(save_dict, f"save/results_{epoch}.pth")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lie")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_epochs", type=int, default=100000)
    parser.add_argument("--n_data", type=int, default=512000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--debug_print", action="store_true")
    args = parser.parse_args()

    main(d_model=args.d_model, n_layers=args.n_layers, n_epochs=args.n_epochs, n_data=args.n_data, batch_size=args.batch_size, model=args.model, debug_print=args.debug_print)