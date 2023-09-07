import torch
import torch.nn as nn
import torch.nn.functional as F
import data_loader
import math
import sys
from datetime import datetime, timedelta

class IntervalCheckPoint():
    def __init__(self, interval_min=10):
        self.interval = interval_min
        self.pre = None

    def checkpoint(self):
        if self.pre is None:
            self.pre = datetime.now()
            return True
        elif self.pre + timedelta(minutes=self.interval) < datetime.now():
            self.pre = datetime.now()
            return True
        else:
            return False

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
    def __init__(self, d_model=128, mode="base"):
        super().__init__()
        self.map = nn.Linear(d_model, d_model)
        self.relu = nn.ReLU()
        self.blacket_product = nn.Linear(d_model*2, d_model)

        def blacket(a, b):
            return self.relu(self.blacket_product(torch.cat([a, b], dim = 1)))

        self.blacket = blacket

        if mode == "base":
            def lie_func(i, src, context):
                if i == 0:
                    x = torch.zeros_like(src[0])
                    y = src[0]
                else:
                    x = src[i - 1]
                    y = src[i]
                context = context + x + blacket(x, y)
                return context, context
        elif mode == "1_without_context":
            def lie_func(i, src, context):
                if i == 0:
                    x = torch.zeros_like(src[0])
                    y = src[0]
                else:
                    x = src[i - 1]
                    y = src[i]
                return context, y + blacket(x, y)
        elif mode == "2_context_forget":
            self.alpha = nn.Parameter(torch.tensor(0.9))
            def lie_func(i, src, context):
                if i == 0:
                    x = torch.zeros_like(src[0])
                    y = src[0]
                else:
                    x = src[i - 1]
                    y = src[i]
                context = self.alpha * context + self.blacket(x, y)
                r = y + context
                return context, r
        elif mode == "3_context_forget":
            self.alpha = nn.Parameter(torch.tensor(0.9))
            self.beta = nn.Parameter(torch.tensor(0.9))
            def lie_func(i, src, context):
                if i == 0:
                    x = torch.zeros_like(src[0])
                    y = src[0]
                else:
                    x = src[i - 1]
                    y = src[i]
                context = self.alpha * context + self.beta * y + self.blacket(x, y)
                r = context
                return context, r
        elif mode == "4_blacket_rule":
            self.alpha = nn.Parameter(torch.tensor(0.5))
            def lie_func(i, src, context):
                if i == 0:
                    x = torch.ones_like(src[0])
                    y = src[0]
                else:
                    x = src[i - 1]
                    y = src[i]
                context = self.blacket(context.clone().detach(), self.blacket(x, y))
                r = self.alpha * y + (1 - self.alpha) * context
                return context, r
        elif mode == "5_without_context":
            self.alpha = nn.Parameter(torch.tensor(0.5))
            def lie_func(i, src, context):
                if i == 0:
                    x = torch.zeros_like(src[0])
                    y = src[0]
                else:
                    x = src[i - 1]
                    y = src[i]
                return context, self.alpha * y + (1 - self.alpha) * blacket(x, y)
        elif mode == "6_vector_condition":
            def lie_func(i, src, context):
                x = src[0]
                t = src[-1]
                v = blacket(x, t)
                if i == 0:
                    ret = x + v
                else:
                    ret = 2 * x + 2 * v - context
                context = x + 2 * v
                return context, ret
        else:
            raise NotImplementedError(f"mode {mode} is not implemented")

        self.lie_func = lie_func
        self.mode = mode

    def forward(self, src):
        if self.mode == "4_blacket_rule":
            context = torch.ones_like(src[0])
        else:
            context = torch.zeros_like(src[0])
        ret = []
        src = self.relu(self.map(src))
        
        for i in range(src.shape[0]):
            context, r = self.lie_func(i, src, context)
            if i == src.shape[0] - 2:
                a = src[0]
                b = src[1]
                c = src[2]
                jacobi_identity = (self.blacket(a, self.blacket(b, c)) +
                                   self.blacket(b, self.blacket(c, a)) +
                                   self.blacket(c, self.blacket(a, b)))
                r += jacobi_identity
            ret.append(r.clone())
        return torch.stack(ret)


class CubeLieNumNet(nn.Module):
    def __init__(self, d_face=24, d_color=6, d_model=128, n_layers=1, dropout=0.0, mode="base"):
        super().__init__()
        self.embed = torch.nn.Embedding(d_color + 1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.compress = torch.nn.Linear(d_face * d_model, d_model)
        self.map = nn.Sequential()
        for i in range(n_layers):
            self.map.add_module(f"lie{i}", LieNet(d_model, mode=mode))
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


class CubeLieStateNet(nn.Module):
    def __init__(self, d_face=24, d_color=6, d_model=128, n_layers=1, dropout=0.0, mode="base"):
        super().__init__()
        self.embed = torch.nn.Embedding(d_color + 1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.compress = torch.nn.Linear(d_face * d_model, d_model)
        self.map = nn.Sequential()
        for i in range(n_layers):
            self.map.add_module(f"lie{i}", LieNet(d_model, mode=mode))
        self.remap = torch.nn.Linear(d_model, d_face * (d_color + 1))
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
        out = self.remap(out)
        out = out.view(n_seq, -1, self.d_face, self.d_color + 1)
        out = F.softmax(out, dim=3)
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


class StateDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(24, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = x / 84. # normalize
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


def main(d_model=128, n_layers=3,
         data_type="num", learning_method="default", discriminator_lr=0.00005,
         n_interval=1000, n_epochs=100000, n_data=512000, batch_size=512,
         model="lie", lie_mode="base", debug_print=False):
    if data_type == "num":
        if model == "GPT":
            model = GPTNumNet(d_model=d_model, n_layers=n_layers)
        else:
            model = CubeLieNumNet(d_model=d_model, n_layers=n_layers, mode=lie_mode)
    else:
        if model == "GPT":
            raise NotImplementedError("GPT is not implemented for state")
        else:
            model = CubeLieStateNet(d_model=d_model, n_layers=n_layers, mode=lie_mode)
        if learning_method == "GAN":
            discriminator = StateDiscriminator()
            discriminator = discriminator.cuda()
        elif learning_method == "default":
            pass
        else:
            raise NotImplementedError(f"learning method {learning_method} is not implemented")
    if data_type == "num":
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)
        if learning_method == "GAN":
            optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=0.00005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)
    if data_type == "num":
        train_loader, test_loader = data_loader.NumLoader(train_rate=0.9, batch_size=batch_size, size=n_data)
    else:
        train_loader, test_loader = data_loader.StateLoader2(train_rate=0.9, batch_size=batch_size, size=n_data)

    model = model.cuda()
    if data_type == "num":
        loss_fn = torch.nn.MSELoss()
    else:
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
        if learning_method == "GAN":
            loss_fn_d = torch.nn.BCELoss()

    if is_colab():
        from tqdm.notebook import tqdm
    else:
        from tqdm import tqdm
    from torch.utils.tensorboard import SummaryWriter
    n_epochs = n_epochs
    save_interval = n_interval
    writer = SummaryWriter()
    interval = IntervalCheckPoint()
    with open(f"{writer.get_logdir()}/save", "w") as f:
        f.write(f"{sys.argv}\n")

    with tqdm(range(n_epochs)) as pbar:
        for epoch in pbar:
            pbar.set_description(f"Epoch {epoch + 1}/{n_epochs}")
            model.train()
            with tqdm(enumerate(train_loader),
                      total=len(train_loader),
                      leave=False) as pbar_batch:
                total_loss = 0
                total_acc = 0
                total_g_loss = 0
                total_d_loss = 0
                for i, (src, tgt) in pbar_batch:
                    optimizer.zero_grad()
                    src = src.cuda()
                    # batch, seq, face -> seq, batch, face
                    src = src.permute(1, 0, 2)
                    tgt = tgt.cuda()
                    out = model(src)
                    if data_type == "num":
                        loss = loss_fn(out.t(), tgt.type(torch.float))
                        total_acc += (torch.round(out.t()) == tgt).sum().item() / (out.t().shape[0] * out.t().shape[1])
                    else:
                        if learning_method == "GAN":
                            optimizer_d.zero_grad()
                            real_data = src.permute(1, 0, 2).view(-1, 24)
                            fake_data = out.view(-1, 24, 7).argmax(dim=-1)
                            real_label = torch.ones(real_data.shape[0], 1).cuda()
                            fake_label = torch.zeros(fake_data.shape[0], 1).cuda()
                            real_pred = discriminator(real_data)
                            fake_pred = discriminator(fake_data.detach())
                            d_loss = 0.5 * (loss_fn_d(real_pred, real_label) + loss_fn_d(fake_pred, fake_label))

                            real_label = torch.ones(fake_data.shape[0], 1).cuda()
                            g_pred = discriminator(fake_data)
                            g_loss = loss_fn_d(g_pred, real_label)

                        out = out.permute(1, 0, 2, 3)
                        out = out.reshape(-1, out.size(-1))
                        tgt = tgt.view(-1)
                        loss = loss_fn(out, tgt)
                        total_acc += (torch.argmax(out, dim=-1) == tgt).float().mean().item()

                    if learning_method == "GAN":
                        d_loss.backward()
                        optimizer_d.step()
                        loss += g_loss
                        total_d_loss += d_loss.item()
                        total_g_loss += g_loss.item()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                    pbar_batch.set_postfix(loss=total_loss / (i + 1))
                    if i % 1000 == 0 and debug_print:
                        for j in range(5):
                                print(torch.round(out.t()[j]), tgt[j])
                writer.add_scalar("Loss/train", total_loss / len(train_loader), epoch)
                writer.add_scalar("Accuracy/train", total_acc / len(train_loader), epoch)
                for name, param in model.named_parameters():
                    if name.find("alpha") != -1:
                        writer.add_scalar(f"Param/alpha", param.item(), epoch)
                    elif name.find("beta") != -1:
                        writer.add_scalar(f"Param/beta", param.item(), epoch)
                if learning_method == "GAN":
                    writer.add_scalar("Loss/discriminator", total_d_loss / len(train_loader), epoch)
                    writer.add_scalar("Loss/generator", total_g_loss / len(train_loader), epoch)
                    for name, param in discriminator.named_parameters():
                        writer.add_histogram(f'Grad/{name}', param.grad, epoch)
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
                    if data_type == "num":
                        loss += loss_fn(out.t(), tgt.type(torch.float))
                        accu += (torch.round(out.t()) == tgt).sum().item() / (out.t().shape[0] * out.t().shape[1])
                    else:
                        out = out.permute(1, 0, 2, 3)
                        out = out.reshape(-1, out.size(-1))
                        tgt = tgt.view(-1)
                        loss += loss_fn(out, tgt)
                        accu += (torch.argmax(out, dim=-1) == tgt).float().mean().item()
                    test_pbar.set_postfix({'loss': loss.item() / (i + 1)})
                writer.add_scalar("Loss/test", loss / len(test_loader), epoch)
                writer.add_scalar("Accuracy/test", accu / len(test_loader), epoch)
            if epoch % save_interval == 0 or interval.checkpoint():
                save_dict = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                }
                torch.save(save_dict, f"{writer.get_logdir()}/{epoch}.pth")
            # scheduler.step()
    save_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'epoch': epoch,
    }
    torch.save(save_dict, f"{writer.get_logdir()}/{epoch}.pth")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lie")
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--data_type", type=str, default="num")
    parser.add_argument("--learning_method", type=str, default="default")
    parser.add_argument("--discriminator_lr", type=float, default=0.00005)
    parser.add_argument("--n_interval", type=int, default=1000)
    parser.add_argument("--n_epochs", type=int, default=100000)
    parser.add_argument("--n_data", type=int, default=512000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--debug_print", action="store_true")
    parser.add_argument("--lie_mode", type=str, default="base")
    args = parser.parse_args()

    main(d_model=args.d_model, n_layers=args.n_layers,
         data_type=args.data_type, learning_method=args.learning_method,
         discriminator_lr=args.discriminator_lr, n_interval=args.n_interval,
         n_epochs=args.n_epochs, n_data=args.n_data, batch_size=args.batch_size,
         model=args.model, lie_mode=args.lie_mode, debug_print=args.debug_print)