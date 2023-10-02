import torch
import torch.nn as nn
import torch.nn.functional as F
import data_loader
import math
import random
import sys
from datetime import datetime, timedelta

class IntervalCheckPoint():
    def __init__(self, interval_min=60):
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

class BracketNet(nn.Module):
    def __init__(self, d_model=128, mode="probabilistic", map_mode="map_first", rate=0.3):
        super().__init__()
        self.bracket_product = nn.Linear(d_model*2, d_model)
        self.activate = nn.GELU()

        def bracket(a, b):
            return self.activate(self.bracket_product(torch.cat([a, b], dim = 1)))
        self.bracket = bracket
        self.target_function = nn.Sequential(nn.Linear(d_model, d_model * 2),
                                             nn.GELU(),
                                             nn.Linear(d_model * 2, d_model))
        self.layer_norm = nn.LayerNorm(d_model)
        self.mode = mode
        self.rate = rate

    def forward(self, src):
        ret = []
        conditions_results = {0:0, 1:0, 2:0}
        for i in range(src.shape[0]):
            r = self.activate(self.target_function(src[i])) + src[i]
            r = self.layer_norm(r)
            context = r
            # add condition by probabilistic
            if self.train() and random.random() < self.rate:
                # choose 0-2
                target = random.randint(0, 2)
                if i <= target:
                    if target == 0:
                        condition = self.bracket(src[i], src[i]) * 0.1
                    elif target == 1:
                        condition = (context - src[i]) * 0.1
                    elif target == 2:
                        a = src[i - 2]
                        b = src[i - 1]
                        c = src[i]
                        condition = (self.bracket(a, self.bracket(b, c)) +
                                     self.bracket(b, self.bracket(c, a)) +
                                     self.bracket(c, self.bracket(a, b))) * 0.1
                    r = r + condition
                    conditions_results[target] += condition.norm(dim=-1).sum() / (src.shape[0] * src.shape[1])
            ret.append(r.clone())
        ret = torch.stack(ret)
        return ret, conditions_results


class BracketMultitaskNet(nn.Module):
    def __init__(self, d_model=128, mode="probabilistic", map_mode="map_first", rate=0.3):
        super().__init__()
        self.bracket_product = nn.Linear(d_model*2, d_model)
        self.activate = nn.GELU()

        def bracket(a, b):
            return self.activate(self.bracket_product(torch.cat([a, b], dim = 1)))
        self.bracket = bracket
        self.target_function = nn.Sequential(nn.Linear(d_model, d_model * 2),
                                             nn.GELU(),
                                             nn.Linear(d_model * 2, d_model))
        self.layer_norm = nn.LayerNorm(d_model)
        self.mode = mode
        self.rate = rate

    def forward(self, src):
        ret = []
        diffs = []
        conditions_results = {0:0, 1:0, 2:0}
        for i in range(src.shape[0]):
            r = self.activate(self.target_function(src[i])) + src[i]
            r = self.layer_norm(r)
            context = r
            if i >= 1:
                diff = self.bracket(src[i - 1], src[i])
                diffs.append(diff)
            # add condition by probabilistic
            if self.train() and random.random() < self.rate:
                # choose 0-2
                target = random.randint(0, 2)
                if i <= target:
                    if target == 0:
                        condition = self.bracket(src[i], src[i]) * 0.1
                    elif target == 1:
                        condition = (context - src[i]) * 0.1
                    elif target == 2:
                        a = src[i - 2]
                        b = src[i - 1]
                        c = src[i]
                        condition = (self.bracket(a, self.bracket(b, c)) +
                                     self.bracket(b, self.bracket(c, a)) +
                                     self.bracket(c, self.bracket(a, b))) * 0.1
                    r = r + condition
                    conditions_results[target] += condition.norm(dim=-1).sum() / (src.shape[0] * src.shape[1])
            ret.append(r.clone())
        ret = torch.stack(ret)
        diffs = torch.stack(diffs)
        return ret, diffs, conditions_results

class LieNet(nn.Module):
    def __init__(self, d_model=128, mode="base", map_mode="map_first"):
        super().__init__()
        self.map = nn.Linear(d_model, d_model)
        self.activate = nn.GELU()
        self.blacket_product = nn.Linear(d_model*2, d_model)

        def blacket(a, b):
            return self.activate(self.blacket_product(torch.cat([a, b], dim = 1)))

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
        self.map_mode = map_mode

    def forward(self, src):
        if self.mode == "4_blacket_rule":
            context = torch.ones_like(src[0])
        else:
            context = torch.zeros_like(src[0])
        ret = []
        if self.map_mode == "map_first":
            src = self.activate(self.map(src))

        for i in range(src.shape[0]):
            context, r = self.lie_func(i, src, context)
            if i >= 2:
                a = src[i - 2]
                b = src[i - 1]
                c = src[i]
                jacobi_identity = (self.blacket(a, self.blacket(b, c)) +
                                   self.blacket(b, self.blacket(c, a)) +
                                   self.blacket(c, self.blacket(a, b)))
                r += jacobi_identity
            ret.append(r.clone())
        if self.map_mode == "map_last":
            ret = self.activate(self.map(torch.stack(ret)))
        else:
            ret = torch.stack(ret)
        return ret


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
        self.activate = nn.GELU()
        self.d_face = d_face
        self.d_color = d_color
        self.d_model = d_model

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        n_seq = len(src)
        src = self.embed(src)
        src = self.pos_encoder(src)
        src = src.view(-1, self.d_face * self.d_model)
        # compress by total faces
        src = self.activate(self.compress(src))
        out = src.view(n_seq, - 1, self.d_model)
        for i in range(len(self.map)):
            out = self.map[i](out)
        out = self.calc_num(out)
        out = out.view(n_seq, -1)
        return out[:-1]


class CubeLieStateNet(nn.Module):
    def __init__(self, d_face=24, d_color=6, d_model=128, n_layers=1, dropout=0.0, mode="base", map_mode="map_first"):
        super().__init__()
        self.embed = torch.nn.Embedding(d_color + 1, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.compress = torch.nn.Linear(d_face * d_model, d_model)
        self.map = nn.Sequential()
        for i in range(n_layers):
            self.map.add_module(f"lie{i}", LieNet(d_model, mode=mode, map_mode=map_mode))
        self.remap = torch.nn.Linear(d_model, d_face * (d_color + 1))
        self.activate = nn.GELU()
        self.d_face = d_face
        self.d_color = d_color
        self.d_model = d_model

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        n_seq = len(src)
        src = self.embed(src)
        src = self.pos_encoder(src)
        src = src.view(-1, self.d_face * self.d_model)
        # compress by total faces
        src = self.activate(self.compress(src))
        out = src.view(n_seq, - 1, self.d_model)
        for i in range(len(self.map)):
            out = self.map[i](out)
        out = self.remap(out)
        out = out.view(n_seq, -1, self.d_face, self.d_color + 1)
        out = F.softmax(out, dim=3)
        return out[:-1]


class Map(nn.Module):
    def __init__(self, d_in, d_out, n_layers=2):
        super().__init__()
        self.map = nn.Sequential()
        for i in range(n_layers):
            if i == 0:
                self.map.add_module(f"map_pre{i}", torch.nn.Linear(d_in, d_out))
            else:
                self.map.add_module(f"map_pre{i}", torch.nn.Linear(d_out, int(d_out / 2)))
                self.map.add_module(f"map_after{i}", torch.nn.Linear(int(d_out / 2), d_out))
        self.activate = nn.GELU()
        self.layer_norm = nn.Sequential()
        for i in range(n_layers):
            self.layer_norm.add_module(f"layer_norm{i}", nn.LayerNorm(d_out))
        self.n_layers = n_layers

    def forward(self, src):
        for i in range(self.n_layers):
            if i == 0:
                src = self.activate(self.map[0](src))
            else:
                out = self.activate(self.map[2*i - 1](src))
                out = self.activate(self.map[2*i](out))
                src = out + src
            src = self.layer_norm[i](src)
        return src


class ReMap(nn.Module):
    def __init__(self, d_in, d_out, n_layers=2, dropout=0.0):
        super().__init__()
        self.remap = nn.Sequential()
        for i in range(n_layers):
            if i != n_layers - 1:
                self.remap.add_module(f"remap_pre{i}", torch.nn.Linear(d_in, d_in * 2))
                self.remap.add_module(f"remap_after{i}", torch.nn.Linear(d_in * 2, d_in))
            else:
                self.remap.add_module(f"remap_pre{i}", torch.nn.Linear(d_in, d_out))
        self.activate = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.Sequential()
        for i in range(n_layers - 1):
            self.layer_norm.add_module(f"layer_norm{i}", nn.LayerNorm(d_in))
        self.n_layers = n_layers

    def forward(self, src):
        for i in range(self.n_layers):
            if i != self.n_layers - 1:
                src = self.dropout(src)
                out = self.activate(self.remap[2*i](src))
                out = self.dropout(out)
                out = self.activate(self.remap[2*i + 1](out))
                src = out + src
                src = self.layer_norm[i](src)
            else:
                src = self.dropout(src)
                out = self.remap[2*i](src)
        return out

class CubeBracketStateNet(nn.Module):
    def __init__(self, d_face=24, d_color=6, d_model=128, n_layers=2, dropout=0.0, mode="probabilistic"):
        super().__init__()
        self.embed = torch.nn.Embedding(d_color + 1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, 0.0)
        self.map = Map(d_face*d_model, d_model, n_layers=n_layers)
        self.calc = BracketNet(d_model)
        self.remap = ReMap(d_model, d_face * (d_color + 1), n_layers=n_layers, dropout=dropout)
        self.activate = nn.GELU()
        self.d_face = d_face
        self.d_color = d_color
        self.d_model = d_model

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        n_seq = len(src)
        src = self.embed(src)
        src = self.pos_encoder(src)
        src = src.view(-1, self.d_face * self.d_model)
        # compress by total faces
        src = self.map(src)
        out = src.view(n_seq, -1, self.d_model)
        out, conditions_results = self.calc(out)
        out = self.remap(out)
        out = out.view(n_seq, -1, self.d_face, self.d_color + 1)
        out = F.softmax(out, dim=3)
        return out[:-1], conditions_results


class CubeBracketMultitaskNet(nn.Module):
    def __init__(self, d_face=24, d_color=6, d_op=9 , d_model=128, n_layers=2, dropout=0.0, mode="probabilistic"):
        super().__init__()
        self.embed = torch.nn.Embedding(d_color + 1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, 0.0)
        self.map = Map(d_face*d_model, d_model, n_layers=n_layers)
        self.calc = BracketMultitaskNet(d_model)
        self.remap = ReMap(d_model, d_face * (d_color + 1), n_layers=n_layers, dropout=dropout)
        self.op_classify = ReMap(d_model, d_op + 1, n_layers=n_layers, dropout=dropout)
        self.activate = nn.GELU()
        self.d_face = d_face
        self.d_color = d_color
        self.d_model = d_model

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        n_seq = len(src)
        src = self.embed(src)
        src = self.pos_encoder(src)
        src = src.view(-1, self.d_face * self.d_model)
        # compress by total faces
        src = self.map(src)
        out = src.view(n_seq, -1, self.d_model)
        out, ops, conditions_results = self.calc(out)
        out = self.remap(out)
        out = out.view(n_seq, -1, self.d_face, self.d_color + 1)
        out = F.softmax(out, dim=3)
        # [seq, batch, d_model]
        ops = self.op_classify(ops)
        ops = F.softmax(ops, dim=2)
        return out[:-1], ops, conditions_results


class CubeGPTStateNet(nn.Module):
    def __init__(self, d_face=24, d_color=6, d_model=128, n_head=4, n_layers=1, dropout=0.0, mode="base"):
        super().__init__()
        self.embed = torch.nn.Embedding(d_color + 1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        self.compress = torch.nn.Linear(d_face * d_model, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=n_head, dim_feedforward=d_model*4, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, norm=None)
        self.remap = torch.nn.Linear(d_model, d_face * (d_color + 1))
        self.activate = nn.GELU()
        self.d_face = d_face
        self.d_color = d_color
        self.d_model = d_model

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        n_seq = len(src)
        src = self.embed(src)
        src = self.pos_encoder(src)
        src = src.view(-1, self.d_face * self.d_model)
        # compress by total faces
        src = self.activate(self.compress(src))
        out = src.view(n_seq, - 1, self.d_model)
        out = self.transformer_encoder(out)
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
        self.activate = nn.GELU()

    def forward(self, x):
        x = x / 84. # normalize
        x = self.activate(self.fc1(x))
        x = self.activate(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


def main(d_model=128, n_layers=3,
         data_type="num", learning_method="default", discriminator_lr=0.00005,
         dropout=0.0,
         n_interval=1000, n_epochs=100000, n_data=512000, batch_size=512,
         model_name="lie", lie_mode="base", lie_map_mode="map_first", debug_print=False):
    if data_type == "num":
        if model_name == "GPT":
            model = GPTNumNet(d_model=d_model, n_layers=n_layers)
        else:
            model = CubeLieNumNet(d_model=d_model, n_layers=n_layers, mode=lie_mode)
    elif data_type == "multitask":
        if model_name == "bracket":
            model = CubeBracketMultitaskNet(d_model=d_model, n_layers=n_layers, dropout=dropout)
        else:
            raise NotImplementedError(f"model {model_name} is not implemented")
        if learning_method == "default":
            pass
        else:
            raise NotImplementedError(f"learning method {learning_method} is not implemented")
    else:
        if model_name == "GPT":
            model = CubeGPTStateNet(d_model=d_model, n_layers=n_layers)
        elif model_name == "bracket":
            model = CubeBracketStateNet(d_model=d_model, n_layers=n_layers, dropout=dropout)
        else:
            model = CubeLieStateNet(d_model=d_model, n_layers=n_layers, mode=lie_mode, map_mode=lie_map_mode)
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
    if data_type == "multitask":
        train_loader, test_loader = data_loader.AllLoader(train_rate=0.9, batch_size=batch_size, size=n_data)
    else:
        train_loader, test_loader = data_loader.StateLoader2(train_rate=0.9, batch_size=batch_size, size=n_data)

    model = model.cuda()
    if data_type == "num":
        loss_fn = torch.nn.MSELoss()
    elif data_type == "multitask":
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)
        loss_fn_op = torch.nn.CrossEntropyLoss(ignore_index=0)
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
                ops_acc = 0
                conditions_results = {0:0, 1:0, 2:0}
                for i, (src, tgt) in pbar_batch:
                    optimizer.zero_grad()
                    src = src.cuda()
                    # batch, seq, face -> seq, batch, face
                    src = src.permute(1, 0, 2)
                    if data_type == "multitask":
                        ops_result = tgt[1]
                        ops_result = ops_result.cuda()
                        tgt = tgt[0]
                    tgt = tgt.cuda()
                    if data_type == "multitask":
                        out, ops, cr = model(src)
                        for i in range(3):
                            conditions_results[i] += cr[i]
                    elif model_name == "bracket":
                        out, cr = model(src)
                        for i in range(3):
                            conditions_results[i] += cr[i]
                    else:
                        out = model(src)
                    if data_type == "num":
                        loss = loss_fn(out.t(), tgt.type(torch.float))
                        total_acc += (torch.round(out.t()) == tgt).sum().item() / (out.t().shape[0] * out.t().shape[1])
                    if data_type == "multitask":
                        out = out.permute(1, 0, 2, 3)
                        out = out.reshape(-1, out.size(-1))
                        tgt = tgt.view(-1)
                        loss = loss_fn(out, tgt)
                        total_acc += (torch.argmax(out, dim=-1) == tgt).float().mean().item()
                        ops = ops.permute(1, 0, 2)
                        ops = ops.reshape(-1, ops.size(-1))
                        ops_result = ops_result.view(-1)
                        loss_op = loss_fn_op(ops, ops_result)
                        loss += loss_op
                        ops_acc += (torch.argmax(ops, dim=-1) == ops_result).float().mean().item()
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
                if model_name == "bracket":
                    writer.add_scalar("Condition/0", conditions_results[0] / len(train_loader), epoch)
                    writer.add_scalar("Condition/1", conditions_results[1] / len(train_loader), epoch)
                    writer.add_scalar("Condition/2", conditions_results[1] / len(train_loader), epoch)
                if data_type == "multitask":
                    writer.add_scalar("Accuracy/ops", ops_acc / len(train_loader), epoch)
                for name, param in model.named_parameters():
                    if name.find("alpha") != -1:
                        writer.add_scalar(f"Param/alpha", param.item(), epoch)
                    elif name.find("beta") != -1:
                        writer.add_scalar(f"Param/beta", param.item(), epoch)
                if learning_method == "GAN":
                    writer.add_scalar("Loss/discriminator", total_d_loss / len(train_loader), epoch)
                    writer.add_scalar("Loss/generator", total_g_loss / len(train_loader), epoch)
                    for name, param in discriminator.named_parameters():
                        if param.grad is not None:
                            writer.add_histogram(f'Grad/{name}', param.grad, epoch)
                if epoch % 10 == 0:
                    for name, param in model.named_parameters():
                        if name.find("bias") == -1 and param.grad is not None:
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
                    if data_type == "multitask":
                        tgt = tgt[0]
                    tgt = tgt.cuda()
                    if data_type == "multitask":
                        out, _, _ = model(src)
                    elif model_name == "bracket":
                        out, _ = model(src)
                    else:
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
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--n_interval", type=int, default=1000)
    parser.add_argument("--n_epochs", type=int, default=100000)
    parser.add_argument("--n_data", type=int, default=512000)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--debug_print", action="store_true")
    parser.add_argument("--lie_mode", type=str, default="base")
    parser.add_argument("--lie_map_mode", type=str, default="map_first")
    args = parser.parse_args()

    main(d_model=args.d_model, n_layers=args.n_layers,
         data_type=args.data_type, learning_method=args.learning_method,
         discriminator_lr=args.discriminator_lr, dropout=args.dropout,
         n_interval=args.n_interval,
         n_epochs=args.n_epochs, n_data=args.n_data, batch_size=args.batch_size,
         model_name=args.model, lie_mode=args.lie_mode, lie_map_mode=args.lie_map_mode,
         debug_print=args.debug_print)