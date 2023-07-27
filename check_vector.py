import torch
from grokking import SimpleFormer, gen_train_test

frac_train = 0.5
p = 113
seed = 0

# runs/Jul08_12-51-19_ae288e22a540
# num_encoder_layer=1, num_decoder_layer=6
model = SimpleFormer(p, nhead=8, num_encoder_layers=1, num_decoder_layers=3, dropout=0.00)
results = torch.load("save/Jul10_15-10-39_128_8_1_3/results_19999.pth")
print(results.keys())
model.load_state_dict(results["model"])

model.eval()

train, test = gen_train_test(frac_train, p, seed)
train_result = [[p, (i+j)%p] for i, j, _ in train]
test_result =  [[p, (i+j)%p] for i, j, _ in test]

def printnorm(self, input, output):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    print('Inside ' + self.__class__.__name__ + ' forward')
    print('')
    print('input: ', type(input), len(input))
    print('input[0]: ', type(input[0]))
    print('output: ', type(output), len(output))
    print('')
    for i in range(len(input)):
        print(f'input[{i}] size:', input[i].size())
    for i in range(len(output)):
        print(f'output[{i}] size:', output[i].size())
    # print('output norm:', output.data.norm())
    print("--")

# net.features[0].register_forward_hook(printnorm)
for name, module in model.named_modules():
    print(name)

model.transformer.encoder.register_forward_hook(printnorm)
model.transformer.decoder.register_forward_hook(printnorm)
# model.transformer.decoder.layers[0].self_attn.register_forward_hook(printnorm)
model.transformer.decoder.layers[0].multihead_attn.register_forward_hook(printnorm)
# model.transformer.decoder.layers[1].self_attn.register_forward_hook(printnorm)
# model.transformer.decoder.layers[1].multihead_attn.register_forward_hook(printnorm)
# model.transformer.decoder.layers[2].self_attn.register_forward_hook(printnorm)
# model.transformer.decoder.layers[2].multihead_attn.register_forward_hook(printnorm)

src = torch.tensor(test[0]).view(1, -1).t()
tgt = torch.tensor(test_result[0]).view(1, -1).t()
print(src.shape, tgt.shape)
model(src, tgt)
