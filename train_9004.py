# program1_scnn_train.py

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import ConcatDataset, DataLoader


from spikingjelly.activation_based import neuron, functional, layer



LEARNING_RATE = 1e-3
# 4096 89.24
# 2048 89.55
# 50 1024
BATCH_SIZE = 2048
EPOCHS = 1000
T_TIMESTEPS = 2


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)


setup_seed(42)


def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)


class SingleDeviceMuonWithAuxAdam(torch.optim.Optimizer):
    """
    Non-distributed variant of MuonWithAuxAdam.
    """
    def __init__(self, param_groups):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "use_muon"])
            else:
                # defaults
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muon_update(p.grad, state["momentum_buffer"], beta=group["momentum"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update.reshape(p.shape), alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        # continue
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        return loss


# input size: 10,0 x 1 x 28x28

class SCNN(nn.Module):
    def __init__(self, T: int):
        super(SCNN, self).__init__()
        self.T = T
        # 1x28x28
        self.conv1 = layer.Conv2d(1, 8, 5) # in_channels: 1, out_channels: 6, kernel_size: 5
        # 6x24x24
        self.if1 = neuron.IFNode()
        self.pool1 = layer.MaxPool2d(2, 2)
        # 6x12x12

        self.conv2 = layer.Conv2d(8, 16, 5) # in_channels: 6, out_channels: 16, kernel_size: 5
        # 16x8x8
        self.if2 = neuron.IFNode()
        self.pool2 = layer.MaxPool2d(2, 2)
        # 16x4x4

        self.flatten = layer.Flatten()
        
        self.fc1 = layer.Linear(16 * 4 * 4, 128)
        # 120
        self.if3 = neuron.IFNode()

        self.fc2 = layer.Linear(128, 96)
        # 84
        self.if4 = neuron.IFNode()

        self.fc3 = layer.Linear(96, 10)
        # 10


    def forward(self, x: torch.Tensor):
        outputs = []
        for t in range(self.T):
            y = self.conv1(x)
            y = self.if1(y)
            y = self.pool1(y)
            y = self.conv2(y)
            y = self.if2(y)
            y = self.pool2(y)
            y = self.flatten(y)
            y = self.fc1(y)
            y = self.if3(y)
            y = self.fc2(y)
            y = self.if4(y)
            y = self.fc3(y)
            outputs.append(y)
        
        outputs = torch.stack(outputs, dim=0)
        return outputs.mean(0)

script_dir = os.path.dirname(os.path.abspath(__file__))


train_transform = transforms.Compose([
   transforms.RandomHorizontalFlip(),
   # transforms.RandomVerticalFlip(),
    # transforms.RandomRotation(45)(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


data_dir = os.path.join(script_dir, 'data')
os.makedirs(data_dir, exist_ok=True)
trainset = torchvision.datasets.FashionMNIST(data_dir, download=True, train=True, transform=train_transform)
testset = torchvision.datasets.FashionMNIST(data_dir, download=True, train=False, transform=test_transform)


# Assume you have dataset1 and dataset2
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = SCNN(T=T_TIMESTEPS).to(device)
# Convert model parameters to FP16 on CUDA
use_fp16 = 0 #  (device == 'cuda')
if use_fp16:
    model.half()

# optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# optimizer = Muon(model.parameters(), lr=0.24, momentum=0.6, nesterov=True)

# 筛选出所有维度 >= 2 的参数，通常是权重 (weights)
muon_params = [
    p for name, p in model.named_parameters() 
    if p.ndim >= 2 and 'embed' not in name
]

# 剩余的所有参数都交给 AdamW
# 包括：所有维度 < 2 的参数 (biases, layernorm gains) 以及 Embedding 层的参数
adam_params = [
    p for name, p in model.named_parameters() 
    if p.ndim < 2 or 'embed' in name
]

# 检查是否所有参数都被分配了
assert len(list(model.parameters())) == len(muon_params) + len(adam_params)


# hidden_weights = [p for p in model.body.parameters() if p.ndim >= 2]
# hidden_gains_biases = [p for p in model.body.parameters() if p.ndim < 2]
# nonhidden_params = [*model.head.parameters(), *model.embed.parameters()]
param_groups = [
    dict(params=muon_params, use_muon=True,
         lr=0.02, weight_decay=0.01),
    dict(params=adam_params, use_muon=False,
         lr=1e-2, betas=(0.9, 0.95), weight_decay=0.01),
]
optimizer = SingleDeviceMuonWithAuxAdam(param_groups)   


scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

criterion = nn.CrossEntropyLoss()


print("--- Starting SCNN Training (Tuned for Convergence) ---")
max_accuracy = 0.0
for epoch in range(EPOCHS):

    model.train()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        functional.reset_net(model)

        if use_fp16:
            inputs = inputs.half()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch [{epoch + 1}/{EPOCHS}], Loss: {running_loss / len(trainloader):.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')
    scheduler.step()


    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)

            functional.reset_net(model)
            if use_fp16:
                images = images.half()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f} %')
    

    if accuracy > max_accuracy:
        max_accuracy = accuracy
        print(f'New best accuracy: {max_accuracy:.2f} %. Saving model parameters...')
        output_dir = os.path.join(script_dir)
        os.makedirs(output_dir, exist_ok=True)
        for name, param in model.named_parameters():
            np.savetxt(os.path.join(output_dir, f'{name}.txt'), param.detach().cpu().numpy().flatten())


print('--- Finished Training ---')
print(f'Best accuracy achieved: {max_accuracy:.2f} %')
print("--- Final model parameters have been exported. ---")
