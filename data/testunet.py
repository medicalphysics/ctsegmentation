from math import inf
import os
import numpy as np
import torch 
import torch.distributed
from torch import nn, autocast
from matplotlib import pylab as plt
from typing import Any, Optional, Tuple, Callable
from tqdm import tqdm
from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualUNet

from totseg_dataloader import TotSegDataset2D

torch.set_num_threads(os.cpu_count()//2)

class AllGatherGrad(torch.autograd.Function):
    # stolen from pytorch lightning
    @staticmethod
    def forward(
        ctx: Any,
        tensor: torch.Tensor,
        group: Optional["torch.distributed.ProcessGroup"] = None,
    ) -> torch.Tensor:
        ctx.group = group

        gathered_tensor = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]

        torch.distributed.all_gather(gathered_tensor, tensor, group=group)
        gathered_tensor = torch.stack(gathered_tensor, dim=0)

        return gathered_tensor

    @staticmethod
    def backward(ctx: Any, *grad_output: torch.Tensor) -> Tuple[torch.Tensor, None]:
        grad_output = torch.cat(grad_output)

        torch.distributed.all_reduce(grad_output, op=torch.distributed.ReduceOp.SUM, async_op=False, group=ctx.group)

        return grad_output[torch.distributed.get_rank()], None

class MemoryEfficientSoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin: Callable = None, batch_dice: bool = False, do_bg: bool = True, smooth: float = 1.,
                 ddp: bool = True):
        """
        saves 1.6 GB on Dataset017 3d_lowres
        """
        super(MemoryEfficientSoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth
        self.ddp = ddp

    def forward(self, x, y, loss_mask=None):
        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        # make everything shape (b, c)
        axes = tuple(range(2, x.ndim))

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

            if x.shape == y.shape:
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                y_onehot = torch.zeros(x.shape, device=x.device, dtype=torch.bool)
                y_onehot.scatter_(1, y.long(), 1)

            if not self.do_bg:
                y_onehot = y_onehot[:, 1:]

            sum_gt = y_onehot.sum(axes) if loss_mask is None else (y_onehot * loss_mask).sum(axes)

        # this one MUST be outside the with torch.no_grad(): context. Otherwise no gradients for you
        if not self.do_bg:
            x = x[:, 1:]

        if loss_mask is None:
            intersect = (x * y_onehot).sum(axes)
            sum_pred = x.sum(axes)
        else:
            intersect = (x * y_onehot * loss_mask).sum(axes)
            sum_pred = (x * loss_mask).sum(axes)

        if self.batch_dice:
            if self.ddp:
                intersect = AllGatherGrad.apply(intersect).sum(0)
                sum_pred = AllGatherGrad.apply(sum_pred).sum(0)
                sum_gt = AllGatherGrad.apply(sum_gt).sum(0)

            intersect = intersect.sum(0)
            sum_pred = sum_pred.sum(0)
            sum_gt = sum_gt.sum(0)

        dc = (2 * intersect + self.smooth) / (torch.clip(sum_gt + sum_pred + self.smooth, 1e-8))

        dc = dc.mean()
        return -dc




class DiceSignLoss(nn.Module):
    """On the dice loss gradient and the ways to mimic it:Hoel Kervadec, Marleen De Bruijne"""
    def __init__(self):
        super(DiceSignLoss, self).__init__()        
    
    def forward(self, x, y):
        x.mul_(y)        
        return -x.sum()

    def diceScore(self, x, y):
        xbin = x.ge(0.5)        
        nom = xbin*y
        return 2*nom.sum()/(xbin.sum()+y.sum())

def train_one_epoch(model, loss, optimizer, data, device, shuffle=True):
    ## see https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
    ## for loss https://arxiv.org/pdf/1707.03237v3
    ## eller CrossEntropyLoss eller NLLLoss med LogSoftMax lag

    total_loss = 0
    model.train(True)  
    if shuffle:
        data.shuffle()

    pbar = tqdm(total=len(data), leave=True)    
    for ind, (image, label) in enumerate(data):
        image = image.to(device, non_blocking = True)
        label = label.to(device, non_blocking = True)
    
        optimizer.zero_grad(set_to_none=True)

        output = model(image)

        l = loss(output, label)        
       
        l.backward()
        optimizer.step()
        total_loss += l.item()       
        pbar.update(1)            
        """
        with torch.no_grad():            
            plt.subplot(1, 3, 1)
            plt.imshow(image[0,0,:,:,image.shape[4]//2], cmap='bone', vmin=0,vmax=1)
            plt.subplot(1, 3, 2)
            plt.imshow(label[0,0,:,:,image.shape[4]//2], vmin=0,vmax=1)
            plt.subplot(1, 3, 3)
            plt.imshow(output[0,0,:,:,image.shape[4]//2], vmin=0,vmax=1)
            plt.show()
        """
    pbar.close()    
    return total_loss/len(data)

def validate(model, data, loss, device):
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()
    running_vloss=0
    running_dice = 0
    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        pbar = tqdm(total=len(data), leave=True)    
        for image, labels in data: 
            image = image.to(device, non_blocking = True)
            labels = labels.to(device, non_blocking = True)        
            out = model(image)
            vloss = loss(out, labels)
            running_vloss += vloss.item()
            running_dice += loss.diceScore(out, labels).item()
            pbar.update(1)                  
        pbar.close()

    avg_loss = running_vloss / len(data)
    avg_dice = running_dice / len(data)
    print('LOSS valid {}, DICE: {}'.format(avg_loss, avg_dice))
    return avg_loss, avg_dice

def plot_loss(train, valid, dice):
    bt, t = zip(*train)
    bv, v = zip(*valid)
    bd, d = zip(*dice)
    plt.subplot(2, 1, 1)
    plt.plot(bt, t, label='Train loss')
    plt.plot(bv, v, label='Validation loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend()    
    plt.subplot(2, 1, 2)
    plt.plot(bd, d, label='Dice coeff')
    plt.xlabel('Batch')
    plt.ylabel('Dice coeff')
    plt.legend()    
    plt.savefig("train.png", dpi=600)


def start_train(n_epochs = 15, device = 'cpu', batch_size=4, load_model=True):
    #model = ResidualUNet(1, 4, (32, 64, 125, 256,), nn.Conv3d, 3, (1, 2, 2, 2, ), (2, 2, 2, 2, ), 1,
    #                            (2, 2, 2, ), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=False).to(device)
    #model = PlainConvUNet(1, 6, (32, 64, 125, 256, 320, 320), nn.Conv3d, 3, (1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2), 1,
    #                         (2, 2, 2, 2, 2), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=False).to(device)

    
    dataset = TotSegDataset2D(r"D:\totseg\Totalsegmentator_dataset_v201", train=True, batch_size=batch_size)
    dataset_val = TotSegDataset2D(r"D:\totseg\Totalsegmentator_dataset_v201", train=False, batch_size=batch_size)


    model = PlainConvUNet(1, 5, (32, 64, 125, 256, 320), nn.Conv2d, 3, (1, 2, 2, 2, 2), (2, 2, 2, 2, 2), dataset._label_tensor_dim,
                                (2, 2, 2, 2), conv_bias=False, norm_op=nn.BatchNorm2d, nonlin=nn.ReLU, deep_supervision=False).to(device)
    
    if load_model:
        if os.path.exists("model.pt"):
            model.load_state_dict(torch.load("model.pt"))        

    #learning_rate = 1e-2
    #learning_decay = 1e-5
    #optimizer = torch.optim.SGD(model.parameters(), learning_rate, weight_decay=learning_decay,
    #                                momentum=0.99, nesterov=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr = 5e-4, betas=(0.99,0.999))


    #loss = InlineDiceLoss(dataset.max_labels).to(device)
    #loss = torch.nn.CrossEntropyLoss(weight=None, reduction='mean', label_smoothing=0.0)
    #loss = MemoryEfficientSoftDiceLoss()
    #loss = torch.nn.MSELoss(reduction='mean').to(device)
    loss = DiceSignLoss()

   
    epoch_loss = list()
    val_loss = list()
    dice_coeff = list()
    epoch_current_loss = 1E6

    for ind in range(n_epochs):
        mean_loss = train_one_epoch(model, loss, optimizer, dataset, device)        
        epoch_loss.append((ind+1, mean_loss))
        if mean_loss < epoch_current_loss:            
            torch.save(model.state_dict(), "model.pt")
            #example = torch.rand((batch_size, 1, 128, 128, 64), dtype=torch.float32)
            #traced_script_module = torch.jit.trace(model, example)
            #traced_script_module.save("traced_model_{}.pt".format(batch_size))
            epoch_current_loss = mean_loss
        
        val_loss_current, val_dice_current = validate(model, dataset_val, loss, device)
        val_loss.append((ind+1, val_loss_current))
        dice_coeff.append((ind+1, val_dice_current))
        if ind > 0:
            plot_loss(epoch_loss, val_loss, dice_coeff)

def predict(data):
    model = PlainConvUNet(1, 5, (32, 64, 125, 256, 320), nn.Conv2d, 3, (1, 2, 2, 2, 2), (2, 2, 2, 2, 2), data._label_tensor_dim,
                                (2, 2, 2, 2), conv_bias=False, norm_op=nn.BatchNorm2d, nonlin=nn.ReLU, deep_supervision=False).to(device)
    model.load_state_dict(torch.load("model.pt"))
    model.eval()
    with torch.no_grad():        
        for image, label in data:                
            out = model(image).detach().cpu().numpy()
            plt.subplot(1, 3, 1)
            plt.imshow(image[0,0,:,:,image.shape[4]//2], cmap='bone')
            plt.subplot(1, 3, 2)
            plt.imshow(label[0,0,:,:,image.shape[4]//2]*117)
            plt.subplot(1, 3, 3)
            
            plt.show()


        

if __name__=='__main__':
    start_train(n_epochs = 1, device='cuda', batch_size=32, load_model=True)
    #start_train(n_epochs = 1, device='cpu', batch_size=32, load_model=False)

   

"""

   data = torch.rand((4, 1, 128, 128, 128))

    punet = PlainConvUNet(1, 6, (32, 64, 125, 256, 320, 320), nn.Conv3d, 3, (1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2), 1,
                             (2, 2, 2, 2, 2), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=True)
    print(punet.compute_conv_feature_map_size(data.shape[2:]))
   
    runet = ResidualUNet(1, 6, (32, 64, 125, 256, 320, 320), nn.Conv3d, 3, (1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2), 1,
                                (2, 2, 2, 2, 2), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=True)
    print(runet.compute_conv_feature_map_size(data.shape[2:]))


    runet = ResidualUNet(1, 4, (32, 64, 125, 256,), nn.Conv3d, 3, (1, 2, 2, 2, ), (2, 2, 2, 2, ), 1,
                                (2, 2, 2, ), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=False)
    print(runet.compute_conv_feature_map_size(data.shape[2:]))
    

    punet.eval()
    with torch.no_grad():
        ans = punet.forward(data)


    plt.imshow(ans[0][0,0,:,:,63])

    plt.show()
    ## test trace
    #traced = torch.jit.trace(runet, data)
   # print(traced)
    #traced.save(r"c:/Users/ander/source/ctsegmentation/data/models/test.pt")

    #print(runet)

    #g = hl.build_graph(runet, data, transforms=None)
    #g.save("network_architecture.pdf")
    
"""