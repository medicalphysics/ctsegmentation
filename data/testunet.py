from math import inf
import os
import numpy as np
import torch 
import torch.distributed
from torch import nn, autocast
from matplotlib import pylab as plt
from typing import Any, Optional, Tuple, Callable
from tqdm import tqdm
from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from totseg_dataloader import TotSegDataset2D
from loss import DC_and_CE_loss


#torch.set_num_threads(os.cpu_count()//2)

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
        #wy = -2*y+0.1        
        #return (x*wy).sum()
        x.mul_(y)        
        return -x.sum()

    def diceScore(self, x, y):
        xbin = x.ge(0.5)        
        nom = xbin*y
        return 2*nom.sum()/(xbin.sum()+y.sum())

def train_one_epoch(model, loss, optimizer, sheduler, data, device, shuffle=True):
    ## see https://pytorch.org/tutorials/beginner/introyt/trainingyt.html    

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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
        optimizer.step()
        sheduler.step()
        total_loss += l.item()       
        pbar.update(1)       
    pbar.close()    
    return total_loss/len(data)


def validate(model, data, loss, device):
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()
    running_vloss=0
    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        pbar = tqdm(total=len(data), leave=True)    
        for image, labels in data: 
            image = image.to(device, non_blocking = True)
            labels = labels.to(device, non_blocking = True)                 
            out = model(image)
            vloss = loss(out, labels)
            running_vloss += vloss.item()            
            pbar.update(1)                  
        pbar.close()

    avg_loss = running_vloss / len(data)    
    print('LOSS valid {}'.format(avg_loss))
    return avg_loss

def plot_loss(train, valid):
    bt, t = zip(*train)
    bv, v = zip(*valid)    
    plt.subplot(1, 1, 1)
    plt.plot(bt, t, label='Train loss')
    plt.plot(bv, v, label='Validation loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.legend()        
    plt.tight_layout()    
    plt.savefig("train.png", dpi=600)
    plt.clf()

def get_model(N):
    #model = PlainConvUNet(1, 5, (32, 64, 125, 256, 320), nn.Conv2d, 3, (1, 2, 2, 2, 2), (2, 2, 2, 2, 2), N,
    #                            (2, 2, 2, 2), conv_bias=False, norm_op=nn.BatchNorm2d, nonlin=nn.Softmax, nonlin_kwargs={'dim':1} ,deep_supervision=False)
    #return model
    #patch size is 256x256
    model = ResidualEncoderUNet(1, #input channels
                                7, #n_stages
                                (32,64,128,256,512,512,512), # features per stage
                                torch.nn.modules.conv.Conv2d, # conv_op
                                3, # kernel sizes
                                (1, 2, 2, 2, 2, 2, 2), # strides
                                (1, 3, 4, 6, 6, 6, 6), # n_blocks per stage
                                N, # num classes 
                                (1,1,1,1,1,1), # n_conv_per_stage_decoder 
                                conv_bias=True, norm_op=torch.nn.modules.instancenorm.InstanceNorm2d, norm_op_kwargs={"eps":1e-5, 'affine':True}, nonlin=torch.nn.LeakyReLU, nonlin_kwargs= {"inplace":True} )
    return model

def start_train(n_epochs = 15, device = 'cpu', batch_size=4, load_model=True, data_path = None):
   
    
    dataset = TotSegDataset2D(data_path, train=True, batch_size=batch_size, dtype = torch.float32)
    dataset_val = TotSegDataset2D(data_path, train=False, batch_size=batch_size, dtype = torch.float32)


    model = get_model(dataset._label_tensor_dim).to(device)
    
   
    initial_lr = 1e-2
    weight_decay = 3e-5
    optimizer = torch.optim.SGD(model.network.parameters(), initial_lr, weight_decay=weight_decay,
                                momentum=0.99, nesterov=True)
    sheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=5,)

    #loss = InlineDiceLoss(dataset.max_labels).to(device)
    lossweights = torch.zeros(dataset._label_tensor_dim, dtype=torch.float32)
    for i in range(1, dataset.max_labels+1):
        lossweights[i] = 1
    lossweights=lossweights.to(device)
    loss = torch.nn.CrossEntropyLoss(weight=lossweights, reduction='mean', label_smoothing=0.0)
    #loss  = torch.nn.NLLLoss(weight=lossweights, reduction='mean') #sammen med logsoftmax som nonlin lag
    #loss = MemoryEfficientSoftDiceLoss()
    #loss = torch.nn.MSELoss(reduction='mean').to(device)
    #loss = DiceSignLoss()

    loss = DC_and_CE_loss({'batch_dice': True,
                           'smooth': 1e-5, 'do_bg': False, 'ddp': self.is_ddp}, 
                           {}, 
                           weight_ce=1, 
                           weight_dice=1,
                           ignore_label=self.label_manager.ignore_label, dice_class=MemoryEfficientSoftDiceLoss)

   
    epoch_loss = list()
    val_loss = list()   
    epoch_current_loss = 1E6

    if load_model:
        if os.path.exists("model.pt"):
            state = torch.load("model.pt")            
            model.load_state_dict(state['state_dict'])
            optimizer.load_state_dict(state['optimizer'])       
            sheduler.load_state_dict(state['sheduler'])
    for ind in range(n_epochs):
        mean_loss = train_one_epoch(model, loss, optimizer, sheduler, dataset, device, shuffle=False)        
        epoch_loss.append((ind+1, mean_loss))        
        if mean_loss < epoch_current_loss: 
            state = {
                'epoch': ind,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'sheduler':sheduler.state_dict(),                
                }
            torch.save(state, "model.pt")           
            #torch.save(model.state_dict(), "model.pt")
            #example = torch.rand((batch_size, 1, 128, 128, 64), dtype=torch.float32)
            #traced_script_module = torch.jit.trace(model, example)
            #traced_script_module.save("traced_model_{}.pt".format(batch_size))
            epoch_current_loss = mean_loss
        
        val_loss_current = validate(model, dataset_val, loss, device)
        val_loss.append((ind+1, val_loss_current))        
        if ind > 0:
            plot_loss(epoch_loss, val_loss)

def predict(data):
    model = get_model(data._label_tensor_dim)
    state = torch.load("model.pt")            
    model.load_state_dict(state['state_dict'])
    model.eval()
    with torch.no_grad():        
        for image, label in data:  
            label_index = label.mul(torch.arange(label.shape[1]).reshape((label.shape[1], 1, 1))).sum(dim=1, keepdim=True)                                 
            out = model(image)
            #out = out.ge(0.5)
            out_index = out.mul(torch.arange(label.shape[1]).reshape((label.shape[1], 1, 1))).sum(dim=1, keepdim=True) 
            plt.subplot(2, 2, 1)
            plt.imshow(image[0,0,:,:])
            plt.subplot(2, 2, 2)
            plt.imshow(label_index[0,0,:,:])
            plt.subplot(2, 2, 3)
            plt.imshow(out[0,21,:,:])
            plt.subplot(2, 2, 4)
            plt.imshow(out_index[0,0,:,:])
            plt.show()
            

        

if __name__=='__main__':
    #dataset_path = r"C:\Users\ander\totseg"
    dataset_path = r"D:\totseg\Totalsegmentator_dataset_v201"
    batch_size=16
    #start_train(n_epochs = 15, device='cuda', batch_size=batch_size, load_model=True, data_path = dataset_path)
    #start_train(n_epochs = 15, device='cpu', batch_size=batch_size, load_model=True, data_path = dataset_path)

    if True:
        dataset = TotSegDataset2D(dataset_path, train=True, batch_size=batch_size)
        predict(dataset)



    if False:
        d = TotSegDataset2D(dataset_path, train=True, batch_size=16)
        image, label = d[40]
        print(image.shape)
        print(label.shape)
        label_exp=label.mul(torch.arange(label.shape[1]).reshape((label.shape[1], 1, 1))).sum(dim=1, keepdim=True)
        print(label_exp.shape)
        plt.subplot(1, 3, 1)
        plt.imshow(image[0, 0,:,:])
        plt.subplot(1, 3, 2)
        plt.imshow(label[0,0,:,:])
        plt.subplot(1, 3, 3)
        plt.imshow(label_exp[0,0,:,:])
        plt.show()
        





   

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
    
"""