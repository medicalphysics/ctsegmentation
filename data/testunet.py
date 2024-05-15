import os
import torch 
import torch.distributed
from torch import nn, autocast
from matplotlib import pylab as plt
from typing import Any, Optional, Tuple, Callable
from tqdm import tqdm
from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualUNet

from totseg_dataloader import TotSegDataset

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




class InlineDiceLoss(nn.Module):
    def __init__(self, n_classes: int):
        super(InlineDiceLoss, self).__init__()

        self.class_delta_sqr = (1 / n_classes)**2

    def forward(self, x, y):
        d = x.subtract(y)
        d.square_()
        d.le_(self.class_delta_sqr)
        return - d.mean()


class dummy_context(object):
    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

def train_one_epoch(model, loss, optimizer, data, device):
## see https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
## for loss https://arxiv.org/pdf/1707.03237v3
## eller CrossEntropyLoss eller NLLLoss med LogSoftMax lag
    model.train(True)
    
    #pbar = tqdm(total=len(data), leave=False)
    #for image, label in tqdm(data.iter_batch()):
    for image, label in data.iter_batch():
        image = image.to(device, non_blocking = True)
        label = label.to(device, non_blocking = False)
    
        optimizer.zero_grad(set_to_none=True)

        output = model(image)
        l = loss(output, label)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        #with autocast(device, enabled=True) if device == 'cuda' else dummy_context():
        #    output = model(data)       
        #    l = loss(output, target)

        l.backward()
        optimizer.step()
        print(l)
        #pbar.update(1)
    #pbar.close()

    torch.save(model.state_dict(), "model.pt")

    return l
        


def start_train(device = 'cpu'):
    #model = ResidualUNet(1, 4, (32, 64, 125, 256,), nn.Conv3d, 3, (1, 2, 2, 2, ), (2, 2, 2, 2, ), 1,
    #                            (2, 2, 2, ), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=False).to(device)
    #model = PlainConvUNet(1, 6, (32, 64, 125, 256, 320, 320), nn.Conv3d, 3, (1, 2, 2, 2, 2, 2), (2, 2, 2, 2, 2, 2), 1,
    #                         (2, 2, 2, 2, 2), False, nn.BatchNorm3d, None, None, None, nn.ReLU, deep_supervision=False).to(device)
    model = PlainConvUNet(1, 5, (32, 64, 125, 256, 320), nn.Conv3d, 3, (1, 2, 2, 2, 2), (2, 2, 2, 2, 2), 1,
                             (2, 2, 2, 2), False, nn.BatchNorm3d, None, None, nn.ReLU, deep_supervision=False).to(device)
    learning_rate = 1e-2
    learning_decay = 1e-5
    optimizer = torch.optim.SGD(model.parameters(), learning_rate, weight_decay=learning_decay,
                                    momentum=0.99, nesterov=True)
    dataset = TotSegDataset(r"D:\totseg\Totalsegmentator_dataset_v201", max_labels = 117, batch_size=4)
    loss = InlineDiceLoss(dataset.max_labels).to(device)
    train_one_epoch(model, loss, optimizer, dataset, device)


if __name__=='__main__':
    start_train('cuda')
    #start_train('cpu')

"""    data = torch.rand((4, 1, 128, 128, 128))

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