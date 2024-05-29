from math import inf
import os
import numpy as np
import torch 
import torch.distributed
from torch import nn, autocast
from typing import Any, Optional, Tuple, Callable
from tqdm import tqdm
from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from totseg_dataloader import TotSegDataset2D
from loss import DC_and_CE_loss, MemoryEfficientSoftDiceLoss, softmax_helper_dim1
from matplotlib import pylab as plt

torch.set_num_threads(os.cpu_count())


def train_one_epoch(model, loss, optimizer, data, device, shuffle=True, n_iter=None):
    ## see https://pytorch.org/tutorials/beginner/introyt/trainingyt.html     
    model.train(True)  
    stop_ind = min(len(data),n_iter)    
    pbar = tqdm(total=stop_ind, leave=False)        
    for ind, (image, label) in enumerate(data.iter_ever(shuffle)):
        image = image.to(device, non_blocking = True)
        label = label.to(device, non_blocking = True)
        optimizer.zero_grad(set_to_none=True)
        output = model(image)        
        l = loss(output, label)                    
        l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
        optimizer.step()                
        pbar.update(1)
        if ind >= stop_ind:
            break
    pbar.close()    
    return 


def validate(model, data, loss, device):
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    model.eval()
    running_vloss=0
    running_dice =0
    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        pbar = tqdm(total=len(data), leave=False)    
        for image, labels in data: 
            image = image.to(device, non_blocking = True)
            labels = labels.to(device, non_blocking = True)                 
            out = model(image)
            vloss = loss(out, labels)
            vdice = loss.diceCoeff(out, labels)
            running_dice += float(vdice.cpu())
            running_vloss += float(vloss.cpu())            
            pbar.update(1)
        pbar.close()

    return running_vloss / len(data), running_dice/len(data)
    

def plot_loss(loss, lr, dice): 
    for ind, (v, l) in enumerate(zip([loss, lr, dice], ['Validation Loss', 'Learning rate', 'Dice Score'])):
        plt.subplot(1, 3, ind+1)
        plt.plot(v)
        plt.xlabel('Batch nr')
        plt.ylabel(l)
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

def start_train(n_epochs = 150, device = 'cpu', batch_size=4, load_model=True, load_only_model=False, data_path = None):   
    volumes = list([10,11,12,13,14])
    
    dataset_val = TotSegDataset2D(data_path, train=False, batch_size=batch_size, volumes=volumes, dtype = torch.float32)
    #dataset = TotSegDataset2D(data_path, train=True, batch_size=batch_size, volumes=volumes, dtype = torch.float32)    
    dataset = dataset_val

    model = get_model(dataset._label_tensor_dim).to(device)
    
    initial_lr = 0.04
    weight_decay = 3e-5
    optimizer = torch.optim.SGD(model.parameters(), initial_lr, weight_decay=weight_decay,
                                momentum=0.99, nesterov=True)
    sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, cooldown=0, factor=0.2, threshold=0.01)
    loss = DC_and_CE_loss({'batch_dice': False, 'smooth': 1e-5, 'do_bg': False}, {'label_smoothing':1e-5}, weight_ce=1, weight_dice=1,
                          ignore_label=None).to(device)

   
    validation_loss = list()
    dice_score = list()   
    lr_rate = list()
    min_val_loss = 1e9

    if load_model:
        if os.path.exists("model.pt"):
            state = torch.load("model.pt")            
            model.load_state_dict(state['model'])
            if not load_only_model:
                optimizer.load_state_dict(state['optimizer'])       
                sheduler.load_state_dict(state['sheduler'])
            validation_loss = state['validation_loss']
            dice_score = state['dice_score']
            lr_rate = state['lr_rate']
            min_val_loss = min(validation_loss)
    
    for ind in range(n_epochs):
        train_one_epoch(model, loss, optimizer, dataset, device, shuffle=True, n_iter=1000)                        
        lossv, dicev = validate(model, dataset_val, loss, device)
        sheduler.step(lossv)
        lr_rate.append(sheduler.get_last_lr()[0])   
        validation_loss.append(lossv)        
        dice_score.append(dicev)
        if min_val_loss > lossv:
            min_val_loss = lossv
            state = {
                'epoch': ind,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'sheduler':sheduler.state_dict(), 
                'validation_loss':validation_loss,
                'dice_score':dice_score,
                'lr_rate': lr_rate          
                }
            torch.save(state, "model.pt")                                  
            #traced_script_module = torch.jit.trace(model.to('cpu'), torch.rand(dataset.batch_shape(), dtype=torch.float32))
            #traced_script_module.save("traced_model.pt")
        plot_loss(validation_loss, lr_rate, dice_score)

def predict(data):
    model = get_model(data._label_tensor_dim)
    state = torch.load("model.pt")            
    model.load_state_dict(state['model'])
    model.eval()
    with torch.no_grad():        
        for image, label in data:  
            #label_index = label.mul(torch.arange(label.shape[1]).reshape((label.shape[1], 1, 1))).sum(dim=1, keepdim=True)                                 
            out_r = model(image)
            out = softmax_helper_dim1(out_r)
            #out = out.ge(0.5)
            out_index = out.mul(torch.arange(out.shape[1]).reshape((out.shape[1], 1, 1))).sum(dim=1, keepdim=True) 
            plt.subplot(2, 2, 1)
            plt.imshow(image[0,0,:,:])
            plt.subplot(2, 2, 2)
            plt.imshow(label[0,0,:,:])
            plt.subplot(2, 2, 3)           
            plt.subplot(2, 2, 4)
            plt.imshow(out[0,1,:,:])
            plt.show()
            


if __name__=='__main__':
    dataset_path = r"C:\Users\ander\totseg"
    dataset_path = r"D:\totseg\Totalsegmentator_dataset_v201"
    

    batch_size=30
    start_train(n_epochs = 150, device='cuda', batch_size=batch_size, load_model=True, data_path = dataset_path)
    #start_train(n_epochs = 3, device='cpu', batch_size=batch_size, load_model=True, data_path = dataset_path)

    if False:
        volumes = list([10,11,12,13,14])
        dataset=TotSegDataset2D(dataset_path, train=False, batch_size=batch_size, volumes=volumes, dtype = torch.float32)
        predict(dataset)



    if False:
        d = TotSegDataset2D(dataset_path, train=True, batch_size=16)
        image, label = d[8]
        model = get_model(d._label_tensor_dim)
        state = torch.load("model.pt")            
        model.load_state_dict(state['model'])
        model.eval()
        with torch.no_grad(): 
            #out = torch.sigmoid(model(image)).detach()
            out = model(image).detach()

        print(image.shape, label.shape, out.shape)
        
        #label_exp=label.mul(torch.arange(label.shape[1]).reshape((label.shape[1], 1, 1))).sum(dim=1, keepdim=True)
        for i in range(image.shape[0]):
            plt.subplot(1, 3, 1)
            plt.imshow(image[0, 0,:,:])
            plt.subplot(1, 3, 2)
            plt.imshow(label[0,0,:,:])
            plt.subplot(1, 3, 3)
            plt.imshow(out[0,80,:,:])
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