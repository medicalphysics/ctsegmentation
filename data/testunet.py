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
from loss import DC_and_CE_loss, MemoryEfficientSoftDiceLoss, softmax_helper_dim1


#torch.set_num_threads(os.cpu_count()//2)


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
    volumes = list([10,11,12,13,14])
    
    dataset_val = TotSegDataset2D(data_path, train=False, batch_size=batch_size, volumes=volumes, dtype = torch.float32)
    dataset = TotSegDataset2D(data_path, train=True, batch_size=batch_size, volumes=volumes, dtype = torch.float32)    

    model = get_model(dataset._label_tensor_dim).to(device)
    
    initial_lr = 1e-4
    weight_decay = 3e-5
    optimizer = torch.optim.SGD(model.parameters(), initial_lr, weight_decay=weight_decay,
                                momentum=0.99, nesterov=True)
    sheduler = torch.optim.lr_scheduler.PolynomialLR(optimizer, total_iters=n_epochs)

    #loss = InlineDiceLoss(dataset.max_labels).to(device)
    #lossweights = torch.zeros(dataset._label_tensor_dim, dtype=torch.float32)
    #for i in range(1, dataset.max_labels+1):
    #    lossweights[i] = 1
    #lossweights=lossweights.to(device)
    #loss = torch.nn.CrossEntropyLoss(weight=lossweights, reduction='mean', label_smoothing=0.0)
    #loss  = torch.nn.NLLLoss(weight=lossweights, reduction='mean') #sammen med logsoftmax som nonlin lag
    #loss = MemoryEfficientSoftDiceLoss()
    #loss = torch.nn.MSELoss(reduction='mean').to(device)
    #loss = DiceSignLoss()

    loss = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {}, weight_ce=1, weight_dice=1,
                          ignore_label=0)

   
    epoch_loss = list()
    val_loss = list()   
    epoch_current_loss = 1E9

    if load_model:
        if os.path.exists("model.pt"):
            state = torch.load("model.pt")            
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])       
            sheduler.load_state_dict(state['sheduler'])
    for ind in range(n_epochs):
        mean_loss = train_one_epoch(model, loss, optimizer, sheduler, dataset, device, shuffle=False)        
        epoch_loss.append((ind+1, mean_loss)) 
        print(mean_loss , epoch_current_loss)       
        if mean_loss < epoch_current_loss: 
            state = {
                'epoch': ind,
                'model': model.state_dict(),
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
            plt.imshow(out_r[0,80,:,:])
            plt.subplot(2, 2, 4)
            plt.imshow(out[0,80,:,:])
            plt.show()
            

        

if __name__=='__main__':
    dataset_path = r"C:\Users\ander\totseg"
    #dataset_path = r"D:\totseg\Totalsegmentator_dataset_v201"
    batch_size=18
    #start_train(n_epochs = 15, device='cuda', batch_size=batch_size, load_model=True, data_path = dataset_path)
    start_train(n_epochs = 3, device='cpu', batch_size=batch_size, load_model=True, data_path = dataset_path)

    if False:
        dataset = TotSegDataset2D(dataset_path, train=True, batch_size=batch_size)
        predict(dataset)



    if True:
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