from math import inf
import os
import numpy as np
import torch
import torch.distributed
from torch import nn, autocast
from typing import Any, Optional, Tuple, Callable
from tqdm import tqdm
from dynamic_network_architectures.architectures.unet import (
    PlainConvUNet,
    ResidualEncoderUNet,
)
from totseg_dataloader import TotSegDataset2D
from loss import DC_and_CE_loss, MemoryEfficientSoftDiceLoss, softmax_helper_dim1
from matplotlib import pylab as plt

torch.set_num_threads(os.cpu_count())


def train_one_epoch(model, loss, optimizer, data, device, shuffle=True, n_iter=None):
    ## see https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
    model.train(True)
    stop_ind = min(len(data), n_iter)
    pbar = tqdm(total=stop_ind, leave=False)
    for ind, (image, label) in enumerate(data.iter_ever(shuffle)):
        image = image.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        # if ind % 2 == 1:
        #   optimizer.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)
        output = model(image)
        l = loss(output, label)
        l.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
        # if ind % 2 == 1:
        #    optimizer.step()
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
    running_vloss = 0
    running_dice = 0
    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        pbar = tqdm(total=len(data), leave=False)
        for image, labels in data:
            image = image.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            out = model(image)
            vloss = loss(out, labels)
            vdice = loss.diceCoeff(out, labels)
            running_dice += float(vdice.cpu())
            running_vloss += float(vloss.cpu())
            pbar.update(1)
        pbar.close()

    return running_vloss / len(data), running_dice / len(data)


def plot_loss(loss, lr, dice, name="train"):
    for ind, (v, l) in enumerate(
        zip([loss, lr, dice], ["Validation Loss", "Learning rate", "Dice Score"])
    ):
        plt.subplot(1, 3, ind + 1)
        plt.plot(v)
        plt.xlabel("Batch nr")
        plt.ylabel(l)
    plt.tight_layout()
    plt.savefig(name + ".png", dpi=600)
    plt.clf()


def get_model(N):
    # model = PlainConvUNet(1, 5, (32, 64, 125, 256, 320), nn.Conv2d, 3, (1, 2, 2, 2, 2), (2, 2, 2, 2, 2), N,
    #                            (2, 2, 2, 2), conv_bias=False, norm_op=nn.BatchNorm2d, nonlin=nn.Softmax, nonlin_kwargs={'dim':1} ,deep_supervision=False)
    # return model
    # patch size is 256x256
    model = ResidualEncoderUNet(
        1,  # input channels
        7,  # n_stages
        (32, 64, 128, 256, 512, 512, 512),  # features per stage
        torch.nn.modules.conv.Conv2d,  # conv_op
        3,  # kernel sizes
        (1, 2, 2, 2, 2, 2, 2),  # strides
        (1, 3, 4, 6, 6, 6, 6),  # n_blocks per stage
        N,  # num classes
        (1, 1, 1, 1, 1, 1),  # n_conv_per_stage_decoder
        conv_bias=True,
        norm_op=torch.nn.modules.instancenorm.InstanceNorm2d,
        norm_op_kwargs={"eps": 1e-5, "affine": True},
        nonlin=torch.nn.LeakyReLU,
        nonlin_kwargs={"inplace": True},
    )
    return model


def start_train(
    n_epochs=150,
    device="cpu",
    batch_size=4,
    part=1,
    load_model=True,
    load_only_model=False,
    data_path=None,
):
    name = "model.pt"
    if part == 1:
        volumes = list(range(1, 16))
        name = "model1.pt"
    elif part == 2:
        volumes = list(range(16, 31))
        name = "model2.pt"
    elif part == 3:
        volumes = list(range(31, 46))
        name = "model3.pt"
    elif part == 4:
        volumes = list(range(46, 61))
        name = "model4.pt"

    dataset_val = TotSegDataset2D(
        data_path,
        train=False,
        batch_size=batch_size,
        volumes=volumes,
        dtype=torch.float32,
    )
    dataset = TotSegDataset2D(
        data_path,
        train=True,
        batch_size=batch_size,
        volumes=volumes,
        dtype=torch.float32,
    )
    # dataset = dataset_val

    model = get_model(dataset._label_tensor_dim).to(device)

    initial_lr = 0.01
    weight_decay = 3e-5
    optimizer = torch.optim.SGD(
        model.parameters(),
        initial_lr,
        weight_decay=weight_decay,
        momentum=0.99,
        nesterov=True,
    )
    sheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=2, cooldown=3, factor=0.2, threshold=0.01
    )
    loss = DC_and_CE_loss(
        {"batch_dice": False, "smooth": 1e-5, "do_bg": False},
        {"label_smoothing": 1e-5},
        weight_ce=0.2,
        weight_dice=1.8,
        ignore_label=None,
    ).to(device)

    validation_loss = list()
    dice_score = list()
    lr_rate = list()
    min_val_loss = 1e9

    if load_model:
        if os.path.exists(name):
            state = torch.load(name)
            model.load_state_dict(state["model"])
            if not load_only_model:
                optimizer.load_state_dict(state["optimizer"])
                sheduler.load_state_dict(state["sheduler"])
            validation_loss = state["validation_loss"]
            dice_score = state["dice_score"]
            lr_rate = state["lr_rate"]
            min_val_loss = min(validation_loss)

    for ind in range(n_epochs):
        train_one_epoch(
            model, loss, optimizer, dataset, device, shuffle=True, n_iter=1000
        )
        lossv, dicev = validate(model, dataset_val, loss, device)
        sheduler.step(lossv)
        lr_rate.append(sheduler.get_last_lr()[0])
        validation_loss.append(lossv)
        dice_score.append(dicev)
        if min_val_loss > lossv:
            min_val_loss = lossv
            state = {
                "epoch": ind,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "sheduler": sheduler.state_dict(),
                "validation_loss": validation_loss,
                "dice_score": dice_score,
                "lr_rate": lr_rate,
            }
            torch.save(state, name)
            # traced_script_module = torch.jit.trace(model.to('cpu'), torch.rand(dataset.batch_shape(), dtype=torch.float32))
            # traced_script_module.save("traced_model.pt")
        plot_loss(validation_loss, lr_rate, dice_score, name)


def save_model(input_shape, out_channel_size=16, part=1):
    model = get_model(out_channel_size)
    state = torch.load(
        "model{}.pt".format(part),
        map_location=torch.device("cpu"),
    )
    model.load_state_dict(state["model"])
    full_model = torch.nn.Sequential(model, torch.nn.Softmax(dim=1))
    full_model.eval()

    # full_model = torch.compile(full_model)
    # trace = torch.jit.trace(full_model, torch.rand(input_shape, dtype=torch.float32))
    trace = torch.jit.optimize_for_inference(torch.jit.script(full_model.eval()))
    trace = torch.jit.freeze(trace)
    trace.save("freezed_model{}.pt".format(part))


def predict(data, part=1):
    model = get_model(data._label_tensor_dim)
    state = torch.load("model{}.pt".format(part), map_location=torch.device("cpu"))
    model.load_state_dict(state["model"])
    model.eval()
    with torch.no_grad():
        for idx in range(len(data)):
            batch = data._item_splits[idx]
            image, label = data[idx]
            # label_index = label.mul(torch.arange(label.shape[1]).reshape((label.shape[1], 1, 1))).sum(dim=1, keepdim=True)
            out = softmax_helper_dim1(model(image))
            seg = np.zeros(image.shape[2:])
            for i in range(1, out.shape[1]):
                s = np.squeeze(out[0, i, :, :])
                seg[s > 0.5] = i
            plt.subplot(1, 3, 1)
            plt.imshow(image[0, 0, :, :])
            plt.subplot(1, 3, 2)
            plt.imshow(label[0, 0, :, :])
            plt.subplot(1, 3, 3)
            plt.imshow(seg)
            print(batch)
            plt.show()


def print_model(dataset_path):
    d = TotSegDataset2D(
        dataset_path,
        train=True,
        batch_size=16,
    )
    model = get_model(16)
    data, _ = d[0]

    from torchview import draw_graph

    model_graph = draw_graph(
        model,
        input_size=data.shape,
        device="meta",
        save_graph=True,
        graph_dir="LR",
        file_format="pdf",
    )
    model_graph.visual_graph


if __name__ == "__main__":
    # dataset_path = r"C:\Users\ander\totseg"
    dataset_path = r"D:\totseg\Totalsegmentator_dataset_v201"
    batch_size = 28

    start_train(
        n_epochs=150,
        device="cuda",
        batch_size=batch_size,
        part=3,
        load_model=True,
        load_only_model=False,
        data_path=dataset_path,
    )
    # start_train(n_epochs = 3, device='cpu', batch_size=batch_size, load_model=True, data_path = dataset_path)

    if False:
        volumes = list([10, 11, 12, 13, 14])
        dataset = TotSegDataset2D(
            dataset_path,
            train=False,
            batch_size=batch_size,
            volumes=volumes,
            dtype=torch.float32,
        )
        save_model(dataset)
        dataset.shuffle()
        predict(dataset)

    if False:
        d = TotSegDataset2D(dataset_path, train=True, batch_size=16)
        image, label = d[8]
        model = get_model(d._label_tensor_dim)
        state = torch.load("model.pt")
        model.load_state_dict(state["model"])
        model.eval()
        with torch.no_grad():
            # out = torch.sigmoid(model(image)).detach()
            out = model(image).detach()

        print(image.shape, label.shape, out.shape)

        # label_exp=label.mul(torch.arange(label.shape[1]).reshape((label.shape[1], 1, 1))).sum(dim=1, keepdim=True)
        for i in range(image.shape[0]):
            plt.subplot(1, 3, 1)
            plt.imshow(image[0, 0, :, :])
            plt.subplot(1, 3, 2)
            plt.imshow(label[0, 0, :, :])
            plt.subplot(1, 3, 3)
            plt.imshow(out[0, 80, :, :])
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
