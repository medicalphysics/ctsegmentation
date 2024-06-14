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


def train_one_epoch(model, loss, optimizer, data, device, shuffle=True, n_iter=100):
    ## see https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
    model.train(True)
    stop_ind = min(len(data), n_iter)
    pbar = tqdm(total=stop_ind, leave=False)
    for ind, (image, label) in enumerate(data.iter_ever(shuffle)):
        image = image.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        output = model(image)
        l = loss(output, label)
        l.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
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
    plt.savefig(name, dpi=600)
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
    train_shape=(256, 256),
    part=1,
    load_model=True,
    load_only_model=False,
    data_path=None,
):
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)

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
    model_path = os.path.join(model_dir, name)

    dataset_val = TotSegDataset2D(
        data_path,
        train=False,
        batch_size=batch_size,
        volumes=volumes,
        train_shape=train_shape,
        dtype=torch.float32,
    )

    dataset = TotSegDataset2D(
        data_path,
        train=True,
        batch_size=batch_size,
        volumes=volumes,
        train_shape=train_shape,
        dtype=torch.float32,
    )

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
        optimizer, "min", patience=2, cooldown=3, factor=0.2, threshold=0.00
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
        if os.path.exists(model_path):
            state = torch.load(model_path, map_location=torch.device(device))
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
            model, loss, optimizer, dataset, device, shuffle=True, n_iter=2000
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
            torch.save(state, model_path)
        plot_loss(
            validation_loss,
            lr_rate,
            dice_score,
            os.path.join(
                model_dir, "loss{}.png".format(part if part in range(1, 5) else "")
            ),
        )


def save_inference_model(input_shape, out_channel_size=16, part=1, device="cpu"):
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    model_path = os.path.join(model_dir, "model{}.pt".format(part))
    model = get_model(out_channel_size).to(device)
    state = torch.load(
        model_path,
        map_location=torch.device(device),
    )
    model.load_state_dict(state["model"])
    if device == 'cpu':
        full_model = torch.nn.Sequential(model, torch.nn.Softmax(dim=1))
    else:
        full_model = torch.nn.Sequential(model, torch.nn.Softmax(dim=1).cuda())
    full_model.eval()
    with torch.no_grad():
        trace = torch.jit.trace(
            full_model, torch.rand(input_shape, dtype=torch.float32).to(device)
        )
        if device == "cpu":
            trace.save(os.path.join(model_dir, "freezed_model{}.pt".format(part)))
        elif device == "cuda":
            trace.save(os.path.join(model_dir, "freezed_cuda_model{}.pt".format(part)))


def predict(data_path, part=0):
    if part == 1:
        volumes = list(range(1, 16))
    elif part == 2:
        volumes = list(range(16, 31))
    elif part == 3:
        volumes = list(range(31, 46))
    elif part == 4:
        volumes = list(range(46, 61))
    else:
        volumes = None
    data = TotSegDataset2D(
        data_path,
        train=False,
        batch_size=1,
        volumes=volumes,
        train_shape=(384, 384),
        dtype=torch.float32,
    )
    data.shuffle()
    models = list()
    if part not in range(1, 5):
        for i in range(1, 5):
            model_dir = os.path.join(os.path.dirname(__file__), "models")
            model_path = os.path.join(model_dir, "model{}.pt".format(i))
            model = get_model(16)
            state = torch.load(model_path, map_location=torch.device("cpu"))
            model.load_state_dict(state["model"])
            full_model = torch.nn.Sequential(model, torch.nn.Softmax(dim=1))
            full_model.eval()
            models.append(full_model)
    else:
        model_dir = os.path.join(os.path.dirname(__file__), "models")
        model_path = os.path.join(model_dir, "model{}.pt".format(part))
        model = get_model(data._label_tensor_dim)
        state = torch.load(model_path, map_location=torch.device("cpu"))
        model.load_state_dict(state["model"])
        full_model = torch.nn.Sequential(model, torch.nn.Softmax(dim=1))
        full_model.eval()
        models.append(full_model)

    with torch.no_grad():
        for idx in range(len(data)):
            image_b, label_b = data[idx]
            out_b = [
                full_model(image_b).detach().cpu().numpy() for full_model in models
            ]
            for nbatch in range(data._batch_size):
                image = np.squeeze(image_b.detach().cpu().numpy()[nbatch, 0, :, :])
                label = np.squeeze(label_b.detach().cpu().numpy()[nbatch, 0, :, :])
                out = [np.squeeze(b[nbatch, :, :, :]) for b in out_b]
                out_labels = np.zeros_like(image)
                for mIdx in range(len(models)):
                    for i in range(1, out[mIdx].shape[0]):
                        seg = np.squeeze(out[mIdx][i, :, :])
                        out_labels[seg > 0.5] = i + 15 * mIdx

                plt.subplot(2, 2, 1)
                plt.imshow(image)
                plt.subplot(2, 2, 2)
                plt.imshow(label)
                plt.subplot(2, 2, 3)
                plt.imshow(out_labels)
                plt.subplot(2, 2, 4)
                # plt.imshow(np.squeeze(out[1:, :, :].max(axis=0)))
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
    batch_size = 12

    # start_train(
    #    n_epochs=150,
    #    device="cuda",
    #    batch_size=batch_size,
    #    part=1,
    #    train_shape=(384, 384),
    #    load_model=True,
    #    load_only_model=False,
    #    data_path=dataset_path,
    # )
    # for part in range(1, 5):
    #     start_train(
    #         n_epochs=5,
    #         device="cuda",
    #         batch_size=batch_size,
    #         part=part,
    #         train_shape=(384, 384),
    #         load_model=True,
    #         load_only_model=False,
    #         data_path=dataset_path,
    #     )

    if True:
        for i in range(1, 5):
            save_inference_model((32, 1, 384, 384), 16, i, device="cpu")

    if False:
        predict(dataset_path, part=0)
