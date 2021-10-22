import json
import os

from discord_logger import log
from frameworks.ImageRegistrationInterface import ImageRegistrationInterface

os.environ['VXM_BACKEND'] = 'pytorch'
import time
import numpy as np
import torch
import torch as th
import voxelmorph as vxm
import SimpleITK as sitk
from torch.utils.tensorboard import SummaryWriter


class VoxelmorphTorch(ImageRegistrationInterface):
    @staticmethod
    def register_images(
            moving_image,
            fixed_image,
            weights_path:
        str = '/home/lschilling/PycharmProjects/image_registration_thesis/models/vxm_mnist.index',
            loss: str = 'MSE',
            gpu: str = '2'):
        device = th.device(f'cuda:{gpu}')
        moving_image_np = sitk.GetArrayFromImage(moving_image)[np.newaxis, :]
        fixed_image_np = sitk.GetArrayFromImage(fixed_image)[np.newaxis, :]
        model = vxm.networks.VxmDense.load(weights_path, device=device)
        model.to(device)
        model.eval()

        moving_image_th = torch.from_numpy(moving_image_np).to(
            device).float().unsqueeze(1)
        fixed_image_th = torch.from_numpy(fixed_image_np).to(
            device).float().unsqueeze(1)

        start_time = time.time()
        moved_image, displacement = model(moving_image_th,
                                          fixed_image_th,
                                          registration=True)
        end_time = time.time()
        if fixed_image.GetDimension() == 3:
            displacement = sitk.GetImageFromArray(displacement, isVector=True)
        if fixed_image.GetDimension() == 2:
            displacement = sitk.GetImageFromArray(displacement.permute(
                0, 2, 3, 1).squeeze().cpu().detach().numpy(),
                                                  isVector=True)
        return sitk.GetImageFromArray(moved_image.squeeze().cpu().detach(
        ).numpy()), displacement, end_time - start_time


def train_vxm_model(train_generator,
                    val_generator,
                    epochs: int = 10,
                    steps_per_epoch: int = 100,
                    gpu: str = '0',
                    multi_gpu: bool = False,
                    learning_rate: float = 0.001,
                    loss: str = 'MSE',
                    model_name: str = 'default_name',
                    dataset: str = 'mnist') -> str:
    start_time = time.time()
    model_dir = f'../models/{dataset}/'
    in_sample, out_sample = next(train_generator)
    fixed_images = in_sample[1]
    fixed_image = fixed_images[0]
    # configure unet input shape (concatenation of moving and fixed images)
    inshape = fixed_image.shape

    device = th.device(f'cuda:{gpu}')

    # configure unet features
    nb_features = [
        [16, 32, 32, 32],    # encoder features
        [32, 32, 32, 32, 32, 16, 16]    # decoder features
    ]
    int_downsize = 1
    model = vxm.networks.VxmDense(inshape=inshape,
                                  nb_unet_features=nb_features,
                                  bidir=False,
                                  int_steps=0,
                                  int_downsize=int_downsize)

    # prepare the model for training and send to device
    model.to(device)
    model.train()

    # set optimizer
    optimizer = th.optim.Adam(model.parameters(),
                              lr=learning_rate,
                              amsgrad=False,
                              eps=1e-7,
                              betas=(0.9, 0.999),
                              weight_decay=0)
    if len(fixed_image.shape) == 2:
        grad_loss = vxm.losses.Grad2D('l2', loss_mult=int_downsize).loss
    else:
        assert len(
            fixed_image.shape
        ) == 3, f'grad is only implemented for 2D and 3D but got {len(fixed_image.shape)}'
        grad_loss = vxm.losses.Grad('l2', loss_mult=int_downsize).loss
    losses = [vxm.losses.MSE().loss, grad_loss]
    weights = [1, 0.05]

    best_model = model
    best_loss = 100
    train_writer = SummaryWriter(
        f'../runs/logs/{dataset}/vxmth_{model_name}/train')
    val_writer = SummaryWriter(
        f'../runs/logs/{dataset}/vxmth_{model_name}/validation')
    # training loops
    for epoch in range(0, epochs):
        best_model.save(os.path.join(model_dir, 'best_model.pt'))

        epoch_loss = []
        epoch_total_loss = []

        for step in range(steps_per_epoch):
            step_start_time = time.time()

            # generate inputs (and true outputs) and convert them to tensors
            inputs, y_true = next(train_generator)
            inputs = [
                th.from_numpy(image).to(device).float().unsqueeze(1)
                for image in inputs
            ]
            y_true = [
                th.from_numpy(image).to(device).float().unsqueeze(1)
                for image in y_true
            ]
            y_true[0] = y_true[0]
            y_true[1] = y_true[1].permute(0, 4, 2, 3, 1)
            y_true[1] = y_true[1].squeeze(4)

            # run inputs through the model to produce a warped image and flow field
            y_pred = model(*inputs)

            y_pred = [image.unsqueeze(4) for image in y_pred]

            # calculate the total loss
            loss = 0
            loss_list = []
            for idx, loss_function in enumerate(losses):
                curr_loss = loss_function(y_true[idx],
                                          y_pred[idx]) * weights[idx]
                loss_list.append('%.6f' % curr_loss.item())
                loss += curr_loss
                if idx == 0 and curr_loss.item() < best_loss:
                    best_loss = curr_loss.item()
                    best_model = model

            epoch_loss.append(loss_list)
            epoch_total_loss.append(loss.item())

            # backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # print step info
            epoch_info = 'epoch: %04d' % (epoch + 1)
            step_info = ('step: %d/%d' % (step + 1, steps_per_epoch)).ljust(14)
            time_info = 'time: %.2f sec' % (time.time() - step_start_time)
            epoch_losses = [
                '%.6f' % f
                for f in np.mean(np.array(epoch_loss, dtype=np.float), axis=0)
            ]
            losses_info = ', '.join(epoch_losses)
            loss_info = 'loss: %.6f  (%s)' % (np.mean(epoch_total_loss),
                                              losses_info)
            print(' '.join((epoch_info, step_info, time_info, loss_info)),
                  flush=True)
        train_writer.add_scalar('epoch_loss', np.mean(epoch_total_loss), epoch)
        train_writer.add_scalar('epoch_flow_loss', float(epoch_losses[0]),
                                epoch)
        train_writer.add_scalar('epoch_transformer_loss',
                                float(epoch_losses[1]), epoch)
    end_time = time.time()
    model_name = f'vxmth_{model_name}_final_loss{str(np.mean(epoch_total_loss)).replace(".", "_")}'
    output_path = os.path.join(
        '/home/lschilling/PycharmProjects/MasterThesis/models', dataset,
        model_name)
    log(f'Finished training model was save to {output_path}')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    json.dump(end_time - start_time,
              open(os.path.join(output_path, 'train_time.json'), 'w'))
    # final model save
    model.save(os.path.join(output_path, 'model.pt'))
    return os.path.join(output_path, 'model.pt')
