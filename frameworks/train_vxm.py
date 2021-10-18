import json

from DataHandler import DataHandler
from VoxelmorphTF import train_vxm_model
#from VoxelmorphTorch import train_vxm_model

batch_size = 8
batch_size_val = 12
epochs = 30
steps = 100
learning_rate = 0.001
resampling = False
dataset = 'synthetic'
multi_gpu = True
dh = DataHandler()
if dataset == 'synthetic':
    dh.get_synthetic_data(
        fixed_path=
        '/home/lschilling/datam2olie/synthetic/orig/t3/Synthetic_CT/',
        moving_path=
        '/home/lschilling/datam2olie/synthetic/orig/t1/Synthetic_CT/')
elif dataset == 'mnist':
    dh.get_mnist_data(select_number=5)
else:
    raise NotImplementedError(
        f'{dataset} is not implemented yet please select one of the following losses [mnist, synthetic]'
    )

train_generator = dh.data_gen_voxelmorph(data_x=dh.x_train,
                                         data_y=dh.y_train,
                                         random_resampling=resampling,
                                         batch_size=batch_size,
                                         shuffle=True)
val_generator = dh.data_gen_voxelmorph(data_x=dh.x_val,
                                       data_y=dh.y_val,
                                       random_resampling=False,
                                       batch_size=batch_size_val,
                                       shuffle=False)

model_name = f'ep{epochs}_st{steps}_lr{str(learning_rate).replace(".", "_")}_bat{batch_size}{"withResampling" if resampling else ""}'

vxm_model, hist, model_name = train_vxm_model(train_generator,
                                              val_generator,
                                              multi_gpu=multi_gpu,
                                              steps_per_epoch=steps,
                                              learning_rate=learning_rate,
                                              loss='MSE',
                                              model_name=model_name,
                                              dataset=dataset,
                                              epochs=epochs)
