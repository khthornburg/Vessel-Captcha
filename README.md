## Semester Project

Listed bellow are the .py files that defined/ trained keras models and the .py that are their pytorch equivalent

- pnetcls.py -> pnetcls_PT.py 
- wnet.py -> wnet_PT.py
- train_pnetcls.py & train_wnetseg.py -> train_PT.py & getdata_PT.py

As we translated the code from keras to pytorch we shifted from a scripting to object oriented coding style. We did 

### pnetcls_PT.py: 

We defined a Pnet class which serves as a pytroch implementation of the PnetCls model.

### wnet_PT.py: 

We defined a Wnet class which serves as a pytroch implementation of the Wnet model.

### getdata_PT.py:

To train the models, the function defined in train_PT.py uses pytorch dataloaders built using dataset classes defined in get_data_PT.py. Futher comments in get_data_PT.py detail the initialization of the class.

### train_PT.py: 

To train the PnetCls and WnetSeg models, we use the following function defined in train_PT.py:
```sh
train_PT_model(train_patch_dir = str, label_patch_dir = str, model_filepath = str, 
               validation_split = float,batch_size = int, 
               patch_size = int, normalize = bool, pixel_wise = bool, 
               epochs = int, Model = Pnet/Wnet, lr = float, momentum = float,
               criterion, optimizer)
```
The trained model is saved in model_filepath
criterion and optimizer are taken from torch.nn and torch.optim respectively
