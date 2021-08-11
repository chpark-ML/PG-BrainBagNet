import socket
from datetime import datetime
import os
import GPUtil
import setting_2 as fst
import shutil
import utils as ut
import numpy as np
import time

""" fold """
kfold = 5
start_fold = 1
end_fold = 5

""" data """
data_type_num = 0
list_data_type = ['ADNI_JSY_3',]  # 0


""" task selection """
if list_data_type[data_type_num] == 'ADNI_JSY_3':
    list_class_type = ['NC', 'AD', 'sMCI', 'pMCI']
    list_class_for_train = [1, 1, 0, 0]
    list_class_for_test = [1, 1, 0, 0]
    # list_class_for_train = [0, 0, 1, 1]
    # list_class_for_test = [0, 0, 1, 1]

    list_class_for_total = [1, 1, 1, 1]  # for plotting

""" selected task """
list_selected_for_train = [] # ['NC', 'MCI', 'AD']
list_selected_for_test = [] # ['NC', 'MCI', 'AD']
list_selected_for_total = [] # ['NC', 'MCI', 'AD']
list_n_sample_selected = []
for i in range(len(list_class_for_total)):
    if list_class_for_total[i] == 1:
        list_selected_for_total.append(list_class_type[i])
    if list_class_for_train[i] == 1:
        list_selected_for_train.append(list_class_type[i])
    if list_class_for_test[i] == 1:
        list_selected_for_test.append(list_class_type[i])
num_class = len(list_selected_for_train)

""" eval metric """
list_standard_eval_dir = ['/val_loss']
list_standard_eval = ['{}'.format(list_standard_eval_dir[i][1:]) for i in range(len(list_standard_eval_dir))]
list_eval_metric = ['MAE', 'RMSE', 'R_squared',  'Acc', 'Sen', 'Spe', 'AUC']

""" parmas """
max_num_loss = 20  # maximum number of objective (need for plot)
flag_beta = True  # flag for balanced cross entropy (CE) (the value has been set to inverse of sample frequency)
label_smoothing = 0.1  # smoothing label in CE loss (e.g., {0.1, 0.9})
focal_gamma = 0.0  # focal loss (Lin et al., 2017)
entropy_gamma = 0.0

class hyperParam_storage_1():
    def __init__(self):
        super(hyperParam_storage_1, self).__init__()
        self.name = 'stage_1'
        self.epoch = 200
        self.batch_size = 4
        self.iter_to_update = 1
        self.v_batch_size = self.batch_size

        self.lr = 1e-4
        self.LR_decay_rate = 1.
        self.step_size = 1

        self.early_stopping_start_epoch = 1
        self.early_stopping_patience = 30
        self.weight_decay = 0

        self.loss_type = ['cls', 'entropy']
        self.flag_aux_cls = [0, 0]
        self.name_aux_cls = []
        self.num_aux_cls = len(self.name_aux_cls)
        self.num_total_loss = len(self.flag_aux_cls) - sum(self.flag_aux_cls) + len(self.name_aux_cls)

        self.loss_lambda = {
            self.loss_type[0]: 1.0,  # cls
            self.loss_type[1]: 0.01,  # entropy
        }
hyperParam_s1 = hyperParam_storage_1()

""" model setting """
model_arch_dir = "/model_arch"

model_num_0 = 2  # to train classifier, generator
model_name = [None] * 60
model_name[0] = "BrainBagNet"
model_name[1] = "FG_BrainBagNet"
model_name[2] = "PG_BrainBagNet"
dir_to_save_1 = './1_' + model_name[model_num_0]
Index_sizeOfPatch = 4
sizeOfPatch = [9, 17, 25, 41, 57] # patch size used in model training

# dir to load the AD diagnosis model parameters in training MCI conversion prediction model
list_dir_preTrain = [
    '/dir_to_load_pretrained_model',
    ]

Index_gatePool = 0
gatePool = ['gate']

flag_plot_epoch = 5  # should be dividable into 1000
if 'BrainBagNet' in model_name[model_num_0]:
    flag_plot = True
else:
    flag_plot = False

""" directory """
dir_root = '/DataCommon3/chpark/ADNI'
num_modality = 1
orig_data_dir = '/DataCommon/chpark/ADNI_orig_JSY'
tmp_data_path = '/' + list_data_type[data_type_num]
exp_data_dir = dir_root + '/ADNI_exp' + tmp_data_path
tadpole_dir = dir_root + '/TADPOLE-challenge/TADPOLE_D1_D2.csv'

x_range = [0, 193]
y_range = [0, 229]
z_range = [0, 193]
x_size = x_range[1] - x_range[0]
y_size = y_range[1] - y_range[0]
z_size = z_range[1] - z_range[0]

""" 1. raw npy dir """
orig_npy_dir = exp_data_dir + '/orig_npy'
ADNI_fold_image_path = []
ADNI_fold_age_path = []
ADNI_fold_MMSE_path = []

ADNI_fold_image_path_2 = []
ADNI_fold_age_path_2 = []
ADNI_fold_MMSE_path_2 = []

for i in range(len(list_class_type)):
    ADNI_fold_image_path.append(orig_npy_dir + "/ADNI_" + str(list_class_type[i]) + "_image.npy")
    ADNI_fold_age_path.append(orig_npy_dir + "/ADNI_" + str(list_class_type[i]) + "_age.npy")
    ADNI_fold_MMSE_path.append(orig_npy_dir + "/ADNI_" + str(list_class_type[i]) + "_MMSE.npy")

""" 2. fold index """
fold_index_dir = exp_data_dir + '/fold_index_5CV'
train_index_dir = []
val_index_dir = []
test_index_dir = []
for i in range(len(list_class_type)):
    train_index_dir.append(fold_index_dir + '/train_index_' + list_class_type[i])
    val_index_dir.append(fold_index_dir + '/val_index_' + list_class_type[i])
    test_index_dir.append(fold_index_dir + '/test_index_' + list_class_type[i])

""" experiment description """
exp_date = str(datetime.today().year) + '%02d'%datetime.today().month + '%02d'% datetime.today().day
exp_title = '/tmp'
exp_description = "classification"

""" data size"""
max_crop_size = [177, 213, 177]
min_crop_size = [177, 213, 177]
eval_crop_size = max_crop_size
crop_pad_size = (0, 0, 0, 0, 0, 0)
data_size = [1, x_size, y_size, z_size]

"""openpyxl setting """
push_start_row = 2

""" print out setting """
print(socket.gethostname())
print("data : {}".format(exp_date))
print(' ')
print("Dataset for train : {}".format(list_selected_for_train))
print("model arch 1 : {}".format(model_name[model_num_0]))
print(' ')
print("data type : {}".format(list_data_type[data_type_num]))
