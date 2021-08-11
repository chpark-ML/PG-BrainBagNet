import nibabel as nib
import numpy as np
import setting as st
import setting_2 as fst
from data_load import data_load as DL
import torch
from torch.autograd import Variable
import torch.nn as nn
import utils as ut

import os
from scipy import stats
from sklearn.metrics import confusion_matrix

def test(config, fold, model, loader, hyperParam, dir_to_load, dir_heatmap, dir_confusion):
    """ free all GPU memory """
    torch.cuda.empty_cache()
    test_loader = loader

    """ load the model """
    model_dir = ut.model_dir_to_load(fold, dir_to_load)
    if model_dir !=None:
        model.load_state_dict(torch.load(model_dir))
    model.eval()

    dict_result = ut.eval_classification_model(config, fold, 1000, test_loader, model, hyperParam, plot_flag= st.flag_plot, confusion_save_dir=dir_confusion)
    return dict_result
