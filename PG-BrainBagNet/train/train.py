import os
import sys
import utils as ut
from torch.autograd import Variable
import torch
import torch.nn as nn
import setting as st
import setting_2 as fst
from data_load import data_load as DL
import numpy as np


def train(config, fold, model, dict_loader, optimizer, scheduler, hyperParam, list_dir_save_model, list_dir_heatmap, dir_pyplot, Validation=True, Test_flag = True):

    train_loader = dict_loader['train']
    val_loader = dict_loader['val']
    test_loader = dict_loader['test']

    EMS = ut.eval_metric_storage()
    list_selected_EMS = []
    list_ES = []
    for i_tmp in range(len(st.list_standard_eval_dir)):
        list_selected_EMS.append(ut.eval_selected_metirc_storage())
        list_ES.append(ut.EarlyStopping(delta=0, patience=hyperParam.early_stopping_patience, verbose=True))

    print('training')
    optimizer.zero_grad()
    optimizer.step()

    """ epoch """
    for epoch in range(1, config.num_epochs+1):
        scheduler.step()
        print(" ")
        print("---------------  epoch {} ----------------".format(epoch))

        """ print learning rate """
        for param_group in optimizer.param_groups:
            print('current LR : {}'.format(param_group['lr']))

        """ train """
        model, optimizer, EMS = ut.train_classification_model(config, fold, epoch, EMS, train_loader, model, optimizer, hyperParam)

        """ val """
        if Validation == True:
            print("------------------  val  --------------------------")
            dict_result = ut.eval_classification_model(config, fold, epoch, val_loader, model, hyperParam, plot_flag= False)
            val_loss = dict_result['Loss']
            acc = dict_result['Acc']
            auc = dict_result['AUC']

            """ save the metric """
            EMS.dict_val_metric['val_loss'].append(val_loss)
            EMS.dict_val_metric['val_acc'].append(acc)
            EMS.dict_val_metric['val_auc'].append(auc)

            EMS.val_step.append(EMS.total_train_step)
            print('Fold : %d, Epoch [%d/%d] val Loss = %f val Acc = %f' % (fold, epoch, config.num_epochs, val_loss, acc))

            """ save model """
            for i_tmp in range(len(list_selected_EMS)):
                save_flag = ut.model_save_through_validation(fold, epoch,
                                                             start_eval_epoch=hyperParam.early_stopping_start_epoch,
                                                             EMS=EMS,
                                                             selected_EMS=list_selected_EMS[i_tmp],
                                                             ES=list_ES[i_tmp],
                                                             model=model,
                                                             dir_save_model=list_dir_save_model[i_tmp],
                                                             metric_1=st.list_standard_eval[i_tmp], metric_2='',
                                                             save_flag=False)

        """ learning rate decay"""
        EMS.LR.append(optimizer.param_groups[0]['lr'])

        if Test_flag== True:
            print("------------------  test _ test dataset  --------------------------")
            dict_result = ut.eval_classification_model(config, fold, epoch, test_loader, model, hyperParam, plot_flag= st.flag_plot)

            acc = dict_result['Acc']
            auc = dict_result['AUC']
            test_loss = dict_result['Loss']

            """ pyplot """
            EMS.test_acc.append(acc)
            EMS.test_auc.append(auc)
            EMS.test_loss.append(test_loss)
            EMS.test_step.append(EMS.total_train_step)
            print('number of test samples : {}'.format(len(test_loader.dataset)))
            print('Fold : %d, Epoch [%d/%d] test Loss = %f test Acc = %f' % (fold, epoch, config.num_epochs, test_loss, acc))

        ##TODO : Plot
        if epoch % 1 == 0:
            ut.plot_training_info_1(fold, dir_pyplot, EMS,  hyperParam, flag='percentile', flag_match=False)

        ##TODO : early stop
        tmp_count = 0
        for i in range(len(list_ES)):
            if list_ES[i].early_stop == True:
                tmp_count += 1
        if tmp_count == len(list_ES):
            break

    """ release the model """
    del model, EMS
    torch.cuda.empty_cache()