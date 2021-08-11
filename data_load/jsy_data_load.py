import pickle
import numpy as np
import setting as st
import pandas as pd
import utils as ut
import os
import nibabel as nib

def Prepare_data_1():
    """ load data jsy processed """
    dat_dir = st.orig_data_dir + '/data.npy'
    cls_dir = st.orig_data_dir + '/label.npy'
    # age_dir = st.orig_data_dir + '/adni_age.npy'
    # id_dir = st.orig_data_dir + '/adni_id.npy'

    adni_dat = np.load(dat_dir, mmap_mode='r')
    adni_cls = np.load(cls_dir, mmap_mode='r')
    # adni_age = np.load(age_dir, mmap_mode='r')
    # adni_id = np.load(id_dir, mmap_mode='r')

    # t_adni_cls = adni_cls

    """ allocation memory """
    list_image_memalloc = []
    list_age_memallow = []
    list_MMSE_memallow = []


    """ the # of the subject depending on the disease label """
    unique, counts = np.unique(adni_cls, return_counts=True)

    n_NC_subjects = counts[0]
    n_MCI_subjects = counts[1]
    n_AD_subjects = counts[2]
    list_n_subjects = [n_NC_subjects, n_MCI_subjects, n_AD_subjects]
    # n_sMCI_subjects = list_final_label.count(1)
    # n_pMCI_subjects = list_final_label.count(2)
    # list_n_subjects = [n_NC_subjects, n_MCI_subjects, n_AD_subjects, n_sMCI_subjects, n_pMCI_subjects]

    for i in range (len(st.list_class_type)):
        list_image_memalloc.append(np.memmap(filename=st.ADNI_fold_image_path[i], mode="w+", shape=(list_n_subjects[i], st.num_modality, st.x_size, st.y_size, st.z_size), dtype=np.float32))
        list_age_memallow.append(np.memmap(filename=st.ADNI_fold_age_path[i], mode="w+", shape=(list_n_subjects[i], 1), dtype=np.float32))
        list_MMSE_memallow.append(np.memmap(filename=st.ADNI_fold_MMSE_path[i], mode="w+", shape=(list_n_subjects[i], 1), dtype=np.float32))
    #
    """ save the data """
    count_NC = 0
    count_MCI = 0
    count_AD = 0
    count_total_samples = 0
    for j in range(adni_dat.shape[0]):
        print(f'{j}th subject.')
        count_total_samples +=1
        if adni_cls[j] == 0:
            list_image_memalloc[0][count_NC, 0, :, :, :]= np.squeeze(adni_dat[j])
            # list_age_memallow[0][count_NC] = np.squeeze(adni_age[j])
            count_NC += 1

        elif adni_cls[j] == 1:
            list_image_memalloc[1][count_MCI, 0, :, :, :]= np.squeeze(adni_dat[j])
            # list_age_memallow[1][count_MCI] = np.squeeze(adni_age[j])
            count_MCI += 1

        elif adni_cls[j] == 2:
            list_image_memalloc[2][count_AD, 0, :, :, :]= np.squeeze(adni_dat[j])
            # list_age_memallow[2][count_AD] = np.squeeze(adni_age[j])
            count_AD += 1

    print("count nc : " + str(count_NC)) # 284
    print("count mci : " + str(count_MCI)) # 374
    print("count ad : " + str(count_AD))  # 329


def Prepare_data_2():
    """ file smri_orig """
    smri_info_dir = '/DataCommon/chpark/ADNI_orig_JSY/info/smri_orig_info.csv'
    data = pd.read_csv(smri_info_dir)
    data = data.sort_values(by=['RID', 'Image ID'])

    """ info """
    n_img = (data['Image ID']).unique().shape[0]  # (21280, )
    label_type = (data['Research Group']).unique()  # ['CN', 'AD', 'MCI', 'EMCI', 'Patient', 'LMCI', 'SMC']

    """ if phase is nan, the phase is ADNI 2. """
    phase_type = data['Phase'].unique()  # [ANDI 1, ADNI GO, ADNI 2, ADNI 3, nan]
    n_phase_wo_nan = data.count()['Phase']  # (21274)
    data['Phase'].value_counts()  # 9317+9105+1653+1199
    for tmp_i in range(data['Phase'][:].isnull().shape[0]):
        if data['Phase'][tmp_i] not in phase_type:
            data['Phase'][tmp_i] = 'ADNI 2'

    """ sbject """
    n_sbj = (data['Subject ID']).unique().shape[0]  # (2428, )
    sbj_type = data['Subject ID'].unique()

    """ visit 1 """
    visit_type = (data['Visit 1']).unique()  # []
    count = 0
    for tmp_i in range(data['Visit 1'].unique().shape[0]):
        if 'ADNI2' in data['Visit 1'].unique()[tmp_i]:
            count += 1

    """ file diagnosis change """
    diag_change_dir = '/DataCommon/chpark/ADNI_orig_JSY/info/diagnosis_change.csv'
    diag_change_data = pd.read_csv(diag_change_dir)
    diag_change_data = diag_change_data.sort_values(by=['RID', 'ImageID'])
    potential_pMCI = diag_change_data['ImageID'].unique()

    """ file DXSUM """
    DXSUM_dir = '/DataCommon/chpark/ADNI_orig_JSY/info/DXSUM_PDXCONV_ADNIALL.csv'
    DXSUM_data = pd.read_csv(DXSUM_dir)
    DXSUM_data = DXSUM_data.sort_values(by=['RID', 'EXAMDATE'])

    """ start """
    """ start """
    """ start """
    flag_stndard_MCI = True  # When False, the standard would be applied
    list_standard_MCI = ['m36', 'm48', 'm60', 'm72', 'm84', 'm96']
    list_standard_sMCI = ['m06', 'm12', 'm18', 'm24', 'm36', 'm48', 'm60', 'm72', 'm84', 'm96']
    # list_standard_sMCI = ['m06', 'm12', 'm18', 'm24', 'm36']
    list_standard_pMCI = ['m06', 'm12', 'm18', 'm24', 'm36']

    ## TODO : Without considering MPRAGE Repeat

    # ['CN', 'AD', 'MCI', 'EMCI', 'Patient', 'LMCI', 'SMC']
    # ADNI 1 : [[231, 200, 414, 0, 0, 0, 0],
    # ADNI GO : [0, 0, 0, 142, 5, 0, 0],
    # ADNI 2 : [202, 159, 0, 191, 0, 178, 111],
    # ADNI 3 : [332, 66, 193, 0, 4, 0, 0],
    # NaN : [0, 0, 0, 0, 0, 0, 0]]

    num_img_label_wise_BL = [np.zeros(label_type.shape) for _ in range(phase_type.shape[0])]  # (phase, label)
    ID_img_label_wise_BL = [[[] for _ in range(label_type.shape[0])] for _ in
                            range(phase_type.shape[0])]  # (phase, label)
    phase_label_wise_imageID = [[[] for _ in range(4)] for _ in range(2)]
    last_PTID = ''
    cur_PTID = ''

    cur_phase = ''
    last_phase = ''
    count_n_BL = 0

    count_Rev_from_AD = np.zeros(2)
    count_empty_in_diag = np.zeros(2)
    count_not_matching = np.zeros(2)
    count_if_both_appear = 0

    list_test = []
    list_test_2 = []
    excluded_year = ['2019', '2020']
    for i in range(n_img):
        cur_image_ID = data['Image ID'][i]
        cur_PTID = data['Subject ID'][i]  # subject id
        cur_RID = data['RID'][i]
        cur_phase = data['Phase'][i]
        cur_research_group = data['Research Group'][i]
        cur_visit = data['Visit 1'][i]
        cur_StudyDate = data['Study Date'][i]
        cur_ImgProt = data['Imaging Protocol'][i]
        if last_PTID == cur_PTID:  # same subject
            pass

        else:
            if not any(s in cur_StudyDate for s in excluded_year):
                if (cur_phase == 'ADNI 1') and (cur_visit == 'ADNI Screening') and (cur_ImgProt[cur_ImgProt.index(
                        'Field Strength') + len('Field Strength') + 1: cur_ImgProt.index('Field Strength') + len(
                        'Field Strength') + 4] == '1.5'):
                    # if (cur_phase == 'ADNI 1') and (cur_visit == 'ADNI Baseline' or cur_visit == 'ADNI Screening'):
                    # if cur_visit == 'ADNI Baseline':
                    count_n_BL += 1

                    ID_img_label_wise_BL[
                        phase_type.tolist().index(cur_phase)][
                        label_type.tolist().index(cur_research_group)
                    ].append(cur_image_ID)

                    num_img_label_wise_BL[
                        phase_type.tolist().index(cur_phase)][
                        label_type.tolist().index(cur_research_group)
                    ] += 1
                    last_PTID = cur_PTID

                    if cur_research_group == 'CN':
                        phase_label_wise_imageID[0][0].append(cur_image_ID)

                    elif cur_research_group == 'AD':
                        phase_label_wise_imageID[0][1].append(cur_image_ID)

                    elif 'MCI' in cur_research_group:  ##TODO : MCI conversion
                        tmp_df = DXSUM_data[(DXSUM_data["RID"] == cur_RID)]
                        if tmp_df['DXCURREN'].empty:
                            # print('empty : {}'.format(cur_PTID))
                            count_empty_in_diag[0] += 1
                            pass
                        elif tmp_df['DXCURREN'].iloc[0] != 2:
                            # print('not matching : {}'.format(cur_PTID))
                            count_not_matching[0] += 1
                            pass
                        else:
                            ## TODO #1 no reversion
                            check_rev = False  # default : False
                            for i_tmp in range(tmp_df.shape[0]):
                                if i_tmp == 0:
                                    prev_DX = tmp_df['DXCURREN'].iloc[i_tmp]
                                else:
                                    if prev_DX == 3 and prev_DX > tmp_df['DXCURREN'].iloc[i_tmp]:
                                        # print('REV : {}'.format(cur_PTID))
                                        count_Rev_from_AD[0] += 1
                                        check_rev = True

                            ## TODO #2 At least one diagnosis is more than 36
                            check_more_than_36 = flag_stndard_MCI  # default : False
                            for i_tmp in range(tmp_df.shape[0]):
                                if tmp_df['VISCODE2'].iloc[i_tmp] in list_standard_MCI:
                                    check_more_than_36 = True

                            ##
                            if check_rev == False and check_more_than_36 == True:
                                tmp_df_2 = tmp_df[tmp_df['DXCURREN'] == 3]  # take only AD
                                if tmp_df_2['VISCODE2'].empty:
                                    # pass
                                    phase_label_wise_imageID[0][2].append(cur_image_ID)
                                elif not any(
                                        tmp_df_2['VISCODE2'].iloc[0] in s for s in list_standard_sMCI):  ##TODO : sMCI
                                    phase_label_wise_imageID[0][2].append(cur_image_ID)
                                elif tmp_df_2['VISCODE2'].iloc[0] in list_standard_pMCI:  ##TODO : pMCI
                                    phase_label_wise_imageID[0][3].append(cur_image_ID)
                                else:  ##TODO : else
                                    pass
                                    # MCI_Conversion[0][0].append(cur_image_ID)
                                    # print('else')


                # elif (cur_phase == 'ADNI 2') and ('Year' not in cur_visit):
                # elif (cur_phase == 'ADNI 2') and ('ADNI2 Initial' in cur_visit or 'ADNI2 Screening' in cur_visit):
                elif (cur_phase == 'ADNI 2') and ('ADNI2 Screening' in cur_visit) and (cur_ImgProt[cur_ImgProt.index(
                        'Field Strength') + len('Field Strength') + 1: cur_ImgProt.index('Field Strength') + len(
                        'Field Strength') + 4] == '3.0'):

                    ## TODO : check if appear in 'ADNI 1'
                    tmp_df = data[(data['Subject ID'] == cur_PTID)]
                    tmp_df = tmp_df[(tmp_df['Phase'] == 'ADNI 1')]
                    if tmp_df.shape[0] != 0:
                        count_if_both_appear += 1
                    count_n_BL += 1

                    ID_img_label_wise_BL[
                        phase_type.tolist().index(cur_phase)][
                        label_type.tolist().index(cur_research_group)
                    ].append(cur_image_ID)

                    num_img_label_wise_BL[
                        phase_type.tolist().index(cur_phase)][
                        label_type.tolist().index(cur_research_group)
                    ] += 1
                    last_PTID = cur_PTID

                    if cur_research_group == 'CN':
                        phase_label_wise_imageID[1][0].append(cur_image_ID)

                    elif cur_research_group == 'AD':
                        phase_label_wise_imageID[1][1].append(cur_image_ID)

                    elif 'MCI' in cur_research_group:
                        tmp_df = DXSUM_data[(DXSUM_data["RID"] == cur_RID)]
                        if tmp_df['DXCHANGE'].empty:
                            # print('empty : {}'.format(cur_PTID))
                            count_empty_in_diag[1] += 1
                        elif tmp_df['DXCHANGE'].iloc[0] != 2:
                            # print('not matching : {}'.format(cur_PTID))
                            count_not_matching[1] += 1
                        else:
                            ## TODO #1 no reversion
                            check_rev = False  # default : False
                            for i_tmp in range(tmp_df.shape[0]):
                                if tmp_df['DXCHANGE'].iloc[i_tmp] == 8 or tmp_df['DXCHANGE'].iloc[i_tmp] == 9:
                                    count_Rev_from_AD[1] += 1
                                    print(cur_PTID)
                                    check_rev = True

                            ## TODO #2 At least one diagnosis is more than 36
                            check_more_than_36 = flag_stndard_MCI  # default : False
                            for i_tmp in range(tmp_df.shape[0]):
                                if tmp_df['VISCODE2'].iloc[i_tmp] in list_standard_MCI:
                                    check_more_than_36 = True

                            if check_rev == False and check_more_than_36 == True:
                                tmp_df_2 = tmp_df[(tmp_df['DXCHANGE'] == 3) | (tmp_df['DXCHANGE'] == 5)]  # take only AD
                                if tmp_df_2['VISCODE2'].empty:
                                    # pass
                                    phase_label_wise_imageID[1][2].append(cur_image_ID)
                                elif not any(
                                        tmp_df_2['VISCODE2'].iloc[0] in s for s in list_standard_sMCI):  ##TODO : sMCI
                                    phase_label_wise_imageID[1][2].append(cur_image_ID)
                                elif tmp_df_2['VISCODE2'].iloc[0] in list_standard_pMCI:  ##TODO : pMCI
                                    phase_label_wise_imageID[1][3].append(cur_image_ID)
                                else:  ##TODO : else
                                    pass
                                    # MCI_Conversion[0][0].append(cur_image_ID)
                                    # print('else')

                else:  # not ADNI 1 or 2
                    pass
    # print(num_img_label_wise_BL)

    print("count_appear_in_both : {}".format(count_if_both_appear))
    print("ADNI1, NC : {}".format(len(phase_label_wise_imageID[0][0])))
    print("ADNI1, AD : {}".format(len(phase_label_wise_imageID[0][1])))
    print("ADNI2, NC : {}".format(len(phase_label_wise_imageID[1][0])))
    print("ADNI2, AD : {}".format(len(phase_label_wise_imageID[1][1])))
    print('------------------------------------------------------')
    print("ADNI1, sMCI : {}".format(len(phase_label_wise_imageID[0][2])))
    print("ADNI1, pMCI : {}".format(len(phase_label_wise_imageID[0][3])))
    print("ADNI2, sMCI : {}".format(len(phase_label_wise_imageID[1][2])))
    print("ADNI2, pMCI : {}".format(len(phase_label_wise_imageID[1][3])))

    print('empty : {}'.format(count_empty_in_diag))
    print('not matching : {}'.format(count_not_matching))
    print('Rev from AD : {}'.format(count_Rev_from_AD))

    """ plot image jsy processed """
    dir_img = '/DataCommon/jsyoon/data/smri_orig/preprocessed'
    # save_dir = './Data_check/AD_diagnosis'
    # ut.make_dir(dir=save_dir, flag_rm=False)
    # count_img = 0
    # phase_type_2 = ['ADNI1', 'ADNI2']
    # class_type_2 = ['CN', 'AD']
    # for i_phase in range(len(AD_diagnosis)):
    #     for i_class in range(len(AD_diagnosis[i_phase])):
    #
    #         for i_img in range(len(AD_diagnosis[i_phase][i_class])):
    #             img = nib.load(dir_img + '/{}_flirt_restore.nii.gz'.format(AD_diagnosis[i_phase][i_class][i_img])).get_fdata(dtype=np.float32)
    #             # img = np.expand_dims(img, axis = 0)
    #             count_img += 1
    #             ut.plot_heatmap_without_overlay(img, save_dir + '/img_{}'.format(count_img), fig_title='Img_{}_{}_{:.2f}_{:.2f}'.format(phase_type_2[i_phase], class_type_2[i_class], img.max(), img.min()), thresh=0.0, percentile=0)


    # save_dir = './Data_check/MCI_conversion'
    # ut.make_dir(dir=save_dir, flag_rm=False)
    # count_img = 0
    # phase_type_2 = ['ADNI1', 'ADNI2']
    # class_type_2 = ['sMCI', 'pMCI']
    # for i_phase in range(len(MCI_Conversion)):
    #     if i_phase == 1:
    #         for i_class in range(len(MCI_Conversion[i_phase])):
    #             for i_img in range(len(MCI_Conversion[i_phase][i_class])):
    #                 img = nib.load(dir_img + '/{}_flirt_restore.nii.gz'.format(MCI_Conversion[i_phase][i_class][i_img])).get_fdata(dtype=np.float32)
    #                 count_img += 1
    #                 ut.plot_heatmap_without_overlay(img, save_dir + '/img_{}'.format(count_img), fig_title='Img_{}_{}_{:.2f}_{:.2f}'.format(phase_type_2[i_phase], class_type_2[i_class], img.max(), img.min()), thresh=0.0, percentile=0)

    list_image_memalloc = []
    list_age_memalloc = []
    list_MMSE_memalloc = []

    list_image_memalloc_2 = []
    list_age_memalloc_2 = []
    list_MMSE_memalloc_2 = []

    """ ADNI 1"""
    for i in range (len(st.list_class_type)):
        list_image_memalloc.append(np.memmap(filename=st.ADNI_fold_image_path[i], mode="w+", shape=(len(phase_label_wise_imageID[0][i]), st.num_modality, st.x_size, st.y_size, st.z_size), dtype=np.float32))
        list_age_memalloc.append(np.memmap(filename=st.ADNI_fold_age_path[i], mode="w+", shape=(len(phase_label_wise_imageID[0][i]), 1), dtype=np.float32))
        list_MMSE_memalloc.append(np.memmap(filename=st.ADNI_fold_MMSE_path[i], mode="w+", shape=(len(phase_label_wise_imageID[0][i]), 1), dtype=np.float32))

    """ ADNI 2"""
    for i in range (len(st.list_class_type)):
        list_image_memalloc_2.append(np.memmap(filename=st.ADNI_fold_image_path_2[i], mode="w+", shape=(len(phase_label_wise_imageID[1][i]), st.num_modality, st.x_size, st.y_size, st.z_size), dtype=np.float32))
        list_age_memalloc_2.append(np.memmap(filename=st.ADNI_fold_age_path_2[i], mode="w+", shape=(len(phase_label_wise_imageID[1][i]), 1), dtype=np.float32))
        list_MMSE_memalloc_2.append(np.memmap(filename=st.ADNI_fold_MMSE_path_2[i], mode="w+", shape=(len(phase_label_wise_imageID[1][i]), 1), dtype=np.float32))

    """ save the data """
    dir_img = '/DataCommon/jsyoon/data/smri_orig/preprocessed'
    for i_modality in range (1):
        for j_class in range (len(st.list_class_type)):
            for i_img in range(len(phase_label_wise_imageID[0][j_class])):
                print("modality: {}, class: {}, n_sample: {}".format(i_modality, j_class, i_img))
                tmp_img = nib.load(dir_img + '/{}_flirt_restore.nii.gz'.format(phase_label_wise_imageID[0][j_class][i_img])).get_fdata(dtype=np.float32)
                tmp_age = data[data['Image ID'] == phase_label_wise_imageID[0][j_class][i_img]]['Age'].iloc()[0]
                tmp_MMSE = data[data['Image ID'] == phase_label_wise_imageID[0][j_class][i_img]]['MMSE Total Score'].iloc()[0]

                list_image_memalloc[j_class][i_img, i_modality, :, :, :] = tmp_img
                list_age_memalloc[j_class][i_img, 0] = tmp_age
                list_MMSE_memalloc[j_class][i_img, 0] = tmp_MMSE

    for i_modality in range (1):
        for j_class in range (len(st.list_class_type)):
            for i_img in range(len(phase_label_wise_imageID[1][j_class])):
                print("modality: {}, class: {}, n_sample: {}".format(i_modality, j_class, i_img))
                tmp_img = nib.load(dir_img + '/{}_flirt_restore.nii.gz'.format(phase_label_wise_imageID[1][j_class][i_img])).get_fdata(dtype=np.float32)
                tmp_age = data[data['Image ID'] == phase_label_wise_imageID[1][j_class][i_img]]['Age'].iloc()[0]
                tmp_MMSE = data[data['Image ID'] == phase_label_wise_imageID[1][j_class][i_img]]['MMSE Total Score'].iloc()[0]

                list_image_memalloc_2[j_class][i_img, i_modality, :, :, :] = tmp_img
                list_age_memalloc_2[j_class][i_img, 0] = tmp_age
                list_MMSE_memalloc_2[j_class][i_img, 0] = tmp_MMSE



def Prepare_data_3():
    """ file smri_orig """
    smri_info_dir = '/DataCommon/chpark/ADNI_orig_JSY/info/smri_orig_info.csv'
    data = pd.read_csv(smri_info_dir)
    data = data.sort_values(by=['RID', 'Image ID'])

    """ info """
    n_img = (data['Image ID']).unique().shape[0]  # (21280, )
    label_type = (data['Research Group']).unique()  # ['CN', 'AD', 'MCI', 'EMCI', 'Patient', 'LMCI', 'SMC']

    """ if phase is nan, the phase is ADNI 2. """
    phase_type = data['Phase'].unique()  # [ANDI 1, ADNI GO, ADNI 2, ADNI 3, nan]
    n_phase_wo_nan = data.count()['Phase']  # (21274)
    data['Phase'].value_counts()  # 9317+9105+1653+1199
    for tmp_i in range(data['Phase'][:].isnull().shape[0]):
        if data['Phase'][tmp_i] not in phase_type:
            data['Phase'][tmp_i] = 'ADNI 2'

    """ sbject """
    n_sbj = (data['Subject ID']).unique().shape[0]  # (2428, )
    sbj_type = data['Subject ID'].unique()

    """ visit 1 """
    visit_type = (data['Visit 1']).unique()  # []
    count = 0
    for tmp_i in range(data['Visit 1'].unique().shape[0]):
        if 'ADNI2' in data['Visit 1'].unique()[tmp_i]:
            count += 1

    """ file diagnosis change """
    # diag_change_dir = '/DataCommon/chpark/ADNI_orig_JSY/info/diagnosis_change.csv'
    # diag_change_data = pd.read_csv(diag_change_dir)
    # diag_change_data = diag_change_data.sort_values(by=['RID', 'ImageID'])
    # potential_pMCI = diag_change_data['ImageID'].unique()

    """ file DXSUM """
    DXSUM_dir = '/DataCommon/chpark/ADNI_orig_JSY/info/DXSUM_PDXCONV_ADNIALL.csv'
    DXSUM_data = pd.read_csv(DXSUM_dir)
    DXSUM_data = DXSUM_data.sort_values(by=['RID', 'EXAMDATE'])

    """ start """
    """ start """
    """ start """
    flag_stndard_MCI = False  # When False, the standard would be applied
    # list_standard_MCI = ['m36', 'm48', 'm60', 'm72', 'm84', 'm96']
    list_standard_MCI = ['m36', 'm48', 'm60', 'm72']
    # list_standard_sMCI = ['m06', 'm12', 'm18', 'm24', 'm36', 'm48', 'm60', 'm72', 'm84', 'm96']
    list_standard_sMCI = ['m06', 'm12', 'm18', 'm24', 'm36', 'm48', 'm60', 'm72']
    # list_standard_sMCI = ['m06', 'm12', 'm18', 'm24', 'm36']
    list_standard_pMCI = ['m06', 'm12', 'm18', 'm24', 'm36']

    ## TODO : Without considering MPRAGE Repeat

    # ['CN', 'AD', 'MCI', 'EMCI', 'Patient', 'LMCI', 'SMC']
    # ADNI 1 : [[231, 200, 414, 0, 0, 0, 0],
    # ADNI GO : [0, 0, 0, 142, 5, 0, 0],
    # ADNI 2 : [202, 159, 0, 191, 0, 178, 111],
    # ADNI 3 : [332, 66, 193, 0, 4, 0, 0],
    # NaN : [0, 0, 0, 0, 0, 0, 0]]

    num_img_label_wise_BL = [np.zeros(label_type.shape) for _ in range(phase_type.shape[0])]  # (phase, label)
    ID_img_label_wise_BL = [[[] for _ in range(label_type.shape[0])] for _ in range(phase_type.shape[0])]  # (phase, label)
    phase_label_wise_imageID = [[[] for _ in range(4)] for _ in range(2)]
    phase_label_wise_PTID = [[[] for _ in range(4)] for _ in range(2)]
    last_PTID = ''
    cur_PTID = ''
    cur_phase = ''
    last_phase = ''
    count_n_BL = 0

    count_Rev_from_AD = np.zeros(2)
    count_empty_in_diag = np.zeros(2)
    count_not_matching = np.zeros(2)
    count_if_both_appear = 0

    list_test = []
    list_test_2 = []
    # excluded_year = ['2019', '2020']
    excluded_year = []
    tmp_counter_1 = 0
    tmp_counter_2 = 0
    for i in range(n_img):
        cur_image_ID = data['Image ID'][i]
        cur_PTID = data['Subject ID'][i]  # subject id
        cur_RID = data['RID'][i]
        cur_phase = data['Phase'][i]
        cur_research_group = data['Research Group'][i]
        cur_visit = data['Visit 1'][i]
        cur_StudyDate = data['Study Date'][i]
        cur_ImgProt = data['Imaging Protocol'][i]
        if last_PTID == cur_PTID:  # same subject
            pass

        else:
            if not any(s in cur_StudyDate for s in excluded_year):
                if (cur_phase == 'ADNI 1') and ('Screening' in cur_visit) and (cur_ImgProt[cur_ImgProt.index(
                        'Field Strength') + len('Field Strength') + 1: cur_ImgProt.index('Field Strength') + len(
                        'Field Strength') + 4] == '1.5'):
                    # if (cur_phase == 'ADNI 1') and (cur_visit == 'ADNI Baseline' or cur_visit == 'ADNI Screening'):
                    # if cur_visit == 'ADNI Baseline':
                    count_n_BL += 1

                    ID_img_label_wise_BL[
                        phase_type.tolist().index(cur_phase)][
                        label_type.tolist().index(cur_research_group)
                    ].append(cur_image_ID)

                    num_img_label_wise_BL[
                        phase_type.tolist().index(cur_phase)][
                        label_type.tolist().index(cur_research_group)
                    ] += 1
                    last_PTID = cur_PTID

                    if cur_research_group == 'CN':
                        phase_label_wise_imageID[0][0].append(cur_image_ID)
                        phase_label_wise_PTID[0][0].append(cur_PTID)

                    elif cur_research_group == 'AD':
                        phase_label_wise_imageID[0][1].append(cur_image_ID)
                        phase_label_wise_PTID[0][1].append(cur_PTID)

                    elif 'MCI' in cur_research_group:  ##TODO : MCI conversion
                        tmp_df = DXSUM_data[(DXSUM_data["RID"] == cur_RID)]
                        if tmp_df['DXCURREN'].empty:
                            # print('empty : {}'.format(cur_PTID))
                            count_empty_in_diag[0] += 1
                            pass
                        elif tmp_df['DXCURREN'].iloc[0] != 2:
                            # print('not matching : {}'.format(cur_PTID))
                            count_not_matching[0] += 1
                            pass
                        else:
                            ## TODO #1 no reversion
                            check_rev = False  # default : False
                            for i_tmp in range(tmp_df.shape[0]):
                                if i_tmp == 0:
                                    prev_DX = tmp_df['DXCURREN'].iloc[i_tmp]
                                else:
                                    if prev_DX == 3 and prev_DX > tmp_df['DXCURREN'].iloc[i_tmp]:
                                        # print('REV : {}'.format(cur_PTID))
                                        count_Rev_from_AD[0] += 1
                                        check_rev = True

                            ## TODO #2 At least one diagnosis is more than 36
                            check_more_than_36 = flag_stndard_MCI  # default : False
                            for i_tmp in range(tmp_df.shape[0]):
                                if tmp_df['VISCODE2'].iloc[i_tmp] in list_standard_MCI:
                                    check_more_than_36 = True

                            ##
                            if check_rev == False and check_more_than_36 == True:
                                tmp_df_2 = tmp_df[tmp_df['DXCURREN'] == 3]  # take only AD
                                if tmp_df_2['VISCODE2'].empty:
                                    # pass
                                    phase_label_wise_imageID[0][2].append(cur_image_ID)
                                    phase_label_wise_PTID[0][2].append(cur_PTID)

                                elif tmp_df_2['VISCODE2'].iloc[0] in list_standard_pMCI:  ##TODO : pMCI
                                    phase_label_wise_imageID[0][3].append(cur_image_ID)
                                    phase_label_wise_PTID[0][3].append(cur_PTID)
                                elif not any(
                                        tmp_df_2['VISCODE2'].iloc[0] in s for s in list_standard_sMCI):  ##TODO : sMCI
                                    phase_label_wise_imageID[0][2].append(cur_image_ID)
                                    phase_label_wise_PTID[0][2].append(cur_PTID)

                                else:  ##TODO : else
                                    tmp_counter_1 += 1
                                    pass
                                    # MCI_Conversion[0][0].append(cur_image_ID)
                                    # print('else')


                # elif (cur_phase == 'ADNI 2') and ('Year' not in cur_visit):
                # elif (cur_phase == 'ADNI 2') and ('ADNI2 Initial' in cur_visit or 'ADNI2 Screening' in cur_visit):
                elif (cur_phase == 'ADNI 2') and ('ADNI2 Screening' in cur_visit) and (cur_ImgProt[cur_ImgProt.index(
                        'Field Strength') + len('Field Strength') + 1: cur_ImgProt.index('Field Strength') + len(
                        'Field Strength') + 4] == '3.0'):

                    ## TODO : check if appear in 'ADNI 1'
                    tmp_df = data[(data['Subject ID'] == cur_PTID)]
                    tmp_df = tmp_df[(tmp_df['Phase'] == 'ADNI 1')]
                    if tmp_df.shape[0] != 0:
                        count_if_both_appear += 1
                    count_n_BL += 1

                    ID_img_label_wise_BL[
                        phase_type.tolist().index(cur_phase)][
                        label_type.tolist().index(cur_research_group)
                    ].append(cur_image_ID)

                    num_img_label_wise_BL[
                        phase_type.tolist().index(cur_phase)][
                        label_type.tolist().index(cur_research_group)
                    ] += 1
                    last_PTID = cur_PTID

                    if cur_research_group == 'CN':
                        phase_label_wise_imageID[1][0].append(cur_image_ID)
                        phase_label_wise_PTID[1][0].append(cur_PTID)

                    elif cur_research_group == 'AD':
                        phase_label_wise_imageID[1][1].append(cur_image_ID)
                        phase_label_wise_PTID[1][1].append(cur_PTID)

                    elif 'MCI' in cur_research_group:
                        tmp_df = DXSUM_data[(DXSUM_data["RID"] == cur_RID)][(DXSUM_data["Phase"] == 'ADNI2')]
                        if tmp_df['DXCHANGE'].empty:
                            # print('empty : {}'.format(cur_PTID))
                            count_empty_in_diag[1] += 1
                        elif tmp_df['DXCHANGE'].iloc[0] != 2:
                            # print('not matching : {}'.format(cur_PTID))
                            count_not_matching[1] += 1
                        else:
                            ## TODO #1 no reversion
                            check_rev = False  # default : False
                            for i_tmp in range(tmp_df.shape[0]):
                                if tmp_df['DXCHANGE'].iloc[i_tmp] == 8 or tmp_df['DXCHANGE'].iloc[i_tmp] == 9:
                                    count_Rev_from_AD[1] += 1
                                    print(cur_PTID)
                                    check_rev = True

                            ## TODO #2 At least one diagnosis is more than 36
                            check_more_than_36 = flag_stndard_MCI  # default : False
                            for i_tmp in range(tmp_df.shape[0]):
                                if tmp_df['VISCODE2'].iloc[i_tmp] in list_standard_MCI:
                                    check_more_than_36 = True

                            if check_rev == False and check_more_than_36 == True:
                                tmp_df_2 = tmp_df[(tmp_df['DXCHANGE'] == 3) | (tmp_df['DXCHANGE'] == 5)]  # take only AD
                                if tmp_df_2['VISCODE2'].empty:
                                    # pass
                                    phase_label_wise_imageID[1][2].append(cur_image_ID)
                                    phase_label_wise_PTID[1][2].append(cur_PTID)
                                elif tmp_df_2['VISCODE2'].iloc[0] in list_standard_pMCI:  ##TODO : pMCI
                                    phase_label_wise_imageID[1][3].append(cur_image_ID)
                                    phase_label_wise_PTID[1][3].append(cur_PTID)

                                elif not any(
                                        tmp_df_2['VISCODE2'].iloc[0] in s for s in list_standard_sMCI):  ##TODO : sMCI
                                    phase_label_wise_imageID[1][2].append(cur_image_ID)
                                    phase_label_wise_PTID[1][2].append(cur_PTID)

                                else:  ##TODO : else
                                    tmp_counter_2 += 1
                                    pass
                                    # MCI_Conversion[0][0].append(cur_image_ID)
                                    # print('else')

                else:  # not ADNI 1 or 2
                    pass
    # print(num_img_label_wise_BL)

    print("count_appear_in_both : {}".format(count_if_both_appear))
    print("ADNI1, NC : {}".format(len(phase_label_wise_imageID[0][0])))
    print("ADNI1, AD : {}".format(len(phase_label_wise_imageID[0][1])))
    print("ADNI2, NC : {}".format(len(phase_label_wise_imageID[1][0])))
    print("ADNI2, AD : {}".format(len(phase_label_wise_imageID[1][1])))
    print('------------------------------------------------------')
    print("ADNI1, sMCI : {}".format(len(phase_label_wise_imageID[0][2])))
    print("ADNI1, pMCI : {}".format(len(phase_label_wise_imageID[0][3])))
    print("ADNI2, sMCI : {}".format(len(phase_label_wise_imageID[1][2])))
    print("ADNI2, pMCI : {}".format(len(phase_label_wise_imageID[1][3])))

    print('empty : {}'.format(count_empty_in_diag))
    print('not matching : {}'.format(count_not_matching))
    print('Rev from AD : {}'.format(count_Rev_from_AD))

    """ plot image jsy processed """
    dir_img = '/DataCommon/jsyoon/data/smri_orig/preprocessed'
    # save_dir = './Data_check/AD_diagnosis'
    # ut.make_dir(dir=save_dir, flag_rm=False)
    # count_img = 0
    # phase_type_2 = ['ADNI1', 'ADNI2']
    # class_type_2 = ['CN', 'AD']
    # for i_phase in range(len(AD_diagnosis)):
    #     for i_class in range(len(AD_diagnosis[i_phase])):
    #
    #         for i_img in range(len(AD_diagnosis[i_phase][i_class])):
    #             img = nib.load(dir_img + '/{}_flirt_restore.nii.gz'.format(AD_diagnosis[i_phase][i_class][i_img])).get_fdata(dtype=np.float32)
    #             # img = np.expand_dims(img, axis = 0)
    #             count_img += 1
    #             ut.plot_heatmap_without_overlay(img, save_dir + '/img_{}'.format(count_img), fig_title='Img_{}_{}_{:.2f}_{:.2f}'.format(phase_type_2[i_phase], class_type_2[i_class], img.max(), img.min()), thresh=0.0, percentile=0)


    # save_dir = './Data_check/MCI_conversion'
    # ut.make_dir(dir=save_dir, flag_rm=False)
    # count_img = 0
    # phase_type_2 = ['ADNI1', 'ADNI2']
    # class_type_2 = ['sMCI', 'pMCI']
    # for i_phase in range(len(MCI_Conversion)):
    #     if i_phase == 1:
    #         for i_class in range(len(MCI_Conversion[i_phase])):
    #             for i_img in range(len(MCI_Conversion[i_phase][i_class])):
    #                 img = nib.load(dir_img + '/{}_flirt_restore.nii.gz'.format(MCI_Conversion[i_phase][i_class][i_img])).get_fdata(dtype=np.float32)
    #                 count_img += 1
    #                 ut.plot_heatmap_without_overlay(img, save_dir + '/img_{}'.format(count_img), fig_title='Img_{}_{}_{:.2f}_{:.2f}'.format(phase_type_2[i_phase], class_type_2[i_class], img.max(), img.min()), thresh=0.0, percentile=0)

    list_image_memalloc = []
    list_age_memalloc = []
    list_MMSE_memalloc = []

    list_image_memalloc_2 = []
    list_age_memalloc_2 = []
    list_MMSE_memalloc_2 = []

    """ ADNI 1"""
    for i in range (len(st.list_class_type)):
        list_image_memalloc.append(np.memmap(filename=st.ADNI_fold_image_path[i], mode="w+",
                                             shape=(len(phase_label_wise_imageID[0][i]) + len(phase_label_wise_imageID[1][i]),
                                                    st.num_modality, st.x_size, st.y_size, st.z_size), dtype=np.float32))
        list_age_memalloc.append(np.memmap(filename=st.ADNI_fold_age_path[i], mode="w+",
                                           shape=(len(phase_label_wise_imageID[0][i])+ len(phase_label_wise_imageID[1][i]), 1), dtype=np.float32))
        list_MMSE_memalloc.append(np.memmap(filename=st.ADNI_fold_MMSE_path[i], mode="w+",
                                            shape=(len(phase_label_wise_imageID[0][i])+ len(phase_label_wise_imageID[1][i]), 1), dtype=np.float32))

    # """ ADNI 2"""
    # for i in range (len(st.list_class_type)):
    #     list_image_memalloc_2.append(np.memmap(filename=st.ADNI_fold_image_path_2[i], mode="w+", shape=(len(phase_label_wise_imageID[1][i]), st.num_modality, st.x_size, st.y_size, st.z_size), dtype=np.float32))
    #     list_age_memalloc_2.append(np.memmap(filename=st.ADNI_fold_age_path_2[i], mode="w+", shape=(len(phase_label_wise_imageID[1][i]), 1), dtype=np.float32))
    #     list_MMSE_memalloc_2.append(np.memmap(filename=st.ADNI_fold_MMSE_path_2[i], mode="w+", shape=(len(phase_label_wise_imageID[1][i]), 1), dtype=np.float32))

    """ save the data """
    dir_img = '/DataCommon/jsyoon/data/smri_orig/preprocessed'
    for i_modality in range (1):
        for j_class in range (len(st.list_class_type)):
            for i_img in range(len(phase_label_wise_imageID[0][j_class])):
                print("modality: {}, class: {}, n_sample: {}".format(i_modality, j_class, i_img))
                tmp_img = nib.load(dir_img + '/{}_flirt_restore.nii.gz'.format(phase_label_wise_imageID[0][j_class][i_img])).get_fdata(dtype=np.float32)
                tmp_age = data[data['Image ID'] == phase_label_wise_imageID[0][j_class][i_img]]['Age'].iloc()[0]
                tmp_MMSE = data[data['Image ID'] == phase_label_wise_imageID[0][j_class][i_img]]['MMSE Total Score'].iloc()[0]

                list_image_memalloc[j_class][i_img, i_modality, :, :, :] = tmp_img
                list_age_memalloc[j_class][i_img, 0] = tmp_age
                list_MMSE_memalloc[j_class][i_img, 0] = tmp_MMSE

            for i_img in range(len(phase_label_wise_imageID[0][j_class]), len(phase_label_wise_imageID[0][j_class]) + len(phase_label_wise_imageID[1][j_class])):
                print("modality: {}, class: {}, n_sample: {}".format(i_modality, j_class, i_img))
                tmp_img = nib.load(dir_img + '/{}_flirt_restore.nii.gz'.format(phase_label_wise_imageID[1][j_class][i_img - len(phase_label_wise_imageID[0][j_class])])).get_fdata(dtype=np.float32)
                tmp_age = data[data['Image ID'] == phase_label_wise_imageID[1][j_class][i_img - len(phase_label_wise_imageID[0][j_class])]]['Age'].iloc()[0]
                tmp_MMSE = data[data['Image ID'] == phase_label_wise_imageID[1][j_class][i_img - len(phase_label_wise_imageID[0][j_class])]]['MMSE Total Score'].iloc()[0]

                list_image_memalloc[j_class][i_img, i_modality, :, :, :] = tmp_img
                list_age_memalloc[j_class][i_img, 0] = tmp_age
                list_MMSE_memalloc[j_class][i_img, 0] = tmp_MMSE



def Prepare_data_4():
    """ file smri_orig """
    smri_info_dir = '/DataCommon/chpark/ADNI_orig_JSY/info/smri_orig_info.csv'
    data = pd.read_csv(smri_info_dir)
    data = data.sort_values(by=['RID', 'Image ID'])

    """ info """
    n_img = (data['Image ID']).unique().shape[0]  # (21280, )
    label_type = (data['Research Group']).unique()  # ['CN', 'AD', 'MCI', 'EMCI', 'Patient', 'LMCI', 'SMC']

    """ if phase is nan, the phase is ADNI 2. """
    phase_type = data['Phase'].unique()  # [ANDI 1, ADNI GO, ADNI 2, ADNI 3, nan]
    n_phase_wo_nan = data.count()['Phase']  # (21274)
    data['Phase'].value_counts()  # 9317+9105+1653+1199
    for tmp_i in range(data['Phase'][:].isnull().shape[0]):
        if data['Phase'][tmp_i] not in phase_type:
            data['Phase'][tmp_i] = 'ADNI 2'

    """ sbject """
    n_sbj = (data['Subject ID']).unique().shape[0]  # (2428, )
    sbj_type = data['Subject ID'].unique()

    """ visit 1 """
    visit_type = (data['Visit 1']).unique()  # []
    count = 0
    for tmp_i in range(data['Visit 1'].unique().shape[0]):
        if 'ADNI2' in data['Visit 1'].unique()[tmp_i]:
            count += 1

    """ file diagnosis change """
    # diag_change_dir = '/DataCommon/chpark/ADNI_orig_JSY/info/diagnosis_change.csv'
    # diag_change_data = pd.read_csv(diag_change_dir)
    # diag_change_data = diag_change_data.sort_values(by=['RID', 'ImageID'])
    # potential_pMCI = diag_change_data['ImageID'].unique()

    """ file DXSUM """
    DXSUM_dir = '/DataCommon/chpark/ADNI_orig_JSY/info/DXSUM_PDXCONV_ADNIALL.csv'
    DXSUM_data = pd.read_csv(DXSUM_dir)
    DXSUM_data = DXSUM_data.sort_values(by=['RID', 'EXAMDATE'])

    """ start """
    """ start """
    """ start """
    flag_stndard_MCI = True  # When False, the standard would be applied
    # list_standard_MCI = ['m36', 'm48', 'm60', 'm72', 'm84', 'm96']
    list_standard_MCI = ['m36', 'm48', 'm60', 'm72']
    # list_standard_sMCI = ['m06', 'm12', 'm18', 'm24', 'm36', 'm48', 'm60', 'm72', 'm84', 'm96']
    list_standard_sMCI = ['m06', 'm12', 'm18', 'm24', 'm36', 'm48', 'm60', 'm72']
    # list_standard_sMCI = ['m06', 'm12', 'm18', 'm24', 'm36']
    list_standard_pMCI = ['m06', 'm12', 'm18', 'm24', 'm36']

    ## TODO : Without considering MPRAGE Repeat

    # ['CN', 'AD', 'MCI', 'EMCI', 'Patient', 'LMCI', 'SMC']
    # ADNI 1 : [[231, 200, 414, 0, 0, 0, 0],
    # ADNI GO : [0, 0, 0, 142, 5, 0, 0],
    # ADNI 2 : [202, 159, 0, 191, 0, 178, 111],
    # ADNI 3 : [332, 66, 193, 0, 4, 0, 0],
    # NaN : [0, 0, 0, 0, 0, 0, 0]]

    num_img_label_wise_BL = [np.zeros(label_type.shape) for _ in range(phase_type.shape[0])]  # (phase, label)
    ID_img_label_wise_BL = [[[] for _ in range(label_type.shape[0])] for _ in range(phase_type.shape[0])]  # (phase, label)
    phase_label_wise_imageID = [[[] for _ in range(4)] for _ in range(2)]
    phase_label_wise_PTID = [[[] for _ in range(4)] for _ in range(2)]
    last_PTID = ''
    cur_PTID = ''
    cur_phase = ''
    last_phase = ''
    count_n_BL = 0

    count_Rev_from_AD = np.zeros(2)
    count_empty_in_diag = np.zeros(2)
    count_not_matching = np.zeros(2)
    count_if_both_appear = 0

    list_test = []
    list_test_2 = []
    # excluded_year = ['2019', '2020']
    excluded_year = []
    tmp_counter_1 = 0
    tmp_counter_2 = 0
    for i in range(n_img):
        cur_image_ID = data['Image ID'][i]
        cur_PTID = data['Subject ID'][i]  # subject id
        cur_RID = data['RID'][i]
        cur_phase = data['Phase'][i]
        cur_research_group = data['Research Group'][i]
        cur_visit = data['Visit 1'][i]
        cur_StudyDate = data['Study Date'][i]
        cur_ImgProt = data['Imaging Protocol'][i]
        if last_PTID == cur_PTID:  # same subject
            pass

        else:
            if not any(s in cur_StudyDate for s in excluded_year):
                if (cur_phase == 'ADNI 1') and ('Screening' in cur_visit) and (cur_ImgProt[cur_ImgProt.index(
                        'Field Strength') + len('Field Strength') + 1: cur_ImgProt.index('Field Strength') + len(
                        'Field Strength') + 4] == '1.5'):
                    # if (cur_phase == 'ADNI 1') and (cur_visit == 'ADNI Baseline' or cur_visit == 'ADNI Screening'):
                    # if cur_visit == 'ADNI Baseline':
                    count_n_BL += 1

                    ID_img_label_wise_BL[
                        phase_type.tolist().index(cur_phase)][
                        label_type.tolist().index(cur_research_group)
                    ].append(cur_image_ID)

                    num_img_label_wise_BL[
                        phase_type.tolist().index(cur_phase)][
                        label_type.tolist().index(cur_research_group)
                    ] += 1
                    last_PTID = cur_PTID

                    if cur_research_group == 'CN':
                        phase_label_wise_imageID[0][0].append(cur_image_ID)
                        phase_label_wise_PTID[0][0].append(cur_PTID)

                    elif cur_research_group == 'AD':
                        phase_label_wise_imageID[0][1].append(cur_image_ID)
                        phase_label_wise_PTID[0][1].append(cur_PTID)

                    elif 'MCI' in cur_research_group:  ##TODO : MCI conversion
                        tmp_df = DXSUM_data[(DXSUM_data["RID"] == cur_RID)]
                        if tmp_df['DXCURREN'].empty:
                            # print('empty : {}'.format(cur_PTID))
                            count_empty_in_diag[0] += 1
                            pass
                        elif tmp_df['DXCURREN'].iloc[0] != 2:
                            # print('not matching : {}'.format(cur_PTID))
                            count_not_matching[0] += 1
                            pass
                        else:
                            ## TODO #1 no reversion
                            check_rev = False  # default : False
                            for i_tmp in range(tmp_df.shape[0]):
                                if i_tmp == 0:
                                    prev_DX = tmp_df['DXCURREN'].iloc[i_tmp]
                                else:
                                    if prev_DX == 3 and prev_DX > tmp_df['DXCURREN'].iloc[i_tmp]:
                                        # print('REV : {}'.format(cur_PTID))
                                        count_Rev_from_AD[0] += 1
                                        check_rev = True

                            ## TODO #2 At least one diagnosis is more than 36
                            check_more_than_36 = flag_stndard_MCI  # default : False
                            for i_tmp in range(tmp_df.shape[0]):
                                if tmp_df['VISCODE2'].iloc[i_tmp] in list_standard_MCI:
                                    check_more_than_36 = True

                            ##
                            if check_rev == False and check_more_than_36 == True:
                                tmp_df_2 = tmp_df[tmp_df['DXCURREN'] == 3]  # take only AD
                                if tmp_df_2['VISCODE2'].empty:
                                    # pass
                                    phase_label_wise_imageID[0][2].append(cur_image_ID)
                                    phase_label_wise_PTID[0][2].append(cur_PTID)

                                elif tmp_df_2['VISCODE2'].iloc[0] in list_standard_pMCI:  ##TODO : pMCI
                                    phase_label_wise_imageID[0][3].append(cur_image_ID)
                                    phase_label_wise_PTID[0][3].append(cur_PTID)
                                elif not any(
                                        tmp_df_2['VISCODE2'].iloc[0] in s for s in list_standard_sMCI):  ##TODO : sMCI
                                    phase_label_wise_imageID[0][2].append(cur_image_ID)
                                    phase_label_wise_PTID[0][2].append(cur_PTID)

                                else:  ##TODO : else
                                    tmp_counter_1 += 1
                                    pass
                                    # MCI_Conversion[0][0].append(cur_image_ID)
                                    # print('else')


                # elif (cur_phase == 'ADNI 2') and ('Year' not in cur_visit):
                # elif (cur_phase == 'ADNI 2') and ('ADNI2 Initial' in cur_visit or 'ADNI2 Screening' in cur_visit):
                elif (cur_phase == 'ADNI 2') and ('ADNI2 Screening' in cur_visit) and (cur_ImgProt[cur_ImgProt.index(
                        'Field Strength') + len('Field Strength') + 1: cur_ImgProt.index('Field Strength') + len(
                        'Field Strength') + 4] == '3.0'):

                    ## TODO : check if appear in 'ADNI 1'
                    tmp_df = data[(data['Subject ID'] == cur_PTID)]
                    tmp_df = tmp_df[(tmp_df['Phase'] == 'ADNI 1')]
                    if tmp_df.shape[0] != 0:
                        count_if_both_appear += 1
                    count_n_BL += 1

                    ID_img_label_wise_BL[
                        phase_type.tolist().index(cur_phase)][
                        label_type.tolist().index(cur_research_group)
                    ].append(cur_image_ID)

                    num_img_label_wise_BL[
                        phase_type.tolist().index(cur_phase)][
                        label_type.tolist().index(cur_research_group)
                    ] += 1
                    last_PTID = cur_PTID

                    if cur_research_group == 'CN':
                        phase_label_wise_imageID[1][0].append(cur_image_ID)
                        phase_label_wise_PTID[1][0].append(cur_PTID)

                    elif cur_research_group == 'AD':
                        phase_label_wise_imageID[1][1].append(cur_image_ID)
                        phase_label_wise_PTID[1][1].append(cur_PTID)

                    elif 'MCI' in cur_research_group:
                        tmp_df = DXSUM_data[(DXSUM_data["RID"] == cur_RID)][(DXSUM_data["Phase"] == 'ADNI2')]
                        if tmp_df['DXCHANGE'].empty:
                            # print('empty : {}'.format(cur_PTID))
                            count_empty_in_diag[1] += 1
                        elif tmp_df['DXCHANGE'].iloc[0] != 2:
                            # print('not matching : {}'.format(cur_PTID))
                            count_not_matching[1] += 1
                        else:
                            ## TODO #1 no reversion
                            check_rev = False  # default : False
                            for i_tmp in range(tmp_df.shape[0]):
                                if tmp_df['DXCHANGE'].iloc[i_tmp] == 8 or tmp_df['DXCHANGE'].iloc[i_tmp] == 9:
                                    count_Rev_from_AD[1] += 1
                                    print(cur_PTID)
                                    check_rev = True

                            ## TODO #2 At least one diagnosis is more than 36
                            check_more_than_36 = flag_stndard_MCI  # default : False
                            for i_tmp in range(tmp_df.shape[0]):
                                if tmp_df['VISCODE2'].iloc[i_tmp] in list_standard_MCI:
                                    check_more_than_36 = True

                            if check_rev == False and check_more_than_36 == True:
                                tmp_df_2 = tmp_df[(tmp_df['DXCHANGE'] == 3) | (tmp_df['DXCHANGE'] == 5)]  # take only AD
                                if tmp_df_2['VISCODE2'].empty:
                                    # pass
                                    phase_label_wise_imageID[1][2].append(cur_image_ID)
                                    phase_label_wise_PTID[1][2].append(cur_PTID)
                                elif tmp_df_2['VISCODE2'].iloc[0] in list_standard_pMCI:  ##TODO : pMCI
                                    phase_label_wise_imageID[1][3].append(cur_image_ID)
                                    phase_label_wise_PTID[1][3].append(cur_PTID)

                                elif not any(
                                        tmp_df_2['VISCODE2'].iloc[0] in s for s in list_standard_sMCI):  ##TODO : sMCI
                                    phase_label_wise_imageID[1][2].append(cur_image_ID)
                                    phase_label_wise_PTID[1][2].append(cur_PTID)

                                else:  ##TODO : else
                                    tmp_counter_2 += 1
                                    pass
                                    # MCI_Conversion[0][0].append(cur_image_ID)
                                    # print('else')

                else:  # not ADNI 1 or 2
                    pass
    # print(num_img_label_wise_BL)

    print("count_appear_in_both : {}".format(count_if_both_appear))
    print("ADNI1, NC : {}".format(len(phase_label_wise_imageID[0][0])))
    print("ADNI1, AD : {}".format(len(phase_label_wise_imageID[0][1])))
    print("ADNI2, NC : {}".format(len(phase_label_wise_imageID[1][0])))
    print("ADNI2, AD : {}".format(len(phase_label_wise_imageID[1][1])))
    print('------------------------------------------------------')
    print("ADNI1, sMCI : {}".format(len(phase_label_wise_imageID[0][2])))
    print("ADNI1, pMCI : {}".format(len(phase_label_wise_imageID[0][3])))
    print("ADNI2, sMCI : {}".format(len(phase_label_wise_imageID[1][2])))
    print("ADNI2, pMCI : {}".format(len(phase_label_wise_imageID[1][3])))

    print('empty : {}'.format(count_empty_in_diag))
    print('not matching : {}'.format(count_not_matching))
    print('Rev from AD : {}'.format(count_Rev_from_AD))

    """ plot image jsy processed """
    dir_img = '/DataCommon/jsyoon/data/smri_orig/preprocessed'
    # save_dir = './Data_check/AD_diagnosis'
    # ut.make_dir(dir=save_dir, flag_rm=False)
    # count_img = 0
    # phase_type_2 = ['ADNI1', 'ADNI2']
    # class_type_2 = ['CN', 'AD']
    # for i_phase in range(len(AD_diagnosis)):
    #     for i_class in range(len(AD_diagnosis[i_phase])):
    #
    #         for i_img in range(len(AD_diagnosis[i_phase][i_class])):
    #             img = nib.load(dir_img + '/{}_flirt_restore.nii.gz'.format(AD_diagnosis[i_phase][i_class][i_img])).get_fdata(dtype=np.float32)
    #             # img = np.expand_dims(img, axis = 0)
    #             count_img += 1
    #             ut.plot_heatmap_without_overlay(img, save_dir + '/img_{}'.format(count_img), fig_title='Img_{}_{}_{:.2f}_{:.2f}'.format(phase_type_2[i_phase], class_type_2[i_class], img.max(), img.min()), thresh=0.0, percentile=0)


    # save_dir = './Data_check/MCI_conversion'
    # ut.make_dir(dir=save_dir, flag_rm=False)
    # count_img = 0
    # phase_type_2 = ['ADNI1', 'ADNI2']
    # class_type_2 = ['sMCI', 'pMCI']
    # for i_phase in range(len(MCI_Conversion)):
    #     if i_phase == 1:
    #         for i_class in range(len(MCI_Conversion[i_phase])):
    #             for i_img in range(len(MCI_Conversion[i_phase][i_class])):
    #                 img = nib.load(dir_img + '/{}_flirt_restore.nii.gz'.format(MCI_Conversion[i_phase][i_class][i_img])).get_fdata(dtype=np.float32)
    #                 count_img += 1
    #                 ut.plot_heatmap_without_overlay(img, save_dir + '/img_{}'.format(count_img), fig_title='Img_{}_{}_{:.2f}_{:.2f}'.format(phase_type_2[i_phase], class_type_2[i_class], img.max(), img.min()), thresh=0.0, percentile=0)

    list_image_memalloc = []
    list_age_memalloc = []
    list_MMSE_memalloc = []

    list_image_memalloc_2 = []
    list_age_memalloc_2 = []
    list_MMSE_memalloc_2 = []

    """ ADNI 1"""
    for i in range (len(st.list_class_type)):
        list_image_memalloc.append(np.memmap(filename=st.ADNI_fold_image_path[i], mode="w+",
                                             shape=(len(phase_label_wise_imageID[0][i]) + len(phase_label_wise_imageID[1][i]),
                                                    st.num_modality, st.x_size, st.y_size, st.z_size), dtype=np.float32))
        list_age_memalloc.append(np.memmap(filename=st.ADNI_fold_age_path[i], mode="w+",
                                           shape=(len(phase_label_wise_imageID[0][i])+ len(phase_label_wise_imageID[1][i]), 1), dtype=np.float32))
        list_MMSE_memalloc.append(np.memmap(filename=st.ADNI_fold_MMSE_path[i], mode="w+",
                                            shape=(len(phase_label_wise_imageID[0][i])+ len(phase_label_wise_imageID[1][i]), 1), dtype=np.float32))

    # """ ADNI 2"""
    # for i in range (len(st.list_class_type)):
    #     list_image_memalloc_2.append(np.memmap(filename=st.ADNI_fold_image_path_2[i], mode="w+", shape=(len(phase_label_wise_imageID[1][i]), st.num_modality, st.x_size, st.y_size, st.z_size), dtype=np.float32))
    #     list_age_memalloc_2.append(np.memmap(filename=st.ADNI_fold_age_path_2[i], mode="w+", shape=(len(phase_label_wise_imageID[1][i]), 1), dtype=np.float32))
    #     list_MMSE_memalloc_2.append(np.memmap(filename=st.ADNI_fold_MMSE_path_2[i], mode="w+", shape=(len(phase_label_wise_imageID[1][i]), 1), dtype=np.float32))

    """ save the data """
    dir_img = '/DataCommon/jsyoon/data/smri_orig/preprocessed'
    for i_modality in range (1):
        for j_class in range (len(st.list_class_type)):
            for i_img in range(len(phase_label_wise_imageID[0][j_class])):
                print("modality: {}, class: {}, n_sample: {}".format(i_modality, j_class, i_img))
                tmp_img = nib.load(dir_img + '/{}_flirt_restore.nii.gz'.format(phase_label_wise_imageID[0][j_class][i_img])).get_fdata(dtype=np.float32)
                tmp_age = data[data['Image ID'] == phase_label_wise_imageID[0][j_class][i_img]]['Age'].iloc()[0]
                tmp_MMSE = data[data['Image ID'] == phase_label_wise_imageID[0][j_class][i_img]]['MMSE Total Score'].iloc()[0]

                list_image_memalloc[j_class][i_img, i_modality, :, :, :] = tmp_img
                list_age_memalloc[j_class][i_img, 0] = tmp_age
                list_MMSE_memalloc[j_class][i_img, 0] = tmp_MMSE

            for i_img in range(len(phase_label_wise_imageID[0][j_class]), len(phase_label_wise_imageID[0][j_class]) + len(phase_label_wise_imageID[1][j_class])):
                print("modality: {}, class: {}, n_sample: {}".format(i_modality, j_class, i_img))
                tmp_img = nib.load(dir_img + '/{}_flirt_restore.nii.gz'.format(phase_label_wise_imageID[1][j_class][i_img - len(phase_label_wise_imageID[0][j_class])])).get_fdata(dtype=np.float32)
                tmp_age = data[data['Image ID'] == phase_label_wise_imageID[1][j_class][i_img - len(phase_label_wise_imageID[0][j_class])]]['Age'].iloc()[0]
                tmp_MMSE = data[data['Image ID'] == phase_label_wise_imageID[1][j_class][i_img - len(phase_label_wise_imageID[0][j_class])]]['MMSE Total Score'].iloc()[0]

                list_image_memalloc[j_class][i_img, i_modality, :, :, :] = tmp_img
                list_age_memalloc[j_class][i_img, 0] = tmp_age
                list_MMSE_memalloc[j_class][i_img, 0] = tmp_MMSE
