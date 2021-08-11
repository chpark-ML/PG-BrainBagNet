import torch.utils.data as data
import numpy as np
import torch
import setting as st
import pickle
from torch.utils.data import Sampler, Dataset, DataLoader, BatchSampler, SequentialSampler, RandomSampler, Subset
import os

# x_range = st.x_range
# y_range = st.y_range
# z_range = st.z_range
# x_size = x_range[1] - x_range[0]
# y_size = y_range[1] - y_range[0]
# z_size = z_range[1] - z_range[0]


class Dataset(data.Dataset):
    def __init__(self, Data_name, cLabel, is_training = True):
        super(Dataset, self).__init__()
        self.data = Data_name
        self.cLabel = cLabel

    def __getitem__(self,idx):
        item = torch.from_numpy(self.data[idx, ...]).float(), torch.from_numpy(self.cLabel[idx, ...])
        # item = self.transform(item)
        return item

    def __len__(self):
        return self.data.shape[0]

    # transfrom = transforms.Compose([
    #     transforms.RandomHorizontalFlip()
    # ])
def convert_Dloader(batch_size, data, label, is_training = False, num_workers = 1, shuffle = True):
        dataset = Dataset(data, label, is_training = is_training)
        # dataset = datasets.
        Data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  num_workers=num_workers,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  drop_last=False
                                                  )
        return Data_loader

class Dataset_2(data.Dataset):
    def __init__(self, Data_name, cLabel, aLabel, is_training = True):
        super(Dataset_2, self).__init__()
        self.data = Data_name
        self.cLabel = cLabel
        self.aLabel = aLabel

    def __getitem__(self,idx):
        item = torch.from_numpy(self.data[idx, ...]).float(), torch.from_numpy(self.cLabel[idx, ...]), torch.from_numpy(self.aLabel[idx, ...]).float()
        # item = self.transform(item)
        return item

    def __len__(self):
        return self.data.shape[0]
def convert_Dloader_2(batch_size, data, label, age, is_training = False, num_workers = 1, shuffle = True):
        dataset = Dataset_2(data, label, age, is_training = is_training)
        Data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  num_workers=num_workers,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  drop_last=False
                                                  )
        return Data_loader


class Dataset_3(data.Dataset):
    def __init__(self, fold, list_class, flag_tr_val_te='train'):
        super(Dataset_3, self).__init__()

        """ data type """
        dtype = np.float32
        dtype_2 = np.float32


        """ fold index """
        list_trIdx = []  # (class, fold, length)
        list_valIdx = []  # (class, fold, length)
        list_teIdx = []  # (class, fold, length)
        for i_class_type in range(len(st.list_class_type)):
            with open(st.train_index_dir[i_class_type], 'rb') as fp:
                list_trIdx.append(pickle.load(fp))
            with open(st.val_index_dir[i_class_type], 'rb') as fp:
                list_valIdx.append(pickle.load(fp))
            with open(st.test_index_dir[i_class_type], 'rb') as fp:
                list_teIdx.append(pickle.load(fp))

        """ all data """
        fold_index = fold - 1
        list_image = []  # (class, (sample, 1, x, y, z))
        list_age = []  # (class, sample)
        list_MMSE = []  # (class, sample)
        list_lbl = []  # (class, sample)

        n_class = 0
        for i_class_type in range(len(list_class)):
            if list_class[i_class_type] == 1:
                n_class += 1

        count_class = 0
        for i_class_type in range(len(list_class)):
            if list_class[i_class_type] == 1:
                list_image.append(
                    np.memmap(filename=st.ADNI_fold_image_path[i_class_type], mode="r", dtype=dtype).reshape(-1, st.num_modality, st.x_size, st.y_size, st.z_size))
                list_age.append(np.memmap(filename=st.ADNI_fold_age_path[i_class_type], mode="r", dtype=dtype_2))
                list_MMSE.append(np.memmap(filename=st.ADNI_fold_MMSE_path[i_class_type], mode="r", dtype=dtype_2))
                if n_class == 4:
                    value = count_class // (n_class / 2)
                    print('value : {}'.format(value))
                    list_lbl.append(np.full(shape=(len(list_image[-1])), fill_value=value, dtype=np.uint8))
                else:
                    list_lbl.append(np.full(shape=(len(list_image[-1])), fill_value=count_class, dtype=np.uint8))
                count_class += 1

        """ Fold """
        self.dataset = []
        self.n_sample = []
        count_class = 0
        for i_class_type in range(len(list_class)):  # (0, 1, 2, 3, 4)
            if list_class[i_class_type] == 1:
                print("disease_class : {}".format(i_class_type))
                if flag_tr_val_te == 'train':
                    print('train')
                    print(len(list_trIdx[i_class_type][fold_index]))
                    self.n_sample.append(len(list_trIdx[i_class_type][fold_index]))
                    self.dataset += [
                        {
                        'data': torch.from_numpy(list_image[count_class][tmp_i]),
                        'label': torch.from_numpy(np.array(list_lbl[count_class][tmp_i])),
                        'age': torch.from_numpy(np.array(list_age[count_class][tmp_i])),
                        'MMSE': torch.from_numpy(np.array(list_MMSE[count_class][tmp_i]))
                        }
                        for tmp_i in (list_trIdx[i_class_type][fold_index])]

                elif flag_tr_val_te == 'val':
                    print('val')
                    print(len(list_valIdx[i_class_type][fold_index]))
                    self.n_sample.append(len(list_valIdx[i_class_type][fold_index]))
                    self.dataset += [{'data': torch.from_numpy(list_image[count_class][tmp_i]),
                                      'label': torch.from_numpy(np.array(list_lbl[count_class][tmp_i])),
                                      'age': torch.from_numpy(np.array(list_age[count_class][tmp_i])),
                                      'MMSE': torch.from_numpy(np.array(list_MMSE[count_class][tmp_i]))}
                                     for tmp_i in (list_valIdx[i_class_type][fold_index])]
                elif flag_tr_val_te == 'test':
                    print('test')
                    print(len(list_teIdx[i_class_type][fold_index]))
                    self.n_sample.append(len(list_teIdx[i_class_type][fold_index]))
                    self.dataset += [{'data': torch.from_numpy(list_image[count_class][tmp_i]),
                                      'label': torch.from_numpy(np.array(list_lbl[count_class][tmp_i])),
                                      'age': torch.from_numpy(np.array(list_age[count_class][tmp_i])),
                                      'MMSE': torch.from_numpy(np.array(list_MMSE[count_class][tmp_i]))}
                                     for tmp_i in (list_teIdx[i_class_type][fold_index])]
                count_class += 1
        print('finished')
    def __getitem__(self,idx):
        item = self.dataset[idx]

        # item = self.transform(item)
        return item

    def __len__(self):
        return len(self.dataset)


class Dataset_4(data.Dataset):
    def __init__(self, fold, list_class, flag_tr_val_te='train'):
        super(Dataset_4, self).__init__()

        """ data type """
        dtype = np.float32
        dtype_2 = np.float32


        """ fold index """
        list_trIdx = []  # (class, fold, length)
        list_valIdx = []  # (class, fold, length)
        list_teIdx = []  # (class, fold, length)

        if st.list_eval_type[st.num_eval_choise] == '1_to_2':
            for i_class_type in range(len(st.list_class_type)):
                with open(st.train_index_dir[i_class_type], 'rb') as fp:
                    list_trIdx.append(pickle.load(fp))
                with open(st.val_index_dir[i_class_type], 'rb') as fp:
                    list_valIdx.append(pickle.load(fp))
                with open(st.test_index_dir[i_class_type], 'rb') as fp:
                    list_teIdx.append(pickle.load(fp))

        elif st.list_eval_type[st.num_eval_choise] == '2_to_1':
            for i_class_type in range(len(st.list_class_type)):
                with open(st.train_index_dir_2[i_class_type], 'rb') as fp:
                    list_trIdx.append(pickle.load(fp))
                with open(st.val_index_dir_2[i_class_type], 'rb') as fp:
                    list_valIdx.append(pickle.load(fp))
                with open(st.test_index_dir_2[i_class_type], 'rb') as fp:
                    list_teIdx.append(pickle.load(fp))

        """ all data """
        fold_index = fold - 1
        list_image = []  # (class, (sample, 1, x, y, z))
        list_age = []  # (class, sample)
        list_MMSE = []  # (class, sample)
        list_lbl = []  # (class, sample)

        list_image_2 = []  # (class, (sample, 1, x, y, z))
        list_age_2 = []  # (class, sample)
        list_MMSE_2 = []  # (class, sample)
        list_lbl_2 = []  # (class, sample)

        n_class = 0
        for i_class_type in range(len(list_class)):
            if list_class[i_class_type] == 1:
                n_class += 1


        if st.list_eval_type[st.num_eval_choise] == '1_to_2':
            count_class = 0
            for i_class_type in range(len(list_class)):
                if list_class[i_class_type] == 1:
                    list_image.append(
                        np.memmap(filename=st.ADNI_fold_image_path[i_class_type], mode="r", dtype=dtype).reshape(-1, st.num_modality, st.x_size, st.y_size, st.z_size))
                    list_age.append(np.memmap(filename=st.ADNI_fold_age_path[i_class_type], mode="r", dtype=dtype_2))
                    list_MMSE.append(np.memmap(filename=st.ADNI_fold_MMSE_path[i_class_type], mode="r", dtype=dtype_2))
                    list_lbl.append(np.full(shape=(len(list_image[-1])), fill_value=count_class, dtype=np.uint8))
                    count_class += 1

            count_class = 0
            for i_class_type in range(len(list_class)):
                if list_class[i_class_type] == 1:
                    list_image_2.append(
                        np.memmap(filename=st.ADNI_fold_image_path_2[i_class_type], mode="r", dtype=dtype).reshape(-1, st.num_modality, st.x_size, st.y_size, st.z_size))
                    list_age_2.append(np.memmap(filename=st.ADNI_fold_age_path_2[i_class_type], mode="r", dtype=dtype_2))
                    list_MMSE_2.append(np.memmap(filename=st.ADNI_fold_MMSE_path_2[i_class_type], mode="r", dtype=dtype_2))
                    list_lbl_2.append(np.full(shape=(len(list_image_2[-1])), fill_value=count_class, dtype=np.uint8))
                    count_class += 1


        elif st.list_eval_type[st.num_eval_choise] == '2_to_1':
            count_class = 0
            for i_class_type in range(len(list_class)):
                if list_class[i_class_type] == 1:
                    list_image.append(
                        np.memmap(filename=st.ADNI_fold_image_path_2[i_class_type], mode="r", dtype=dtype).reshape(-1, st.num_modality, st.x_size, st.y_size, st.z_size))
                    list_age.append(np.memmap(filename=st.ADNI_fold_age_path_2[i_class_type], mode="r", dtype=dtype_2))
                    list_MMSE.append(np.memmap(filename=st.ADNI_fold_MMSE_path_2[i_class_type], mode="r", dtype=dtype_2))
                    list_lbl.append(np.full(shape=(len(list_image[-1])), fill_value=count_class, dtype=np.uint8))
                    count_class += 1

            count_class = 0
            for i_class_type in range(len(list_class)):
                if list_class[i_class_type] == 1:
                    list_image_2.append(
                        np.memmap(filename=st.ADNI_fold_image_path[i_class_type], mode="r", dtype=dtype).reshape(-1, st.num_modality, st.x_size, st.y_size, st.z_size))
                    list_age_2.append(np.memmap(filename=st.ADNI_fold_age_path[i_class_type], mode="r", dtype=dtype_2))
                    list_MMSE_2.append(np.memmap(filename=st.ADNI_fold_MMSE_path[i_class_type], mode="r", dtype=dtype_2))
                    list_lbl_2.append(np.full(shape=(len(list_image_2[-1])), fill_value=count_class, dtype=np.uint8))
                    count_class += 1



        """ Fold """
        self.dataset = []
        count_class = 0
        for i_class_type in range(len(list_class)):  # (0, 1, 2, 3, 4)
            if list_class[i_class_type] == 1:
                print("disease_class : {}".format(i_class_type))
                if flag_tr_val_te == 'train':
                    print('train')
                    self.dataset += [
                        {
                        'data': (torch.from_numpy(list_image[count_class][tmp_i])),
                        'label': torch.from_numpy(np.array(list_lbl[count_class][tmp_i])),
                        'age': torch.from_numpy(np.array(list_age[count_class][tmp_i])),
                        'MMSE': torch.from_numpy(np.array(list_MMSE[count_class][tmp_i]))
                        }
                        for tmp_i in (list_trIdx[i_class_type][fold_index])]

                elif flag_tr_val_te == 'val':
                    print('val')

                    self.dataset += [
                        {
                        'data': (torch.from_numpy(list_image[count_class][tmp_i])),
                        'label': torch.from_numpy(np.array(list_lbl[count_class][tmp_i])),
                        'age': torch.from_numpy(np.array(list_age[count_class][tmp_i])),
                        'MMSE': torch.from_numpy(np.array(list_MMSE[count_class][tmp_i]))
                        }
                        for tmp_i in (list_valIdx[i_class_type][fold_index])]

                elif flag_tr_val_te == 'test':
                    print('test')

                    self.dataset += [{'data': (torch.from_numpy(list_image_2[count_class][tmp_i])),
                                      'label': torch.from_numpy(np.array(list_lbl_2[count_class][tmp_i])),
                                      'age': torch.from_numpy(np.array(list_age_2[count_class][tmp_i])),
                                      'MMSE': torch.from_numpy(np.array(list_MMSE_2[count_class][tmp_i]))}
                                     for tmp_i in (list_teIdx[i_class_type][fold_index])]

                count_class += 1
        print('finished')


    def __getitem__(self,idx):
        item = self.dataset[idx]

        # item = self.transform(item)
        return item

    def __len__(self):
        return len(self.dataset)

def Extract_mean_std_in_trainData(fold, list_class, flag_tr_val_te='train'):
    """ data type """
    dtype = np.float32
    dtype_2 = np.float32

    """ fold index """
    list_trIdx = []  # (class, fold, length)
    list_valIdx = []  # (class, fold, length)
    list_teIdx = []  # (class, fold, length)

    if st.list_eval_type[st.num_eval_choise] == '1_to_2':
        for i_class_type in range(len(st.list_class_type)):
            with open(st.train_index_dir[i_class_type], 'rb') as fp:
                list_trIdx.append(pickle.load(fp))
            with open(st.val_index_dir[i_class_type], 'rb') as fp:
                list_valIdx.append(pickle.load(fp))
            with open(st.test_index_dir[i_class_type], 'rb') as fp:
                list_teIdx.append(pickle.load(fp))

    elif st.list_eval_type[st.num_eval_choise] == '2_to_1':
        for i_class_type in range(len(st.list_class_type)):
            with open(st.train_index_dir_2[i_class_type], 'rb') as fp:
                list_trIdx.append(pickle.load(fp))
            with open(st.val_index_dir_2[i_class_type], 'rb') as fp:
                list_valIdx.append(pickle.load(fp))
            with open(st.test_index_dir_2[i_class_type], 'rb') as fp:
                list_teIdx.append(pickle.load(fp))

    """ all data """
    fold_index = fold - 1
    list_image = []  # (class, (sample, 1, x, y, z))
    list_age = []  # (class, sample)
    list_MMSE = []  # (class, sample)
    list_lbl = []  # (class, sample)

    list_image_2 = []  # (class, (sample, 1, x, y, z))
    list_age_2 = []  # (class, sample)
    list_MMSE_2 = []  # (class, sample)
    list_lbl_2 = []  # (class, sample)

    n_class = 0
    for i_class_type in range(len(list_class)):
        if list_class[i_class_type] == 1:
            n_class += 1

    if st.list_eval_type[st.num_eval_choise] == '1_to_2':
        count_class = 0
        for i_class_type in range(len(list_class)):
            if list_class[i_class_type] == 1:
                list_image.append(
                    np.memmap(filename=st.ADNI_fold_image_path[i_class_type], mode="r", dtype=dtype).reshape(-1,
                                                                                                             st.num_modality,
                                                                                                             st.x_size,
                                                                                                             st.y_size,
                                                                                                             st.z_size))
                list_age.append(np.memmap(filename=st.ADNI_fold_age_path[i_class_type], mode="r", dtype=dtype_2))
                list_MMSE.append(np.memmap(filename=st.ADNI_fold_MMSE_path[i_class_type], mode="r", dtype=dtype_2))
                list_lbl.append(np.full(shape=(len(list_image[-1])), fill_value=count_class, dtype=np.uint8))
                count_class += 1

        count_class = 0
        for i_class_type in range(len(list_class)):
            if list_class[i_class_type] == 1:
                list_image_2.append(
                    np.memmap(filename=st.ADNI_fold_image_path_2[i_class_type], mode="r", dtype=dtype).reshape(-1,
                                                                                                               st.num_modality,
                                                                                                               st.x_size,
                                                                                                               st.y_size,
                                                                                                               st.z_size))
                list_age_2.append(np.memmap(filename=st.ADNI_fold_age_path_2[i_class_type], mode="r", dtype=dtype_2))
                list_MMSE_2.append(np.memmap(filename=st.ADNI_fold_MMSE_path_2[i_class_type], mode="r", dtype=dtype_2))
                list_lbl_2.append(np.full(shape=(len(list_image_2[-1])), fill_value=count_class, dtype=np.uint8))
                count_class += 1


    elif st.list_eval_type[st.num_eval_choise] == '2_to_1':
        count_class = 0
        for i_class_type in range(len(list_class)):
            if list_class[i_class_type] == 1:
                list_image.append(
                    np.memmap(filename=st.ADNI_fold_image_path_2[i_class_type], mode="r", dtype=dtype).reshape(-1,
                                                                                                               st.num_modality,
                                                                                                               st.x_size,
                                                                                                               st.y_size,
                                                                                                               st.z_size))
                list_age.append(np.memmap(filename=st.ADNI_fold_age_path_2[i_class_type], mode="r", dtype=dtype_2))
                list_MMSE.append(np.memmap(filename=st.ADNI_fold_MMSE_path_2[i_class_type], mode="r", dtype=dtype_2))
                list_lbl.append(np.full(shape=(len(list_image[-1])), fill_value=count_class, dtype=np.uint8))
                count_class += 1

        count_class = 0
        for i_class_type in range(len(list_class)):
            if list_class[i_class_type] == 1:
                list_image_2.append(
                    np.memmap(filename=st.ADNI_fold_image_path[i_class_type], mode="r", dtype=dtype).reshape(-1,
                                                                                                             st.num_modality,
                                                                                                             st.x_size,
                                                                                                             st.y_size,
                                                                                                             st.z_size))
                list_age_2.append(np.memmap(filename=st.ADNI_fold_age_path[i_class_type], mode="r", dtype=dtype_2))
                list_MMSE_2.append(np.memmap(filename=st.ADNI_fold_MMSE_path[i_class_type], mode="r", dtype=dtype_2))
                list_lbl_2.append(np.full(shape=(len(list_image_2[-1])), fill_value=count_class, dtype=np.uint8))
                count_class += 1

    """ normalize data """
    print('1')
    train_data_for_catching_distribution = []
    count_class = 0
    for i_class_type in range(len(list_class)):  # (0, 1, 2, 3, 4)
        if list_class[i_class_type] == 1:
            for tmp_i in (list_trIdx[i_class_type][fold_index]):
                train_data_for_catching_distribution.append((list_image[count_class][tmp_i]))
            count_class += 1
    print('2')
    train_data_for_norm = np.empty((len(train_data_for_catching_distribution), st.x_size, st.y_size, st.z_size), dtype=np.float32)
    for i_tmp in range(len(train_data_for_catching_distribution)):
        train_data_for_norm[i_tmp] = train_data_for_catching_distribution[i_tmp]
    print('3')
    mean = train_data_for_norm.mean()
    std = train_data_for_norm.std()
    print('4')
    return mean, std

def convert_Dloader_3(fold, list_class, flag_tr_val_te, batch_size, num_workers = 1, shuffle = True, drop_last = False):
        dataset = Dataset_3(fold, list_class, flag_tr_val_te)
        Data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  num_workers=num_workers,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  drop_last=drop_last
                                                  )
        return Data_loader


def concat_class_of_interest(config, fold, list_class, flag_tr_val_te ='train'):
    """
    fold = 1~10
    list_class_of_interset = [0,0,0,1,1] # should be one-hot encoded
    """
    if st.list_data_type[st.data_type_num] == 'Density':
        dtype = np.float64
    elif st.list_data_type[st.data_type_num] == 'ADNI_JSY':
        dtype = np.float32
    elif 'ADNI_Jacob' in st.list_data_type[st.data_type_num] :
        dtype = np.uint8
    elif 'ADNI_AAL_256' in st.list_data_type[st.data_type_num]:
        dtype = np.uint8

    load_dir = st.fold_npy_dir
    fold_index = fold - 1

    list_image = []
    list_lbl = []
    list_age = []
    list_MMSE = []
    num_sample = 0
    for i_class_type in range(len(list_class)):
        if list_class[i_class_type] == 1:

            if flag_tr_val_te == 'train':
                tmp_dat_dir = st.train_fold_dir[fold_index][i_class_type]
            elif flag_tr_val_te == 'val':
                tmp_dat_dir = st.val_fold_dir[fold_index][i_class_type]
            elif flag_tr_val_te == 'test':
                tmp_dat_dir = st.test_fold_dir[fold_index][i_class_type]

            dat_dir = load_dir + tmp_dat_dir + '_' + st.list_data_name[0] + '.npy'
            list_image.append(np.memmap(filename=dat_dir, mode="r", dtype=dtype).reshape(-1, config.modality, st.x_size, st.y_size, st.z_size))
            num_sample += list_image[-1].shape[0]

            lbl_dir = load_dir + tmp_dat_dir + '_' + st.list_data_name[1] + '.npy'
            list_lbl.append(np.load(lbl_dir))

            age_dir = load_dir + tmp_dat_dir + '_' + st.list_data_name[2] + '.npy'
            list_age.append(np.load(age_dir))

            MMSE_dir = load_dir + tmp_dat_dir + '_' + st.list_data_name[3] + '.npy'
            list_MMSE.append(np.load(MMSE_dir))


    dat_image = np.vstack(list_image)
    dat_image = dat_image.astype('float32')
    dat_age = np.hstack(list_age)
    dat_MMSE = np.hstack(list_MMSE)

    """ make the label sequential """
    for i in range(sum(list_class)):
        if list_lbl[i][0] != i:
            list_lbl[i] = np.full_like(list_lbl[i], i, dtype=np.uint8)
    dat_lbl = np.hstack(list_lbl)

    return [dat_image, dat_lbl, dat_age, dat_MMSE]

