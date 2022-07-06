import cv2
import numpy as np
import os
import torch
from torch.utils.data.dataset import Dataset

iso_list = [1600, 3200, 6400, 12800, 25600]
a_list = [3.513262, 6.955588, 13.486051, 26.585953, 52.032536]
g_noise_var_list = [11.917691, 38.117816, 130.818508, 484.539790, 1819.818657]

def pack_gbrg_raw_torch(raw):  # T H W
    T, H, W = raw.shape
    im = raw.unsqueeze(1)

    out = torch.cat((im[:, :, 1:H:2, 0:W:2],
                     im[:, :, 1:H:2, 1:W:2],
                     im[:, :, 0:H:2, 0:W:2],
                     im[:, :, 0:H:2, 1:W:2]), dim=1)
    return out

def normalize_raw_torch(raw):
    black_level = 240
    white_level = 2 ** 12 - 1
    raw = torch.clamp(raw.type(torch.float32) - black_level, 0) / (white_level - black_level)
    return raw

def open_CRVD_seq_raw(seq_path, file_pattern='frame%d_noisy0.tiff'):
    frame_list = []
    for i in range(7):
        raw = cv2.imread(os.path.join(seq_path, file_pattern % (i+1)), -1)
        raw = np.asarray(raw)
        raw = np.expand_dims(raw, axis=0)
        frame_list.append(raw)
    seq = np.concatenate(frame_list, axis=0)
    return seq

def open_CRVD_seq_raw_outdoor(seq_path, file_pattern='frame%d_noisy0.tiff'):
    frame_list = []
    for i in range(50):
        raw = cv2.imread(os.path.join(seq_path, file_pattern % i), -1)
        raw = np.asarray(raw)
        raw = np.expand_dims(raw, axis=0)
        frame_list.append(raw)
    seq = np.concatenate(frame_list, axis=0)
    return seq

def crop_position(patch_size, H, W):
    position_h = np.random.randint(0, (H - patch_size)//2 - 1) * 2
    position_w = np.random.randint(0, (W - patch_size)//2 - 1) * 2
    aug = np.random.randint(0, 8)
    return position_h, position_w, aug

def aug_crop(img, patch_size, position_h, position_w, aug):
    patch = img[:, position_h:position_h + patch_size + 2, position_w:position_w + patch_size + 2]

    if aug == 0:
        patch = patch[:, :-2, :-2]
    elif aug == 1:
        patch = np.flip(patch, axis=1)
        patch = patch[:, 1:-1, :-2]
    elif aug == 2:
        patch = np.flip(np.flip(patch, axis=1), axis=2)
        patch = patch[:, 1:-1, 1:-1]
    elif aug == 3:
        patch = np.flip(patch, axis=2)
        patch = patch[:, :-2, 1:-1]
    elif aug == 4:
        patch = np.transpose(np.flip(patch, axis=2), (0, 2, 1))
        patch = patch[:, :-2, 1:-1]
    elif aug == 5:
        patch = np.transpose(np.flip(np.flip(patch, axis=1), axis=2), (0, 2, 1))
        patch = patch[:, :-2, :-2]
    elif aug == 6:
        patch = np.transpose(patch, (0, 2, 1))
        patch = patch[:, 1:-1, 1:-1]
    elif aug == 7:
        patch = np.transpose(np.flip(patch, axis=1), (0, 2, 1))
        patch = patch[:, 1:-1, :-2]
    return patch


class CRVDTrainDataset(Dataset):
    def __init__(self, CRVD_path, patch_size, patches_per_epoch, mirror_seq=True):
        self.CRVD_path = CRVD_path
        self.patches_per_epoch = patches_per_epoch
        self.patch_size = patch_size * 2
        self.mirror_seq = mirror_seq
        self.scene_id_list = [1, 2, 3, 4, 5, 6]
        self.seqs = {}

        for iso in iso_list:
            for scene_id in self.scene_id_list:
                self.seqs['%d_%d_clean' % (iso, scene_id)] = open_CRVD_seq_raw(os.path.join(self.CRVD_path, 'indoor_raw_gt/scene%d/ISO%d' % (scene_id, iso)),
                                                                           'frame%d_clean_and_slightly_denoised.tiff')
                for i in range(10):
                    self.seqs['%d_%d_noisy_%d' % (iso, scene_id, i)] = open_CRVD_seq_raw(os.path.join(self.CRVD_path, 'indoor_raw_noisy/scene%d/ISO%d' % (scene_id, iso)),
                                                                           'frame%d_noisy{}.tiff'.format(i))

    def __getitem__(self, index):
        index = index % (len(iso_list) * len(self.scene_id_list) * 10)
        iso_index = index // (len(self.scene_id_list) * 10)
        scene_index = (index - iso_index * len(self.scene_id_list) * 10) // 10
        noisy_index = index % 10
        iso = iso_list[iso_index]
        scene_id = self.scene_id_list[scene_index]

        seq = self.seqs['%d_%d_clean' % (iso, scene_id)]
        seqn = self.seqs['%d_%d_noisy_%d' % (iso, scene_id, noisy_index)]
        T, H, W = seq.shape
        position_h, position_w, aug = crop_position(self.patch_size, H, W)
        seq = aug_crop(seq, self.patch_size, position_h, position_w, aug)
        seqn = aug_crop(seqn, self.patch_size, position_h, position_w, aug)
        clean_list, noisy_list = [], []
        for i in range(T):
            clean_list.append(np.expand_dims(seq[i], axis=0))
            noisy_list.append(np.expand_dims(seqn[i], axis=0))
        seq = torch.from_numpy(np.concatenate(clean_list, axis=0).astype(np.int32))
        seqn = torch.from_numpy(np.concatenate(noisy_list, axis=0).astype(np.int32))
        seq = normalize_raw_torch(pack_gbrg_raw_torch(seq))
        seqn = normalize_raw_torch(pack_gbrg_raw_torch(seqn))

        if self.mirror_seq:
            seq = torch.cat((seq, torch.flip(seq, dims=[0])), dim=0)
            seqn = torch.cat((seqn, torch.flip(seqn, dims=[0])), dim=0)

        a = torch.tensor(a_list[iso_index], dtype=torch.float32).view((1, 1, 1, 1)) / (2 ** 12 - 1 - 240)
        b = torch.tensor(g_noise_var_list[iso_index], dtype=torch.float32).view((1, 1, 1, 1))  / ((2 ** 12 - 1 - 240) ** 2)

        return {'seq': seq,
                'seqn': seqn,
                'a': a, 'b': b}

    def __len__(self):
        return self.patches_per_epoch

class CRVDTestDataset(Dataset):
    def __init__(self, CRVD_path):
        self.CRVD_path = CRVD_path
        self.scene_id_list = [7, 8, 9, 10, 11]
        self.seqs = {}

        for iso in iso_list:
            for scene_id in self.scene_id_list:
                self.seqs['%d_%d_clean' % (iso, scene_id)] = open_CRVD_seq_raw(os.path.join(self.CRVD_path, 'indoor_raw_gt/scene%d/ISO%d' % (scene_id, iso)),
                                                                           'frame%d_clean_and_slightly_denoised.tiff')
                self.seqs['%d_%d_noisy' % (iso, scene_id)] = open_CRVD_seq_raw(os.path.join(self.CRVD_path, 'indoor_raw_noisy/scene%d/ISO%d' % (scene_id, iso)),
                                                                           'frame%d_noisy0.tiff')

    def __getitem__(self, index):
        iso = iso_list[index // len(self.scene_id_list)]
        scene_id = self.scene_id_list[index % len(self.scene_id_list)]

        seq = torch.from_numpy(self.seqs['%d_%d_clean' % (iso, scene_id)].astype(np.float32))
        seqn = torch.from_numpy(self.seqs['%d_%d_noisy' % (iso, scene_id)].astype(np.float32))
        seq = normalize_raw_torch(pack_gbrg_raw_torch(seq))
        seqn = normalize_raw_torch(pack_gbrg_raw_torch(seqn))
        a = torch.tensor(a_list[index // len(self.scene_id_list)], dtype=torch.float32).view((1, 1, 1, 1))  / (2 ** 12 - 1 - 240)
        b = torch.tensor(g_noise_var_list[index // len(self.scene_id_list)], dtype=torch.float32).view((1, 1, 1, 1))  / ((2 ** 12 - 1 - 240) ** 2)

        return {'seq': seq,
                'seqn': seqn,
                'iso': iso, 'a': a, 'b': b, 'scene_id': scene_id}

    def __len__(self):
        return len(iso_list) * len(self.scene_id_list)

class CRVDOurdoorDataset(Dataset):
    def __init__(self, CRVD_path):
        self.CRVD_path = CRVD_path
        self.scene_id_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.seqs = {}

        self.iso = 25600
        for scene_id in self.scene_id_list:
            self.seqs['%d_%d_noisy' % (self.iso, scene_id)] = open_CRVD_seq_raw_outdoor(os.path.join(self.CRVD_path, 'outdoor_raw_noisy/scene%d/iso%d' % (scene_id, self.iso)),
                                                                           'frame%d.tiff')

    def __getitem__(self, index):
        scene_id = self.scene_id_list[index]

        seqn = torch.from_numpy(self.seqs['%d_%d_noisy' % (self.iso, scene_id)].astype(np.float32))
        seqn = normalize_raw_torch(pack_gbrg_raw_torch(seqn))
        a = torch.tensor(a_list[4], dtype=torch.float32).view((1, 1, 1, 1))  / (2 ** 12 - 1 - 240)
        b = torch.tensor(g_noise_var_list[4], dtype=torch.float32).view((1, 1, 1, 1))  / ((2 ** 12 - 1 - 240) ** 2)

        return {'seqn': seqn,
                'iso': self.iso, 'a': a, 'b': b, 'scene_id': scene_id}

    def __len__(self):
        return len(self.scene_id_list)