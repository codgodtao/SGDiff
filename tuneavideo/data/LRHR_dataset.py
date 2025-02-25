import h5py
from torch.utils.data import Dataset
import torch
import numpy as np
import clip
from einops import rearrange
import json
import os


class MMDataset(Dataset):
    def __init__(self, dataroot, grounding_file="xx.json", scene_file=""):
        data = h5py.File(dataroot)
        if data.get('gt', None) is None:
            gt1 = data['lms'][...]
        else:
            gt1 = data['gt'][...]
        if "gf2" in dataroot: #10位数据
            img_scale = 1023.0
        else:
            img_scale = 2047.0 # 11位数据
        gt1 = np.array(gt1, dtype=np.float32) / img_scale
        self.gt = torch.from_numpy(gt1)

        ms1 = data["ms"][...]  # convert to np tpye for CV2.filter
        ms1 = np.array(ms1, dtype=np.float32) / img_scale
        self.ms = torch.from_numpy(ms1)

        lms1 = data["lms"][...]  # convert to np tpye for CV2.filter
        lms1 = np.array(lms1, dtype=np.float32) / img_scale
        self.lms = torch.from_numpy(lms1)

        pan1 = data['pan'][...]  # Nx1xHxW
        pan1 = np.array(pan1, dtype=np.float32) / img_scale  # Nx1xHxW
        self.pan = torch.from_numpy(pan1)  # In the satellite image, there are

        self.grounding = [json.loads(q)['answer'][37:] for q in open(os.path.expanduser(grounding_file), "r")]
        self.scene = [json.loads(q)['answer'] for q in open(os.path.expanduser(scene_file), "r")]

        # 提取所有的answer字段，并存储在一个列表中
        self.dataset_len = self.ms.shape[0]  # 目录下所有图像的数量
        print(self.dataset_len, len(self.grounding), len(self.scene))
        self.scene_ids = []
        self.grounding_ids = []
        print(self.pan.shape, self.lms.shape, self.gt.shape, self.ms.shape, self.dataset_len)
        print(torch.max(self.gt), torch.min(self.gt))

    def __len__(self):
        return self.dataset_len

    def encoding(self):
        for i in range(self.dataset_len):
            scene = self.scene[i]
            grounding = self.grounding[i]
            self.scene_ids.append(clip.tokenize(scene,truncate = True)[0])
            self.grounding_ids.append(clip.tokenize(grounding,truncate = True)[0])

    def __getitem__(self, index):
        img_HR = self.gt[index, :, :, :].float()
        img_MS = self.lms[index, :, :, :].float()
        img_PAN = self.pan[index, :, :, :].float()
        pan_concat = img_PAN.repeat(img_HR.shape[0], 1, 1)  # C H W
        condition = torch.stack([pan_concat, img_MS], dim=0)

        return {'PAN': img_PAN, 'MS': img_MS, 'HR': img_HR, 'condition': condition,
                'Res': rearrange(img_HR - img_MS, "c h w -> 1 c h w"), # residual map between HRMS and UpSampled MS
                'scene_id': self.scene_ids[index],
                'grounding_id': self.grounding_ids[index]}


if __name__ == "__main__":
    root = "/home/wangsong/geochat/pansharpening/test_data/WV3/test_wv3_data_FR.h5"
    grounding_file = "/home/wangsong/geochat/geochat/eval/WV3_full/answer_grounding.jsonl"
    scene_file = "/home/wangsong/geochat/geochat/eval/WV3_full/answer_scene.jsonl"
    dataset = MMDataset(root, grounding_file=grounding_file, scene_file=scene_file)
    # 测试 __len__ 方法
    dataset.encoding()
