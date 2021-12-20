import json
import os
import io
import sys
sys.path.insert(0, '../')
from utils.imgio_gen import readgen
import numpy as np


# For VideoDeblur dataset
file = io.open('DVD_test.json','w',encoding='utf-8')
root = '/home/xuqian/dataset_DVD'
samples = []
phase = ['test']
for ph in phase:
    names = sorted(os.listdir(os.path.join(root, ph)))
    for name in names:
        sample_list = sorted(os.listdir(os.path.join(root, ph, name, 'input')))
        sample = [sample_list[i][:-4] for i in range(len(sample_list))]
        sample_sub = []
        for sam in sample:
            if not sam == ".DS_S":
                sample_sub.append(sam)
        l = {'name': name,'phase': ph,'sample': sample_sub}
        samples.append(l)

js = json.dump(samples, file, sort_keys=False, indent=4)
