import os
import sys
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image

DSET_PATH = "/mnt/d/Dataset/train"

def create_split(path, out_file):
    """
    Creates a valid split files 
    """
    f = open(out_file, 'w')
    files = os.listdir(path)
    out_string = ""
    for file in files:
        if file[-3:] == "txt":
            continue
        kidney_file = os.path.join(path,file)
        image_file = os.path.join(kidney_file, "images")
        label_file = os.path.join(kidney_file, "labels")
        for k_slice in os.listdir(image_file):
            slice_image = os.path.join(image_file, k_slice)
            slice_label = os.path.join(label_file, k_slice)
            out_string += str(os.path.abspath(slice_image)) + \
                        "," + str(os.path.abspath(slice_label)) + "\n"
    f.write(out_string)

class LiverSet(Dataset):
    def __init__(self, path):
        """
        File Format
        Feature_File, Label_File
        """
        file = open(path, 'r')
        self.dset  = self.process_file(file.readlines())
        file.close()

    #ADD CACHING
    def __getitem__(self, ind):
        feature_file, label_file = self.dset[ind]
        feature_file, label_file = feature_file.strip(), label_file.strip()
        feature_img, label_img = Image.open(feature_file) , Image.open(label_file)
        feature_img.show()
        label_img.show()
        feature_tensor, label_tensor = torch.from_numpy(np.array(feature_img).astype(float)), \
                                       torch.from_numpy(np.array(label_img).astype(int))
        return ([feature_tensor], label_tensor)

    def __len__(self):
        return len(self.dset)

    def process_file(self, file_lines):
        return [line.split(",") for line in file_lines]

if __name__ == "__main__":
    #create_split(DSET_PATH, DSET_PATH+"/dset.txt")
    d = LiverSet(DSET_PATH+"/dset.txt")
    for _ in range(len(d)):
        d[_]
        print(_/len(d))


