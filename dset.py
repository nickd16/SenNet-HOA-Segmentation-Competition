import os
import sys
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from typing import IO, List

DSET_PATH = "/media/lenny/e8491f8e-2ac1-4d31-a37f-c115e009ec90/Dataset/train"

def create_split(path:str, out_file: IO):
    """
    Takes in the dataset and splits it into a file of filenames 
    This is to allow quick partitions of the dataset without having to 
    physically move the dataset on disk.
    """
    f = open(out_file, 'w')
    files = os.listdir(path)
    out_string = ""
    for file in files:
        if file != "kidney_2":
            continue
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
    def __init__(self, path:str):
        """
        File Format
        Feature_File, Label_File
        """
        file = open(path, 'r')
        self.dset  = self.process_file(file.readlines())
        file.close()

    #ADD CACHING
    def __getitem__(self, ind:int):
        """
        Takes in the index of the dataset and returns a tuple of a ([features], label)
        """
        feature_file, label_file = self.dset[ind]
        feature_file, label_file = feature_file.strip(), label_file.strip()
        feature_img, label_img = Image.open(feature_file) , Image.open(label_file)
        feature_tensor = torch.from_numpy(np.array(feature_img).astype(np.int32)).to(torch.float32)
        label_tensor = torch.from_numpy(np.array(label_img))
        label_tensor = (label_tensor==255).to(torch.float32)
        return ([feature_tensor], label_tensor)

    def __len__(self):
        return len(self.dset)

    def process_file(self, file_lines:List[str]):
        return [line.split(",") for line in file_lines]

if __name__ == "__main__":
    #create_split(DSET_PATH, DSET_PATH+"/dset.txt")
    d = LiverSet(DSET_PATH+"/dset.txt")



