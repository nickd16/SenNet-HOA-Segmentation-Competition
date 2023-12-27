import os
import sys
sys.path.append(os.path.join(os.getcwd(), "Pipeline"))
from pipeline import Pipeline
from dset import LiverSet, DSET_PATH
from unet import UNET


def main():
    d = LiverSet(DSET_PATH+"/dset.txt")
    p = Pipeline(d, [], {"batch_size":32}, [0.9, .05, .05])
    model = UNET().cuda()

if __name__ == "__main__":
    main()

