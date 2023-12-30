import os
import sys
sys.path.append(os.path.join(os.getcwd(), "Pipeline"))
from pipeline import Pipeline, Eval
from utils import get_accuracy, add_dimension
from dset import LiverSet, DSET_PATH
from unet import UNET
import torch


def main():
    device = torch.device('cuda')
    d = LiverSet(DSET_PATH+"/dset.txt")
    p = Pipeline(d, [add_dimension], {"batch_size":1}, [0.9, .05, .05], device)
    model = UNET().to(device)
    adam = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = torch.nn.BCELoss()
    eval_fns = Eval([get_accuracy], [[p.val_loader, p.device]])
    p.train(model, 3, adam, loss_fn, eval_fns, 100, 10, print)


if __name__ == "__main__":
    main()

