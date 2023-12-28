from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from typing import List, Dict, Callable, Any
import torch
import torch.nn as nn

class Eval:
    """
    This Class will be called during the training loop and collect
    information on the model.
    """
    def __init__(self, metrics:List[Callable], args:List[List[Any]]):
        """
        metrics: List of function to be called at train time
        args: List of Lists of arguments for each metric function
              Eath metric function will be called with with the following
              format FN(model, *(args[fn_index])
        """
        self.metrics = metrics
        self.args = args
    def __call__(self, model: nn.Module):
        return [metric(model, *arg) for (metric, arg) in zip(self.metrics, self.args)]

class Pipeline():
    def __init__(self, dset: Dataset, transforms : List[Callable], 
            loader_kwargs : Dict[str, Any], dset_split: List[float], device: torch.device):
        """
        dset: Dataset, torch Dataset representing the dataset
              Dataset[i] should return a tuple (features: List of features, label)
        transforms: List, List of functionals which will be called in this order
                    when loading each dataset item through the dataloader
        loader_kwargs: Dict[kwarg, value] a dictionary containing the 
                       arguements to the dataloader
        dset_split: List[float], list describing the train, val, test split
        """
        self.train_set, self.test_set, self.val_set  = \
                        random_split(dset, dset_split)
        self.train_loader = DataLoader(self.train_set, 
                collate_fn=self.create_collate(transforms), **loader_kwargs)
        self.val_loader = DataLoader(self.val_set, 
                collate_fn=self.create_collate(transforms), **loader_kwargs)
        self.test_set = DataLoader(self.test_set, 
                collate_fn=self.create_collate(transforms), **loader_kwargs)
        self.device = device

    def create_collate(self, transforms:List) -> Callable:
        #Features can be an array of more than one feature
        if len(transforms) == 0:
            return None 

        def collate(model_input):
            #Batch the Data
            labels = torch.stack([batch[1] for batch in model_input])
            temp = [batch[0] for batch in model_input]
            features = []
            for i in range(len(temp[0])):
                features.append(torch.stack([item[i] for item in temp]))
            for transform in transforms:
                features = [transform(feature) for feature in features]
                labels   = transform(labels)
            return (features, labels)

        return collate

    def train(self, model : Callable, epochs:int,  optim: torch.optim.Optimizer, 
                            loss_fn: nn.Module , eval_fn: Callable, eval_frequency: int, 
                            loss_frequency: int, display_metric: Callable=None) -> None:
        """
        eval_fn: Function which will evaluate the model and return a list of metrics
                 in a predefined order
        display_metric: Displays the latest metrics to the terminal
        eval_frequency: how often the model is evaluated
        loss_frequency: how often the loss is printed
        """
        metrics = []
        running_loss = 0
        model = model.to(self.device)
        for e in range(epochs):
            for i, (features, label) in enumerate(self.train_loader):
                if i % eval_frequency == 0 and i!=0:
                    print("Evaluating Model.")
                    metrics.append(eval_fn(model).append(running_loss/eval_frequency))
                    display_metric(metrics[-1]) #Display the most recently  metrics
                if i % loss_frequency == 0:
                    print(f"Loss Epoch {e}, Iteration {i} : {running_loss/eval_frequency}")
                    running_loss = 0
                label = label.to(self.device)
                features = [feature.to(self.device) for feature in features]
                optim.zero_grad()
                output = model(*features)
                loss = loss_fn(output, label)
                running_loss += loss.item()
                loss.backward()
                optim.step()
            if scheduler != None:
                scheduler.step()
        return metrics
