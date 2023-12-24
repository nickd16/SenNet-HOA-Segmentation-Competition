from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import List, Dict, Callable

class Pipeline():
    def __init__(self, dset: Dataset, transforms : List[Callable], 
            loader_kwargs : Dict[str, Any], dset_split: List[float]):
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

    def create_collate(self, transforms:List) -> Callable:
        #Features can be an array of more than one feature
        def collate((features, labels)):
            for transform in transforms:
                features = [transform(feature) for feature in features]
                labels   = transform(labels)
            return (features, labels)
        return collate

    def train(self, model : Callable, epochs:int,  optim: torch.optim.Optimizer, 
                            loss_fn: Callable, eval_fn: Callable, eval_frequency: int, 
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
        for e in range(epochs):
            for i, (features, label) in enumerate(self.data_loader):
                if i % eval_frequency == 0:
                    metrics.append(eval_fn(model) + [running_loss/eval_frequency])
                    display_metric(metrics[-1]) #Display the most recently  metrics
                if i % loss_frequency == 0:
                    print(f"Loss Epoch {e}, Iteration {i} : {running_loss/eval_frequency}")
                    running_loss = 0
                optim.zero_grad()
                output = model(*features)
                loss = loss_fn(output, label)
                loss.backward()
                optim.step()
            if scheduler != None:
                scheduler.step()
        return metrics

            



