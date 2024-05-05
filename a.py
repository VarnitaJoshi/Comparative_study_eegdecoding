import torch
from braindecode.datasets import MOABBDataset
from numpy import multiply
from braindecode.preprocessing import(Preprocessor, exponential_moving_standardize, preprocess)
from braindecode.models import Deep4Net
from braindecode.util import set_random_seeds
from braindecode.preprocessing import create_windows_from_events

from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
from braindecode import EEGClassifier

import time
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import confusion_matrix
from braindecode.visualization import plot_confusion_matrix



y_true = [1, 2, 3, 4, 5, 6, 7, 8, 9]
y_pred = [1, 2, 3, 4, 5, 6, 7, 8, 9]

confusion_mat = confusion_matrix(y_true, y_pred)

# label_dict = windows_dataset.datasets[0].window_kwargs[0][1]['mapping']

labels = [1, 2, 3, 4, 5, 6, 7, 8, 9]

fig = plot_confusion_matrix(confusion_mat, class_names = labels, figsize = (20, 15))
fig.savefig('./output/deep4net/confusion_mat111')


# sed -r "s/\x1B\[([0-9]{1,2}(;[0-9]{1,2})?)?[mGK]//g" imput > output
# ssh 22CS60R61@10.5.18.108