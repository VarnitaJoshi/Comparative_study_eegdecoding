
import os
import logging

import mne
from mne.channels import make_standard_montage
from contextlib import redirect_stdout
from moabb.datasets.base import BaseDataset

log = logging.getLogger(__name__)
output_file = 'schirrmeister_cn.txt'
with open(output_file,'w') as file:
    with redirect_stdout(file):

        LOCAL_DATA_PATH = "/home/mt0/22CS60R61/MTP/NipsPaper2023-8E26/MTP_2024/Braindecode/schirr"


        class Schirrmeister2017(BaseDataset):

            def __init__(self):
                # print("Flow 1")
                super().__init__(
                    subjects=list(range(1, 15)),
                    sessions_per_subject=1,
                    events=dict(right_hand=1, left_hand=2, rest=3, feet=4),
                    code="Schirrmeister2017",
                    interval=[0, 4],
                    paradigm="imagery",
                    doi="10.1002/hbm.23730",
                )

            def data_path(
                self, subject, path=None, force_update=False, update_path=None, verbose=None
            ):
                # print("Flow 2")
                if subject not in self.subject_list:
                    raise ValueError("Invalid subject number")

                data_path2 = LOCAL_DATA_PATH

                # Check if the data path exists
                if not os.path.exists(data_path2):
                    print(data_path2)
                    raise FileNotFoundError(f"Data directory for subject {subject} not found.")

                return [os.path.join(data_path2, d) for d in ["train", "test"]]

            def _get_single_subject_data(self, subject):
                # print("Flow 3")
                train_raw, test_raw = [
                    mne.io.read_raw_edf(os.path.join(self.data_path(subject)[i], f"{subject}.edf"),
                                        infer_types=True, preload=True)
                    for i in range(2)
                ]

                # Select only EEG sensors (remove EOG, EMG),
                # and also set montage for visualizations
                montage = make_standard_montage("standard_1005")
                train_raw, test_raw = [
                    raw.pick_types(eeg=True).set_montage(montage) for raw in (train_raw, test_raw)
                ]
                sessions = {
                    "0": {"0train": train_raw, "1test": test_raw},
                }
                return sessions

        # Instantiate the Schirrmeister2017 dataset
        dataset = Schirrmeister2017()

        # Dictionary to store data for all subjects
        all_subject_data = {}

        # Load data for each subject
        for subject in dataset.subject_list:
            all_subject_data[subject] = dataset._get_single_subject_data(subject)

        import numpy as np
        from mne import Epochs
        import mne

        def extract_labels_features(all_subjects_data, event_id):
            all_labels = []
            all_features = []

            for subject_data in all_subjects_data.values():
                for session_data in subject_data.values():  # Iterate over session dictionaries
                    train_data = session_data.get("0train", None)  # Get train data
                    if train_data is not None:
                        # Extract events from annotations
                        events, _ = mne.events_from_annotations(train_data, event_id=event_id)

                        # Create epochs based on event IDs
                        epoch = Epochs(train_data, events, event_id=event_id, tmin=-0.1, tmax=0.7, on_missing='warn')

                        # Extract labels and features
                        labels = epoch.events[:, -1]
                        features = epoch.get_data()

                        # Append labels and features to lists
                        all_labels.append(labels)
                        all_features.append(features)

            return np.concatenate(all_labels), np.concatenate(all_features, axis=0)

        # Define event IDs for the four classes
        event_id = {'right_hand': 1, 'left_hand': 2, 'rest': 3, 'feet': 4}

        # Extract labels and features
        labels, features = extract_labels_features(all_subject_data, event_id)

        # Commented out IPython magic to ensure Python compatibility.
        import mne
        from mne.io import concatenate_raws

        import os
        import re
        import io
        # import cv2
        import random
        import string
        import warnings
        import numpy as np
        import matplotlib.pyplot as plt

        # from google.colab.patches import cv2_imshow

        import torch
        import torch.nn as nn
        import torch.optim as optim
        import torch.utils.data as data
        import torch.nn.functional as F
        from torch.autograd import Variable

        import lightning as L
        from lightning.pytorch import Trainer, seed_everything
        from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
        from lightning.pytorch.callbacks.early_stopping import EarlyStopping
        from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

        from torchmetrics.classification import Accuracy

        # %matplotlib inline
        # plt.rcParams['axes.facecolor'] = 'lightgray'

        # !pip install -q --no-cache-dir mne lightning torchmetrics

        class EEGDataset(data.Dataset):
            def __init__(self, x, y, inference=False):
                super().__init__()

                N_SAMPLE = x.shape[0]
                val_idx = int(0.9 * N_SAMPLE)
                train_idx = int(0.81 * N_SAMPLE)

                if not inference:
                    self.train_ds = {
                        'x': x[:train_idx],
                        'y': y[:train_idx],
                    }
                    self.val_ds = {
                        'x': x[train_idx:val_idx],
                        'y': y[train_idx:val_idx],
                    }
                    self.test_ds = {
                        'x': x[val_idx:],
                        'y': y[val_idx:],
                    }
                else:
                    self.__split = "inference"
                    self.inference_ds = {
                        'x': [x],
                    }

            def __len__(self):
                return len(self.dataset['x'])

            def __getitem__(self, idx):

                x = self.dataset['x'][idx]
                if self.__split != "inference":
                    y = self.dataset['y'][idx]
                    x = torch.tensor(x).float()
                    y = torch.tensor(y).unsqueeze(-1).float()
                    return x, y
                else:
                    x = torch.tensor(x).float()
                    return x

            def split(self, __split):
                self.__split = __split
                return self

            @classmethod
            def inference_dataset(cls, x):
                return cls(x, inference=True)

            @property
            def dataset(self):
                assert self.__split is not None, "Please specify the split of dataset!"

                if self.__split == "train":
                    return self.train_ds
                elif self.__split == "val":
                    return self.val_ds
                elif self.__split == "test":
                    return self.test_ds
                elif self.__split == "inference":
                    return self.inference_ds
                else:
                    raise TypeError("Unknown type of split!")

        # labels

        # concatenated_labels_train.squeeze()
        # concatenated_labels_test.squeeze()
        num_classes = 4
        labels=labels.astype(int)
        # Assuming labels is your array containing labels 1, 2, 3, 4
        labels = labels - 1
        labels=np.eye(num_classes)[labels]
        eeg_dataset = EEGDataset(x= features, y=labels)

        # labels

        class AvgMeter(object):
            def __init__(self, num=64):
                self.num = num
                self.reset()

            def reset(self):
                self.losses = []

            def update(self, val):
                self.losses.append(val)

            def show(self):
                out = torch.mean(
                    torch.stack(
                        self.losses[np.maximum(len(self.losses)-self.num, 0):]
                    )
                )
                return out

        class ModelWrapper(L.LightningModule):
            def __init__(self, arch, dataset, batch_size, lr, max_epoch):
                super().__init__()

                self.arch = arch
                self.dataset = dataset
                self.batch_size = batch_size
                self.lr = lr
                self.max_epoch = max_epoch

                # self.train_accuracy = Accuracy(task="binary")
                self.train_accuracy = Accuracy(task="MULTICLASS", num_classes = 4)
                self.val_accuracy = Accuracy(task="MULTICLASS", num_classes = 4)
                self.test_accuracy = Accuracy(task="MULTICLASS", num_classes = 4)

                self.automatic_optimization = False

                self.train_loss = []
                self.val_loss = []

                self.train_acc = []
                self.val_acc = []

                self.train_loss_recorder = AvgMeter()
                self.val_loss_recorder = AvgMeter()

                self.train_acc_recorder = AvgMeter()
                self.val_acc_recorder = AvgMeter()

            def forward(self, x):
                # return self.arch(x)
                # print("Hello in forward pass of model wrapper ")
                # print("forward pass of model wrapper")
                # print(x.shape)
                z = self.arch(x)
                # print("after callinf models architecture shape is ")
                # print(z.shape)
                # print("Shape of x after calling model architecture is ")
                # print(z.shape)
                # return z
                # print(z)
                return F.softmax(z, dim = 1)

            def training_step(self, batch, batch_nb):
                # print("------Shape predictions in training step-------  ")
                # print("training step of model wrapper")
                x, y = batch
                # print("shape of x and y")
                # print(x.shape)
                # print(y.shape)
                # y_hat = self(x)
                y_hat_probs = self(x)
                # print("here in training step, y_hat shape is ")
                # print(y_hat_probs.shape)
                # y.squeeze()
                # print("----printing the predictions in train step , value of y_hat------")
                # print(y_hat)
                # print("------shape predictions.-------  ")
                # print(y_hat.shape)
                # print("--------shape targets.------  ")
                # print(y.shape)
                # print("-------shape y_hatprobs in training step------")
                # print(y_hat_probs)
                import torch
                # -------------------------------------------------------------------------------------------------------------
                # Assuming y is your original target tensor with shape [batch_size, num_classes, 1]
                # Convert y to one-hot encoded tensor
                sh = y_hat_probs.shape[0]
                num_classes = 4  # Assuming you have 4 classes_
                if(sh == 64):
                    y_one_hot = torch.zeros(64, 4)
                    y_indices = y.squeeze().long()
                else :
                    y_one_hot = torch.zeros(sh, 4)
                    y_indices = y.squeeze().long()

                # y_index = y.squeeze().to(torch.int64)
                # y_one_hot.scatter_(1, y_index.unsqueeze(1), 1)
                y_one_hot.scatter_add_(1, y_indices, torch.ones_like(y_indices, dtype=torch.float32))
                # y_one_hot.scatter_(1, y.squeeze(), 1)  # Assuming y is 2-dimensional, squeeze removes the last dimension

                # Now, y_one_hot should have shape [batch_size, num_classes]
                # print("shape of y_one_hot, shape should be [batch_size, num_classes] in training step")
                # print(y_one_hot.shape)

                # Reshape y_one_hot to match the shape expected by the model (1D or 0D tensor)
                # For example, if you want to take argmax to get the predicted class index
                y = torch.argmax(y_one_hot, dim=1)

                # print("shape of y should be [batch_size] in training step")
                # print(y.shape)
                # print("shape end in training ")
                # print("shape of y_hatprobs and y are in training step before cross entropy")
                # print(y_hat_probs.shape)
                # print(y.shape)

                loss = F.cross_entropy(y_hat_probs, y)
                # loss = F.binary_cross_entropy_with_logits(y_hat, y)
                #####printing the predictions
                # print("----printing the predictions in train step------")
                # print(y_hat)
                # print("------shape predictions.-------  ")
                # print(y_hat.shape)
                # print("--------shape targets.------  ")
                # print(y.shape)
                # print("shape end")

                # print("---------prediction end in train step-------")
                # loss = F.cross_entropy(y_hat, y)
                self.train_accuracy.update(y_hat_probs, y)
                acc = self.train_accuracy.compute().data.cpu()

                opt = self.optimizers()
                opt.zero_grad()
                self.manual_backward(loss)
                opt.step()
                self.train_loss_recorder.update(loss.data)
                self.train_acc_recorder.update(acc)

                self.log("train_loss", loss, prog_bar=True)
                self.log("train_acc", acc, prog_bar=True)

            def on_train_epoch_end(self):
                # print("train epoch end of model wrapper")
                sch = self.lr_schedulers()
                sch.step()

                self.train_loss.append(self.train_loss_recorder.show().data.cpu().numpy())
                self.train_loss_recorder = AvgMeter()

                self.train_acc.append(self.train_acc_recorder.show().data.cpu().numpy())
                self.train_acc_recorder = AvgMeter()

            def validation_step(self, batch, batch_nb):
                # print("validation step of model wrapper")
                x, y = batch
                # print("shape of x and y")
                # print(x.shape)
                # print(y.shape)
                # y_hat = self(x)
                y_hat_probs = self(x)
                # print("Here in validation step, y_hat_probs shape is ")
                # print(y_hat_probs.shape)
                # print(y_hat_probs)
                # _, y_hat = torch.max(y_hat_probs, dim=1)  # Get the index of the maximum probability
                # y.squeeze()
                # y.squeeze()
                import torch
                # -------------------------------------------------------------------------------------------------------------
                # Assuming y is your original target tensor with shape [batch_size, num_classes, 1]
                # Convert y to one-hot encoded tensor
                num_classes = 4  # Assuming you have 4 classes
                y_one_hot = torch.zeros(64, 4)
                y_indices = y.squeeze().long()
                # y_index = y.squeeze().to(torch.int64)
                # y_one_hot.scatter_(1, y_index.unsqueeze(1), 1)
                y_one_hot.scatter_add_(1, y_indices, torch.ones_like(y_indices, dtype=torch.float32))
                # y_one_hot.scatter_(1, y.squeeze(), 1)  # Assuming y is 2-dimensional, squeeze removes the last dimension

                # Now, y_one_hot should have shape [batch_size, num_classes]
                # print("shape of y_one_hot, shape should be [batch_size, num_classes] in validation step")
                # print(y_one_hot.shape)

                # Reshape y_one_hot to match the shape expected by the model (1D or 0D tensor)
                # For example, if you want to take argmax to get the predicted class index
                y = torch.argmax(y_one_hot, dim=1)

                # print("shape of y should be [batch_size] in validation step")
                # print(y.shape)
                # Now, y should have shape [batch_size]
                # -----------------------------------------------------------------------------------------------------------------------
                # print("in validation step shape of y")
                # print(y.shape)
                # print(y)
                # y.squeeze()
                # print("print shape of y after y squeeze")
                # print(y.shape)
                # print("in validation step shape of y_hat in validation step")
                # print(y_hat.shape)
                # print(y_hat)
                # print("shape before cross_entropy y_hat_probs and y")
                # print(y_hat_probs.shape)
                # print(y.shape)
                loss = F.cross_entropy(y_hat_probs, y)
                # loss = F.binary_cross_entropy_with_logits(y_hat, y)
                # loss = F.cross_entropy(y_hat, y)
                self.val_accuracy.update(y_hat_probs, y)
                acc = self.val_accuracy.compute().data.cpu()

                self.val_loss_recorder.update(loss.data)
                self.val_acc_recorder.update(acc)

                self.log("val_loss", loss, prog_bar=True)
                self.log("val_acc", acc, prog_bar=True)

            def on_validation_epoch_end(self):
                # print("validation epoch end of model wrapper ")
                self.val_loss.append(self.val_loss_recorder.show().data.cpu().numpy())
                self.val_loss_recorder = AvgMeter()

                self.val_acc.append(self.val_acc_recorder.show().data.cpu().numpy())
                self.val_acc_recorder = AvgMeter()

            def test_step(self, batch, batch_nb):
                # print("test step of model wrapper")
                x, y = batch
                # print("hum hai test step mein")
                # print("shape of x")
                # print(len(x))
                # print("shape of y")
                # print(len(y))
                y_hat_probs = self(x)
                # print("Here in  step, y_hat shape is ")
                # print(y_hat_probs.shape)
                # print(y_hat_probs)
                # _, y_hat = torch.max(y_hat_probs, dim=1)  # Get the index of the maximum probability
                # y.squeeze()
                # y.squeeze()
                import torch
                # -------------------------------------------------------------------------------------------------------------
                # Assuming y is your original target tensor with shape [batch_size, num_classes, 1]
                # Convert y to one-hot encoded tensor
                num_classes = 4  # Assuming you have 4 classes
                y_one_hot = torch.zeros(4, 4)
                # print("y squeeze ke pehle, y ki shape")
                # print(y.shape)
                y_indices = y.squeeze().long()
                # print("hello  after y_indices")
                # y_index = y.squeeze().to(torch.int64)
                # y_one_hot.scatter_(1, y_index.unsqueeze(1), 1)
                y_one_hot.scatter_add_(1, y_indices, torch.ones_like(y_indices, dtype=torch.float32))
                # y_one_hot.scatter_(1, y.squeeze(), 1)  # Assuming y is 2-dimensional, squeeze removes the last dimension
                # print("hello after y_one_hot")
                # Now, y_one_hot should have shape [batch_size, num_classes]
                # print("shape of y_one_hot, shape should be [batch_size, num_classes] in validation step")
                # print(y_one_hot.shape)

                # Reshape y_one_hot to match the shape expected by the model (1D or 0D tensor)
                # For example, if you want to take argmax to get the predicted class index
                y = torch.argmax(y_one_hot, dim=1)
                # print("length of y")
                # print(len(y))

                # print("shape of y should be [batch_size] in validation step")
                # print(y.shape)
                # Now, y should have shape [batch_size]
                # y_hat = self(x)
                # loss = F.binary_cross_entropy_with_logits(y_hat, y)
                # print("length of y_hat_probs and y before cross entropy")
                # print(len(y_hat_probs))
                # print(len(y))
                loss = F.cross_entropy(y_hat_probs, y)  # problem is here
                self.test_accuracy.update(y_hat_probs, y)

                self.log(
                    "test_loss",
                    loss,
                    prog_bar=True,
                    logger=True,
                )
                self.log(
                    "test_acc",
                    self.test_accuracy.compute(),
                    prog_bar=True,
                    logger=True,
                )

            def on_train_end(self):

                # Loss
                loss_img_file = "./output/deep4net/loss_plot_sh
                irr.png"
                plt.plot(self.train_loss, color = 'r', label='train')
                plt.plot(self.val_loss, color = 'b', label='validation')
                plt.title("Loss Curves")
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.legend()
                plt.grid()
                plt.savefig(loss_img_file)
                plt.clf()
                # img = cv2.imread(loss_img_file)
                # cv2_imshow(img)

                # Accuracy
                acc_img_file = "./output/deep4net/acc_plot_schirr.png"
                plt.plot(self.train_acc, color = 'r', label='train')
                plt.plot(self.val_acc, color = 'b', label='validation')
                plt.title("Accuracy Curves")
                plt.xlabel("Epoch")
                plt.ylabel("Accuracy")
                plt.legend()
                plt.grid()
                plt.savefig(acc_img_file)
                plt.clf()
                # img = cv2.imread(acc_img_file)
                # cv2_imshow(img)

            def train_dataloader(self):
                # print("train dataloader of model wrapper")
                return data.DataLoader(
                    dataset=self.dataset.split("train"),
                    batch_size=self.batch_size,
                    shuffle=True,
                )

            def val_dataloader(self):
                # print("validation dataloader of model wrappper")
                return data.DataLoader(
                    dataset=self.dataset.split("val"),
                    batch_size=self.batch_size,
                    shuffle=False,
                )

            def test_dataloader(self):
                # print("test dataloader of model wrapper")
                return data.DataLoader(
                    dataset=self.dataset.split("test"),
                    batch_size=4,
                    shuffle=False,
                )

            def configure_optimizers(self):
                # print("optimizer fxn in model wrapper")
                optimizer = optim.Adam(
                    self.parameters(),
                    lr=self.lr,
                )
                lr_scheduler = {
                    "scheduler": optim.lr_scheduler.MultiStepLR(
                        optimizer,
                        milestones=[
                            int(self.max_epoch * 0.25),
                            int(self.max_epoch * 0.5),
                            int(self.max_epoch * 0.75),
                        ],
                        gamma=0.1
                    ),
                    "name": "lr_scheduler",
                }
                return [optimizer], [lr_scheduler]

        class PositionalEncoding(nn.Module):
            def __init__(self, num_hiddens, dropout, max_len=1000):
                super().__init__()
                self.dropout = nn.Dropout(dropout)
                # Create a long enough P
                self.p = torch.zeros((1, max_len, num_hiddens))
                x = torch.arange(max_len, dtype=torch.float32).reshape(
                    -1, 1) / torch.pow(10000, torch.arange(
                    0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
                self.p[:, :, 0::2] = torch.sin(x)
                self.p[:, :, 1::2] = torch.cos(x)

            def forward(self, x):
                x = x + self.p[:, :x.shape[1], :].to(x.device)
                return self.dropout(x)

        class TransformerBlock(nn.Module):
            def __init__(self, embed_dim, num_heads, dim_feedforward, dropout=0.1):
                super().__init__()

                self.attention = nn.MultiheadAttention(
                    embed_dim,
                    num_heads,
                    dropout,
                    batch_first=True,
                )
                self.mlp = nn.Sequential(
                    nn.Linear(embed_dim, dim_feedforward),
                    nn.ReLU(True),
                    nn.Dropout(dropout),
                    nn.Linear(dim_feedforward, embed_dim),
                    #nn.Linear(dim_feedforward, 4),
                )

                self.layernorm0 = nn.LayerNorm(embed_dim)
                self.layernorm1 = nn.LayerNorm(embed_dim)

                self.dropout = dropout

            def forward(self, x):
                y, att = self.attention(x, x, x)
                y = F.dropout(y, self.dropout, training=self.training)
                x = self.layernorm0(x + y)
                y = self.mlp(x)
                y = F.dropout(y, self.dropout, training=self.training)
                x = self.layernorm1(x + y)
                return x

        class EEGClassificationModel(nn.Module):
            def __init__(self, eeg_channel, dropout):
                super().__init__()

                self.conv = nn.Sequential(
                    nn.Conv1d(
                        eeg_channel, eeg_channel, 11, 1, padding=5, bias=False
                    ),
                    nn.BatchNorm1d(eeg_channel),
                    nn.ReLU(True),
                    nn.Dropout1d(dropout),
                    nn.Conv1d(
                        eeg_channel, eeg_channel * 2, 11, 1, padding=5, bias=False
                    ),
                    nn.BatchNorm1d(eeg_channel * 2),
                )

                self.transformer = nn.Sequential(
                    PositionalEncoding(eeg_channel * 2, dropout),
                    TransformerBlock(eeg_channel * 2, 4, eeg_channel // 8, dropout),
                    TransformerBlock(eeg_channel * 2, 4, eeg_channel // 8, dropout),
                )

                self.mlp = nn.Sequential(
                    nn.Linear(eeg_channel * 2, eeg_channel // 2),
                    nn.ReLU(True),
                    nn.Dropout(dropout),
                    nn.Linear(eeg_channel // 2,4),
                )

            def forward(self, x):
                # print("forward pass of EEG classification")
                # print(x.shape)
                x = self.conv(x)
                # print("hello after conv layer")
                # print(x.shape)
                x = x.permute(0, 2, 1)
                # print("after permutation shape is ")
                # print(x.shape)
                x = self.transformer(x)
                # print("hello after transformer layer")
                # print(x.shape)
                x = x.permute(0, 2, 1)
                # print("shape after permuattion")
                # print(x.shape)
                x = x.mean(dim=-1)
                # print("shape after mean")
                # print(x.shape)
                x = self.mlp(x)
                # print("hello after mlp layer--- printing shape of x")
                # x = x.squeeze()
                # print(x.shape)
                return x

        EEG_CHANNEL = 128

        MODEL_NAME = "EEGClassificationModel"
        model = EEGClassificationModel(eeg_channel=EEG_CHANNEL, dropout=0.125)

        MAX_EPOCH = 3
        BATCH_SIZE = 64
        LR = 0.02
        CHECKPOINT_DIR = os.getcwd()
        SEED = int(np.random.randint(2147483647))

        print(f"Random seed: {SEED}")

        model = ModelWrapper(model, eeg_dataset, BATCH_SIZE, LR, MAX_EPOCH)

        # !rm -rf logs/

        # Commented out IPython magic to ensure Python compatibility.

        # %reload_ext tensorboard
        # %tensorboard --logdir=logs/lightning_logs/

        tensorboardlogger = TensorBoardLogger(save_dir="logs/")
        csvlogger = CSVLogger(save_dir="logs/")
        lr_monitor = LearningRateMonitor(logging_interval='step')
        optimizer = optim.Adam(model.parameters(), lr=LR)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCH)
        checkpoint = ModelCheckpoint(
            monitor='val_acc',
            dirpath=CHECKPOINT_DIR,
            mode='max',
        )
        early_stopping = EarlyStopping(
            monitor="val_acc", min_delta=0.00, patience=3, verbose=False, mode="max"
        )


        seed_everything(SEED, workers=True)


        trainer = Trainer(
            accelerator="auto",
            devices=1,
            max_epochs=MAX_EPOCH,
            logger=[tensorboardlogger, csvlogger],
            callbacks=[lr_monitor, checkpoint, early_stopping],
            log_every_n_steps=5,
        )
        trainer.fit(model)

        # trainer.test(ckpt_path="best")

        # os.rename(
        #     checkpoint.best_model_path,
        #     os.path.join(CHECKPOINT_DIR, f"{MODEL_NAME}_best.ckpt")
        # )