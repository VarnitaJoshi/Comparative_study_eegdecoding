import os
import numpy as np
from mne import create_info
from mne.channels import make_standard_montage
from mne.io import RawArray
from scipy.io import loadmat
from contextlib import redirect_stdout


output_file = 'cho17_cn.txt'
with open(output_file,'w') as file:
    with redirect_stdout(file):

        class Cho2017:
            def __init__(self):
                self.subject_list = list(range(1, 53))
                # self.GIGA_URL = "https://ftp.cngb.org/pub/gigadb/pub/10.5524/100001_101000/100295/mat_data/"
                self.dataset_directory = "/home/mt0/22CS60R61/MTP/NipsPaper2023-8E26/MTP_2024/Braindecode/dataset"  # Path to the local dataset directory

            def _get_single_subject_data(self, subject):
                """Return data for a single subject."""
                fname = self.data_path(subject)

                data = loadmat(
                    fname,
                    squeeze_me=True,
                    struct_as_record=False,
                    verify_compressed_data_integrity=False,
                )["eeg"]

                # Define channel names and types
                eeg_ch_names = [
                    "Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3", "FC1",
                    "C1", "C3", "C5", "T7", "TP7", "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7",
                    "P9", "PO7", "PO3", "O1", "Iz", "Oz", "POz", "Pz", "CPz", "Fpz", "Fp2",
                    "AF8", "AF4", "AFz", "Fz", "F2", "F4", "F6", "F8", "FT8", "FC6", "FC4",
                    "FC2", "FCz", "Cz", "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2",
                    "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2",
                ]
                emg_ch_names = ["EMG1", "EMG2", "EMG3", "EMG4"]
                ch_names = eeg_ch_names + emg_ch_names + ["Stim"]
                ch_types = ["eeg"] * 64 + ["emg"] * 4 + ["stim"]
                montage = make_standard_montage("standard_1005")
                imagery_left = data.imagery_left - data.imagery_left.mean(axis=1, keepdims=True)
                imagery_right = data.imagery_right - data.imagery_right.mean(
                    axis=1, keepdims=True
                )

                eeg_data_l = np.vstack([imagery_left * 1e-6, data.imagery_event])
                eeg_data_r = np.vstack([imagery_right * 1e-6, data.imagery_event * 2])

                # Trials are already non-continuous. Edge artifact can appear but are likely to be present during rest / inter-trial activity
                eeg_data = np.hstack(
                    [eeg_data_l, np.zeros((eeg_data_l.shape[0], 500)), eeg_data_r]
                )

                info = create_info(ch_names=ch_names, ch_types=ch_types, sfreq=data.srate)
                raw = RawArray(data=eeg_data, info=info, verbose=False)
                raw.set_montage(montage)

                return {"0": {"0": raw}}

            def data_path(self, subject, path=None, force_update=False, update_path=None, verbose=None):
                if subject not in self.subject_list:
                    raise ValueError("Invalid subject number")

                filename = f"s{subject:02d}.mat"
                return os.path.join(self.dataset_directory, filename)

        # # Example usage:
        cho_dataset = Cho2017()
        all_subjects_data = {}
        for subject in cho_dataset.subject_list:
            if(subject != 1):
                all_subjects_data[subject] = cho_dataset._get_single_subject_data(subject)

        # DATA VISUALISATION
        # subject_data = all_subjects_data[2]["0"]  # Get data for subject 1, session 0
        # subject_data = all_subjects_data[2]["0"]["0"]

        # Print basic information about the data
        # print("Number of channels:", len(subject_data.info["ch_names"]))
        # print("Channel names:", subject_data.info["ch_names"])
        # print("Sampling frequency:", subject_data.info["sfreq"])
        # print("Number of samples:", subject_data.n_times)

        import matplotlib.pyplot as plt

        # Get the EEG data for the subject
        # subject_data = all_subjects_data[2]["0"]["0"]

        # Plot the EEG data
        # subject_data.plot(n_channels=64, duration=20, scalings={"eeg": 50e-6})
        # plt.show()

        # all_subjects_data[3]['0']['0']

        # all_subjects_data[2]

        # Print the EEG data for subject 3
        # eeg_data_subject_3 = all_subjects_data[3]['0']['0'].get_data()
        # print(eeg_data_subject_3)

        # Iterate over the subjects in all_subjects_data
        # for subject_id, subject_data in all_subjects_data.items():
        #     print(f"Subject {subject_id}:")
        #     # Iterate over the sessions for each subject
        #     for session_id, session_data in subject_data.items():
        #         # Access the labels for each session
        #         labels = session_data['0']['0'].events[:, -1]
        #         print(f"Labels for session {session_id}: {labels}")

        # Define a mapping of event IDs to event names based on your dataset
        import mne
        event_id_to_name = {
            1: 'Left Hand',
            2: 'Right Hand',
            # Add more event IDs and corresponding names as needed
        }

        # Choose the subject and session you want to analyze
        subject_id = 2
        session_id = "0"

        # Retrieve the event information for the desired subject and session
        subject_data = all_subjects_data[subject_id][session_id][session_id]
        events = mne.find_events(subject_data, initial_event=True)
        # print("printing events")
        # print(events)
        # Match the event IDs with their corresponding event names using the mapping
        event_names = [event_id_to_name[event_id] for _, _, event_id in events]

        # Print the event names and IDs
        print("Event names and IDs for Subject", subject_id, "Session", session_id)
        for event_id, event_name in zip(events[:, -1], event_names):
            print(f"Event ID: {event_id}, Event Name: {event_name}")

        # events

        import numpy as np
        from mne import Epochs
        import mne

        def extract_labels_features(all_subjects_data, event_id):
            all_labels = []
            all_features = []

            for subject_data in all_subjects_data.values():
                for session_data in subject_data.values():  # Iterate over session dictionaries
                    for raw in session_data.values():  # Iterate over raw objects
                        # Extract events from annotations
                        # # print("debug check")
                        # if raw.annotations is not None:
                        #     # Print annotation descriptions and corresponding time points
                        #     for desc, start, end in zip(raw.annotations.description,
                        #                  raw.annotations.onset,
                        #                  raw.annotations.onset + raw.annotations.duration):
                        #             # print("hello")
                        #             # print(f"Description: {desc}, Start: {start}, End: {end}")
                        # else:
                        #     print("No annotations found in the raw data.")

                        # print("debug check end")
                        # events, _ = mne.events_from_annotations(raw) # no events annotated hence use events below find events from original data
                        events = mne.find_events(raw, initial_event=True)
                        # print("Events:", events)  # Debug print
                        # print("Events:", events2)
                        # Define event IDs
                        event_dict = {'left_hand': 1, 'right_hand': 2}  # Customize as needed

                        # Create epochs based on event IDs
                        epoch = Epochs(raw, events, event_id=[1, 2], tmin=-0.1, tmax=0.7, on_missing='warn')

                        # Extract labels and features
                        labels = epoch.events[:, -1]
                        features = epoch.get_data()

                        # Append labels and features to lists
                        all_labels.append(labels)
                        all_features.append(features)

            return np.concatenate(all_labels), np.concatenate(all_features, axis=0)

        # Example usage:
        event_id = {'left_hand': 1, 'right_hand': 2}  # Define event IDs
        labels, features = extract_labels_features(all_subjects_data, event_id)

        # !pip install -q --no-cache-dir mne lightning torchmetrics

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
        plt.rcParams['axes.facecolor'] = 'lightgray'

        # labels

        import numpy as np

        # Sample array
        # labels2 = np.array([1, 1, 1, 2, 2, 2, 1, 1, 2, 2])

        # Replace 1 with 0 and 2 with 1
        labels[labels == 1] = 0
        labels[labels == 2] = 1

        # print(labels)

        EEG_CHANNEL = 69

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
                        'x': x[train_idx + 1:val_idx],
                        'y': y[train_idx + 1:val_idx],
                    }
                    self.test_ds = {
                        'x': x[val_idx + 1:],
                        'y': y[val_idx + 1:],
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

        eeg_dataset = EEGDataset(x=features, y=labels)

        # labels.shape, features.shape

        # features, labels

        class AvgMeter(object):
            def __init__(self, num=20):
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

                self.train_accuracy = Accuracy(task="binary")
                self.val_accuracy = Accuracy(task="binary")
                self.test_accuracy = Accuracy(task="binary")

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
                # print("Forward pass of model wrapper......")
                # print("shape of x is ---> ")
                # print(x.shape)
                # print(x)
                return self.arch(x)

            def training_step(self, batch, batch_nb):
                x, y = batch
                # print("In training step")
                # print("shape x and y")
                # print(x.shape)
                # print(y.shape)
                # print(y)
                y_hat = self(x)
                # print("shape of y_hat in train step")
                # print(y_hat.shape)
                loss = F.binary_cross_entropy_with_logits(y_hat, y)
                self.train_accuracy.update(y_hat, y)
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
                # print("In train epoch end")
                sch = self.lr_schedulers()
                sch.step()

                self.train_loss.append(self.train_loss_recorder.show().data.cpu().numpy())
                self.train_loss_recorder = AvgMeter()

                self.train_acc.append(self.train_acc_recorder.show().data.cpu().numpy())
                self.train_acc_recorder = AvgMeter()

            def validation_step(self, batch, batch_nb):
                x, y = batch
                # print("In validation step ")
                # print("size of x and y")
                # print(x.shape)
                # print(y.shape)
                # print("value of y")
                # print(y)
                # print("value of x")
                # print(x)
                y_hat = self(x)
                # print("shape of y_hat in validation step")
                # print(y_hat.shape)
                # print("in validation step shape of y_hat")
                # print(y_hat.shape)
                loss = F.binary_cross_entropy_with_logits(y_hat, y)
                self.val_accuracy.update(y_hat, y)
                acc = self.val_accuracy.compute().data.cpu()

                self.val_loss_recorder.update(loss.data)
                self.val_acc_recorder.update(acc)

                self.log("val_loss", loss, prog_bar=True)
                self.log("val_acc", acc, prog_bar=True)

            def on_validation_epoch_end(self):
                # print("In validation epoch end")
                self.val_loss.append(self.val_loss_recorder.show().data.cpu().numpy())
                self.val_loss_recorder = AvgMeter()

                self.val_acc.append(self.val_acc_recorder.show().data.cpu().numpy())
                self.val_acc_recorder = AvgMeter()

            def test_step(self, batch, batch_nb):
                x, y = batch
                # print("I m in test step")
                # print("shape of x and y")
                # print(x.shape)
                # print(y.shape)
                y_hat = self(x)
                # print("shape of y_hat")
                # print(y_hat.shape)
                loss = F.binary_cross_entropy_with_logits(y_hat, y)
                self.test_accuracy.update(y_hat, y)

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
                loss_img_file = "./output/deep4net/loss_plot.png"
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
                acc_img_file = "./output/deep4net/acc_plot.png"
                # fig.savefig('./output/deep4net/confusion_mat2_deep4exp.png')
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
                # print("in train dataloader of model wrapper")
                return data.DataLoader(
                    dataset=self.dataset.split("train"),
                    batch_size=self.batch_size,
                    shuffle=True,
                )

            def val_dataloader(self):
                # print("in validation dataloader of model wrapper")
                return data.DataLoader(
                    dataset=self.dataset.split("val"),
                    batch_size=self.batch_size,
                    shuffle=False,
                )

            def test_dataloader(self):
                # print("in test dataloader of model wrapper")
                return data.DataLoader(
                    dataset=self.dataset.split("test"),
                    batch_size=self.batch_size,
                    shuffle=False,
                )

            def configure_optimizers(self):
                # print("hello i m in optimizers")
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
            """Positional encoding.
            https://d2l.ai/chapter_attention-mechanisms-and-transformers/self-attention-and-positional-encoding.html
            """
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
            def __init__(self, eeg_channel, dropout=0.1):
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
                    TransformerBlock(eeg_channel * 2, 2, eeg_channel // 8, dropout),
                    TransformerBlock(eeg_channel * 2, 2, eeg_channel // 8, dropout),
                )

                self.mlp = nn.Sequential(
                    nn.Linear(eeg_channel * 2, eeg_channel // 2),
                    nn.ReLU(True),
                    nn.Dropout(dropout),
                    nn.Linear(eeg_channel // 2, 1),
                )

            def forward(self, x):
                # print("In forward step of EEG classification")
                # print("Shape of x")
                # print(x.shape)
                # print(x)
                x = self.conv(x)  # problem is here
                # print("After conv layer")
                # print(x.shape)
                x = x.permute(0, 2, 1)
                x = self.transformer(x)
                # print("After transformer layer")
                # print(x.shape)
                x = x.permute(0, 2, 1)
                x = x.mean(dim=-1)
                x = self.mlp(x)
                # print("after mlp layer")
                # print(x.shape)
                # print("shape of x")
                # print(x.shape)
                # print("x array")
                # print(x)
                return x

        MODEL_NAME = "EEGClassificationModel"
        model = EEGClassificationModel(eeg_channel=EEG_CHANNEL, dropout=0.125)

        MAX_EPOCH = 4
        BATCH_SIZE = 64
        LR = 0.0009
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

