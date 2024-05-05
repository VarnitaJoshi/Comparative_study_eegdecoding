from braindecode.datasets import MOABBDataset
from numpy import multiply
from braindecode.preprocessing import(Preprocessor, exponential_moving_standardize, preprocess)
from braindecode.preprocessing import create_windows_from_events
import torch
from braindecode.models import ATCNet
from braindecode.util import set_random_seeds
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from skorch.callbacks import LRScheduler
from skorch.helper import predefined_split
import time
from braindecode import EEGClassifier
from sklearn.metrics import confusion_matrix
from braindecode.visualization import plot_confusion_matrix
from skorch.callbacks import EarlyStopping
import sys


subject_id = [1, 2, 3, 4, 5, 6, 7, 8, 9]

for subject in subject_id: 
    print("hello in subject, b4 data load")
    dataset = MOABBDataset(dataset_name = "BNCI2014001", subject_ids = [subject])
    print("hello after data load")
    # low_cut_hz = 0.5
    # high_cut_hz = 150
    factor_new = 1e-3
    init_block_size = 1000
    factor = 1e6

    preprocessors =[
        Preprocessor('pick_types', eeg = True, meg = False, stim = False),
        Preprocessor(lambda data: multiply(data, factor)),
        # Preprocessor('filter', l_freq = low_cut_hz, h_freq = high_cut_hz),
        Preprocessor(exponential_moving_standardize, factor_new = factor_new,
                    init_block_size = init_block_size)
    ]

    preprocess(dataset, preprocessors, n_jobs = -1)



    trial_start_offset_seconds = -0.5

    sfreq = dataset.datasets[0].raw.info['sfreq']
    assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])
    trial_start_offset_samples = int(trial_start_offset_seconds *  sfreq)
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples = trial_start_offset_samples,
        trial_stop_offset_samples = 0,
        preload = True
    )

    splitted = windows_dataset.split('session')
    train_set = splitted['0train']
    valid_set = splitted['1test']



    print("b4 cuda load")
    cuda = torch.cuda.is_available()
    device = 'cuda' if cuda else 'cpu'
    if cuda:
        print("GPU Available")
        torch.backends.cudnn.benchmark = True
    else:
        print("GPU not available")
    print("after cuda load")

    seed = 20200220
    set_random_seeds(seed = seed, cuda = cuda)

    n_classes = 4
    classes = list(range(n_classes))
    n_chans = train_set[0][0].shape[0]
    input_window_samples = train_set[0][0].shape[1]


    print("model definition")
    model = ATCNet(
        n_chans,
        n_classes, conv_block_dropout=0.4, att_dropout=0.6, tcn_dropout=0.5)

    print("after model definition")

    if cuda:
     model = model.cuda()

    print("b4 log file path")

    # log_file_path = "./output/atcnet_within_subject/training_log_withinsubject.txt"
    log_file_path = f"./output/atcnet_within_subject/training_log_subject_{subject}.txt"
    print("after log file path")
    sys.stdout = open(log_file_path,"w")
    start_time = time.time()


    lr = 0.0009
    weight_decay = 0
    start_time = time.time()

    batch_size = 30
    n_epochs = 100

    early_stopping = EarlyStopping(patience = 18, monitor = 'valid_loss', lower_is_better = True,  threshold=0.0001)

    clf = EEGClassifier(
        model,
        criterion = torch.nn.NLLLoss,
        optimizer = torch.optim.AdamW,
        train_split = predefined_split(valid_set),
        optimizer__lr = lr,
        optimizer__weight_decay = weight_decay,
        batch_size = batch_size,
        callbacks = ["accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max = n_epochs - 1)), early_stopping],
        device = device,
        classes = classes
    )

    print("before model fit")
    _ = clf.fit(train_set, y = None, epochs = n_epochs)

    print("After model fit")
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    # sys.stdout.close()

    # ===============================================Plotting ==========================================
    results_columns = ['train_loss', 'valid_loss', 'train_accuracy', 'valid_accuracy']
    df = pd.DataFrame(clf.history[:, results_columns], columns = results_columns,
                    index = clf.history[:, 'epoch'])
    df = df.assign(train_misclass = 100 - 100 * df.train_accuracy,
                valid_misclass = 100 - 100 * df.valid_accuracy)

    fig, ax1 = plt.subplots(figsize = (8, 3))
    df.loc[:, ['train_loss', 'valid_loss']].plot(
        ax = ax1, style = ['-', ':'], marker = 'o', color ='tab:blue', legend = False, fontsize = 14)
    ax1.tick_params(axis = 'y', labelcolor = 'tab:blue', labelsize = 14)
    ax1.set_ylabel("Loss", color = 'tab:blue', fontsize = 14)

    ax2 = ax1.twinx()
    df.loc[:, ['train_misclass', 'valid_misclass']].plot(
        ax = ax2, style = ['-', ':'], marker = 'o', color = 'tab:red', legend = False)

    ax2.tick_params(axis = 'y', labelcolor = 'tab:red', labelsize = 14)
    ax2.set_ylabel("Misclassification Rate [%]", color ='tab:red', fontsize = 14)
    ax2.set_ylim(ax2.get_ylim()[0], 85)
    ax1.set_xlabel("Epoch", fontsize = 14)

    handlers = []
    handlers.append(Line2D([0], [0], color ='black', linewidth = 1, linestyle = '-', label = 'Train'))
    handlers.append(Line2D([0], [0], color = 'black', linewidth = 1, linestyle = ':', label = 'Valid'))
    plt.legend(handlers, [h.get_label() for h in handlers], fontsize = 14)
    plt.tight_layout()
    plt.savefig(f'./output/atcnet_within_subject/Lossgraph_subject_{subject}.png')

    # plt.savefig('./output/atcnet/Lossgraph_su.png')


    y_true = valid_set.get_metadata().target
    y_pred = clf.predict(valid_set)

    confusion_mat = confusion_matrix(y_true, y_pred)

    label_dict = windows_dataset.datasets[0].window_kwargs[0][1]['mapping']

    labels = [k for k, v in sorted(label_dict.items(), key = lambda kv: kv[1])]

    fig = plot_confusion_matrix(confusion_mat, class_names = labels, figsize = (30, 15))
    fig.savefig(f'./output/atcnet_within_subject/confusion_mat_subject{subject}.png')
    print("last line of code ")



