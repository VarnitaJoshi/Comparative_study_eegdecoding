# Comparative_study_eegdecoding

## output Folder ---> it contains all the outputs generated(.txt files) by the following code files below).
## Code Files
1. EEGnetv1try_exp.py
2. atc_copy2.py
3. atc_withinsubject.py
4. deep4net_exp.py
5. deep4net_withinsubject.py
6. eeginceptionmi_exp.py
7. eeginception_within_subject.py
8. eegnetv1.py
9. eegnetv1_within_subject.py
10. shallownet_exp.py
11. shallownet_within_subject.py
### All this .py files i have reffered from Braindecode website, all the models are prebuilt in Braindecode library. For all this models i have used BCI competition Iv 2a dataset on them. All outputs in outputs folder are for BCI competition IV 2a dataset.

## CNN+Transformer+MLP model
## Code files 
1. cho17_cnn.py ----> In this file i have used this transformer model for CHO17 dataset this dataset i had already downloaded on my system or another way is to download dataset using moabb library in my code i had given path to my local system where i had stored the dataset but if you run this code in your system, first download the dataset, you can search on google how to download or another way is to use the moabb library, you can use it download the dataset it, see this https://moabb.neurotechx.com/docs/datasets.html to know how to download data using moabb.
2. schirrmeister_cnn.py-------> same story as above and output is in file schirrmeister_cn.txt.
3.  BCI_CNN+TF+MLP.ipynb ---> for this code i had download and saved dataset in y google drive, so you can do the same only then code will run seccessfully if you rerun it.
4.  Physionet_EEG_Motor_Imagery_Classification_Using_CNN_Transformer_and_MLP.ipynb------> this is the main code of the github repository that i reffered which was-----> https://github.com/reshalfahsi/eeg-motor-imagery-classification


## Box plot and violin plot file 

1. boxplot.ipynb -----> in this file i have used two .csv files a.csv and ab.csv to produce boxplot and violin plots

2. a.csv----> i have created this file myself by just copying the outputs from the .txt files present in output folder, now as a.csv produced within subject boxplot i have created this file using the outputs from the witin_subject folder for each model. See to get better understanding first try to understand what box plot is representing it is representing the accuracy distribution for each model.
3. ab.csv ----> this file i created by copying the content that i needed from the .txt files in the outputs folder.
4. 





    


