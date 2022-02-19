# [Nome do Projeto]

## Preparing the Database

To download the EEG Imagery/Motor Database from Physionet, used in the article, go to [](https://physionet.org/content/eegmmidb/1.0.0/) and proceed with one of the steps listed under *Access the files*. You also need to include the path of the folder in which the database is stored in the **database_path** variable in the **EEG_biometric.py** file.

While the tests described in the article were being done, errors regarding the imcompatibility of numpy and pyedflib packages were encountered. To work around this problem, the .edf files were converted to .csv, using the **create_csv_database_from_edf** command from the **utils.py** file.

e.g.: To convert all .edf files stored in a folder with path './Dataset/' in .csv files stored in a folder with path './Dataset_CSV/', the syntax would be:

`create_csv_database_from_edf('./Dataset/','./Dataset_CSV/', 109)`

A list containg specific electrode channels can also be specified using the **channels** parameter, and a list containg all the avaliable channels is present on the **loader.py** file.

While the conversion from .edf files to .csv files were being done, the used versions of numpy and pyedflib packages were 1.20.2 and 0.1.18, respectively. After the conversion was done, the used version of numpy was changed to 1.19.5 and pyedflib was uninstalled.

If you want to use a database present in another folder, don't forget to update the file path in the **database_path** variable in the **EEG_biometric.py** file.

## Running the application

To run the application, execute the **EEG_biometric.py** file followed by the **-train** argument, the tasks that will be used for training/validation, the **-test** argument and the tasks that will be used for testing. The tasks need to be specified by number, and can be seen in the **EEG_biometric.py** file.

e.g.: `python EEG_biometric.py -train 1 -test 2`

Hyperparameters and parameters related to pre-processing and data augmentation can also be seen and changed in the **EEG_biometric.py** file. 

## Using Data Generators

Using data augmentation before training the model can result in a huge amount of data, especially if numerous training/validation tasks are chosen, and some machines aren't able to support it. In that sense, you can try using the **-datagen** argument, which will use data generators to feed cropped data into the model while it's training.

Firstly, you need to specify a folder path in the **processed_data_path** variable in the **EEG_biometric.py** file, where a folder named **processed_data** will be created (if it doesn't exist yet). When running the application, data from tasks that were not stored yet will be pre-processed and stored in .csv format, separated by task.

**NOTE:** If you changed the parameters for loading, pre-processing the data or using data augmentation, and pretend to use a task that already has a folder inside the **processed_data** folder, delete that folder.

## Other Optional Arguments
If you already trained a model and want to skip it's training, you can use the **--nofit** argument. By default, the model weights will be loaded from a file named **model_weights.h5**.

If you don't want the model to run in identification mode, you can use the **--noimode** argument. Similiarly, if you don't want the model to run verification mode, you can use the **--novmode** argument. Note that using both these commands will make the application stop after processing the data and training the model.
