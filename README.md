# braindiagnose
An independent study to assess the usage of deep learning artificial intelligence (convolutional neural networks) in the field of medical image analysis (brain tumors and cancers). 

This specific artificial intelligence model is able to evaluate brain magnetic resonance imaging (MRI) scans to detect the presence of brain tumors / cancers.

Created by Azhaan Salam
# Setup
1. [Install Python3](https://realpython.com/installing-python/)

2. Download this repository and cd into it.

3. On your terminal, execute `pip install -r requirements.txt` to install the packages necessary to using this program.

4. To create and train the model, execute `python trainbrain.py train test model.h5` and wait for training to complete.
    
    The model is being trained on the `train` data split, and its accuracy is being evaluated on the `test` data split. Afterwards, it gets saved into a file named `model.h5`
    
    When prompted, confirm if you want to save the model or not.

The model is now setup and ready for use.
# Usage

1. Obtain the scans you would like to get a diagnosis of and load it onto your computer.

2. Create an `input` folder, where images will be fed in, and an `output` folder, where images will go when evaluated.

3. To execute the diagnosis program, run `python braindiagnosis.py model.h5 [INPUT_FOLDER] [OUTPUT_FOLDER]`, replacing `[INPUT_FOLDER]` and `[OUTPUT_FOLDER]` with the path to your input and output folders.

4. Move the scans into the `input` folder

The program will then output its diagnosis, and move each image into the output folder.



-azh
