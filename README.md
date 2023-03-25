# braindiagnose
An independent study to assess the usage of deep learning artificial intelligence (convolutional neural networks) in the field of medical image analysis (brain tumors and cancers).

Conducted by Azhaan Salam
# Setup
1. [Install Python3](https://realpython.com/installing-python/)

2. Download this repository and cd into it

3. On your terminal, execute `pip install -r requirements.txt` to install the packages necessary to using this program

4. To create and train the model, execute `python trainbrain.py train test model.h5` and wait for training to complete.
    When prompted, confirm if you want to save the model or not.

The model is now setup and ready for use.
# Usage

1. Obtain the image you would like to get a diagnosis of and load it onto your computer

2. To execute the diagnosis program, run `python braindiagnosis.py model.h5 [YOUR_IMAGE_HERE]`, replacing `[YOUR_IMAGE_HERE]` with the path to your image.

The program will then output its diagnosis.
