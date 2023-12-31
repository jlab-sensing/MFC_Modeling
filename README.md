# MFC_Modeling
This project trains a random forest to predict the power output of an MFC based on the current environmental conditions (Temp, VWC, Soil EC).

The main code for the project is located in dataloader.ipynb. It be opened as a Jupyter Notebok in Google Colab by clicking the Binder link below, and selecting Open in Colab at the top of the screen once the notebook opens.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/jlab-sensing/MFC_Modeling/HEAD?labpath=dataloader.ipynb)

The code can then be executed by running each cell in order from top to bottom.

The code can also be accessed by navigating to dataloader.ipynb in the repository and clicking "Open in Colab", but the notebook may not always render properly in Github.

Running Code Locally:

To run this project on your local computer, download and execute dataloader.py. The packages required for the code can be found in requirements.txt, and the neccessary data can be found in the Jlab directory, in stanfordMFCDataset.zip, by following this link:

https://drive.google.com/drive/folders/1cCIrRNQCjUkE9vV7oOfoW-dt9SuHpU9-

Note that certain parts of the code can take 2-3 minutes to excecute, namely loading the power data and training the model. Sorting the data by timestamp and merging the data can take up to 30 seconds.





