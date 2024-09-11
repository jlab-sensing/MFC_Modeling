# MFC_Modeling
This is the Github repository for the code used in the paper Deep Learning for Predicting Microbial Fuel Cell Energy Output. 

All relevant datasets and models are stored in Hugging Face at https://huggingface.co/datasets/adunlop621/Soil_MFC/tree/main

In order for the code to run properly, it is neccesary to download the datasets from Hugging Face and place them in the same directory as the code with the following file names. Dataset 1, as defined in the paper, is stored in stanfordMFCDataset.zip, which expands into the directory rocket4. Dataset 2 is stored in the ucscMFCDataset directory. In order to load the pretrained models used in the paper, it is neccesary to download the trained_models directory.

[Type1.ipynb](https://github.com/jlab-sensing/MFC_Modeling/blob/main/Type1.ipynb), [Type2.ipynb](https://github.com/jlab-sensing/MFC_Modeling/blob/main/Type2.ipynb), and [SNN.ipynb](https://github.com/jlab-sensing/MFC_Modeling/blob/main/SNN.ipynb) contained the code used to train all all the models in the paper, as well as load pretrained models and run time series rolling validation. 

Pretrained models have the following naming conventions: model type (Type 1, 2, 1A, 1B, etc.), time horizon (3min, 5min, etc.), and quantile (quant5, quant50, etc.). For example, a Type 1A model predicting lower bound values in time intervals of 15 minutes would be called type1A_15min_quant5.

The same naming convention holds for SNN models, with the model type simply being snn, ie. snn_15min_quant50






