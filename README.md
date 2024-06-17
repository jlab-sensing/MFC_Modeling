# MFC_Modeling
This is the Github repository for the code used in the paper Deep Learning for Predicting Microbial Fuel Cell Energy Output. All relevant datasets and models are stored at https://huggingface.co/datasets/adunlop621/Soil_MFC/tree/main

Type1.ipynb, Type2.ipynb, and SNN.ipynb contained the code used to train all all the models in the paper, as well as load pretrained models and run time series rolling validation.

Pretrained models have the following naming conventions: model type (Type 1, 2, 1A, 1B, etc.), time horizon (3min, 5min, etc.), and quantile (quant5, quant50, etc.). For example, a Type 1A model predicting lower bound values in time intervals of 15 minutes would be called type1A_15min_quant5.

The same naming convention holds for SNN models, with the model type simply being snn, ie. snn_15min_quant50






