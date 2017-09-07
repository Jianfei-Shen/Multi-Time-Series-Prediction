# Multi-Time-Series-Prediction
This repository serves to investigate ways of forecasting sequences or time series using different tools. 

## Two culture of prediction 
The method of prediction varies when different sequences need to be predicted. Especially, there exists a huge distinction between "Strong Predictable Sequence"(SPS) and "Weak Predictable Sequence"(WPS).

SPS usually contains strong signal while the structure of the signal is hard to capture. An example of SPS in Natural Language Processing where you need to predict next word based on the context. In this case, people tend to apply "heavy tools" (like LSTM, a kind of Neuaral Network) since they are able to approximate the complex architecture of the signal.

WPS, on the other hand, is composed of huge amount of noise with a little bit signal. Heavy tools usually result in overfitting. If you apply deep learning to stock market, all you are going to get are the features of Brownian Motion. There are two prevalent ways on handling WPS. You can either predict with light but robust tools (like linear models, SVM ..) or you can built a statistical model to get the future distribution by simulation. 

(To be continued...) 
