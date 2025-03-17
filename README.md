# Comparison of Forecasting Capabilities of Deep Learning Models with Varying Time Series Resolution for Accurate Karst Spring Discharge Predictions
By: Xiao Xia Liang, Philippe Renard, Erwan Gloaguen, Julien Straubhaar, and Maxime Claprood

This paper presents predictions of karst spring discharges using 3 different DNN models. The resolutions of the time series are tested along with forecasting capabilities of the models. 

The models used are [GRU and LSTM auto-encoder](https://arxiv.org/pdf/1406.1078) and [Graph WaveNet](https://arxiv.org/pdf/1906.00121)

The [Graph WaveNet codes](https://github.com/sshleifer/Graph-WaveNet) adapted for this work is the improved version from [Shleifer et al., (2019)](https://arxiv.org/pdf/1912.07390)

### Data

The resample data can be found [here](https://www.dropbox.com/scl/fo/r511pkjkm0tpz8vv1uadk/AN7W8UzDGwDdzNUquF3pOj0?rlkey=pnntlyhiiwpy6id59i2djs9d7&st=4fsws3k5&dl=0).

The training and testing data, and trained models of the auto-encoders can be found [here](https://www.dropbox.com/scl/fo/ktv1ys9vex9541808xwzr/AIFaTaIpL1Gfch0O8mHYlmE?rlkey=jvt1ehljzfbp0mudl1qxzs4ln&st=h6h8s9lg&dl=0).

The training and testing data, and trained models of the Graph WavwNet can be found [here](https://www.dropbox.com/scl/fo/ylnhaz27x6v5zvcprhmtm/ACJw8_jMrwzH0-JvrX2mN3o?rlkey=qaizwdichofixnxor1le0bxsi&st=24weh4iz&dl=0).


### Package Requirements
- python 3
- pandas (2.1.1)
- numpy (1.26.4)
- scikit-learn (1.4.0)
- scipy (1.13.1)
- torch (2.1.2+cu121)
- tensorflow (2.14.0)

Need help? 
If you are a student and need help, contact me at xl3138 (at) princeton (dot) edu. I will be happy to help you. 
