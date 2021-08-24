# Planet: Understanding the Amazon from Space 🌳🦌
***By Nhan Phan, November 2019, as an entry to the competition [Planet: Understanding the Amazon from Space](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/data) by Kaggle.***

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1s8iFtj7D4D0BNlsR7P9hvfzsqV8XhjTD?authuser=3#scrollTo=okbCcmr-AKeN)

![](https://storage.googleapis.com/kaggle-competitions/kaggle/6322/logos/header.png)

Every minute, the world loses an area of forest the size of 48 football fields. And deforestation in the Amazon Basin accounts for the largest share, contributing to reduced biodiversity, habitat loss, climate change, and other devastating effects. But better data about the location of deforestation and human encroachment on forests can help governments and local stakeholders respond more quickly and effectively.

This analysis uses Deep Learning to classify the spatial images of the Amazon forest taken by the satilite. And from that, it hopes to shed a light on understanding how the forest has change naturally and manually. Thus, help preventing deforestation.

The dataset is acquired from the Kaggle competition in 2016: https://www.kaggle.com/c/planet-understanding-the-amazon-from-space/data

The dataset contains more than 40.000 images, taken by Planet using sattelites.



> Planet, designer and builder of the world’s largest constellation of Earth-imaging satellites, will soon be collecting daily imagery of the entire land surface of the earth at 3-5 meter resolution. While considerable research has been devoted to tracking changes in forests, it typically depends on coarse-resolution imagery from Landsat (30 meter pixels) or MODIS (250 meter pixels). This limits its effectiveness in areas where small-scale deforestation or forest degradation dominate.

<center><img src="https://storage.googleapis.com/kaggle-competitions/kaggle/6322/media/planet.png" width=300'></center>



### ☆ **RESULT**
The project successfully got the score of 0.90 on the official test set.

|  | THIS PROJECT | WINNER |
|:--:|:--:|:--:|
| **Score (F-Beta)** | 0.90 | 0.93 |

Training information:

|  | Loss | F-Beta Score |
|:--:|:--:|:--:|
| **Train** | 0.09 | 0.90 |
| **Validation** | 0.11 | 0.89 |




### **☆ CHALLENGE**
Several key learnings undercovered through the analysis:

1. **Multi-label:** Each image is labeled with multiple tags (at least 2, at max 9). The tags fall into 17 categories, which are the forest landscape types. Since the tags in each label are mutually exclusive, they are treated as multiple binary classification problems. Thus, `binary cross-entropy` are chosen to be the loss function. 

2. **Imbalance:** The dataset is severely imbalance with tags like Primary or Agriculture appear in 90% of the dataset. While other tags like Blooming or Conventional Mine can only be seen in less than 500 observations (even less than 100 for Burn Down).

    - To tackle the problem of imbalance dataset, evaluation metrics has to be chosen carefully. In the first base-line experiment, the model was totally bias toward the major tags. It predicts the major tags to appear in every data and almost never made a prediction with the minor tags. 

    - `F2` is chosen to be the main metrics to evaluate the training. It watches over the harmonic mean between the Precision and Recall while favors Recall specifically. In other word, it is the attempt to reduce the number of False Negative, where the model fails to identify the absence of a tag. 

3. **Optimization:** 400.000 images, a CNN model, and Google Colab's limited resource do not seem to mix well together. The training was slow at first and interupted often. Several improvements, mostly on the Tensorflow pipeline, were conducted to speed up the training:

    - Using **TFRecord** to convert the raw images into byte-like data to reduce the amount of time spending on reading data from their paths. 
    - Using [**tf.data.Dataset**](https://www.tensorflow.org/guide/data_performance) with `shuffle`, `map`, `batch`, `prefetch` to optimize the reading data process by redistributing the tasks for agents to work concurrently, thus, avoid bottleneck. An attempt to use `cache` was also made but failed due to the limited RAM. 

4. **Processing image with Tensorflow:** The dataset contains images in JPG - RGBA. The built-in decode function `tf.io.decode_jpeg` only works on 1 or 3-channel image. Attempt on encoding a JPG RGBA image returns black black and black. We need a tensorflow encoding function to work in this part because the pipeline is built entirely on Tensor for the optimization purpose. 

    - To tackle the problem, the raw images were first read by Matplotlib then converted into byte-like and wrote into TFRecords. When reading the data from TF Record, instead of using the built-in decode image function, we use `tf.io.parse_tensor` following with reshaping.