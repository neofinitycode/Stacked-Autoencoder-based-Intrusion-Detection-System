# Stacked Autoencoder based Intrusion Detection System using One-Class Classification

Link to <a href="https://ieeexplore.ieee.org/document/9377069">Research Paper</a>

## Introduction
Intrusion Detection System (IDS) is a vital security service which can help us with timely detection. This IDS has become an critical component of network security for this intrusion detection system which is used to monitor network traffic and produce warnings when attacks occur. The data collected by the IDS is obtained from different sources in a computer network and then analyzed for the monitoring purpose. Attacks analyzed by the IDS involve	both intrusions (external attacks) and misuse (internal attacks).Hence, it is important to address this issue when such malicious packets has been sent and must be classified correctly. For this we will be using a deep learning model called autoencoder to classify the packets since, deep learning models can recognise the complex patterns very effectively.

## One-Class Classification
One class classification defines the algorithm is trained over single labeled instances. The motivation for such training methods occurs when the data includes less labeled data of a particular class in such cases where the label of those instances are uncertain it is necessary to make a representation in such a way that the model identifies the pattern inside the labeled data and classify the unlabeled data accordingly. It is similar to a problem of two-class classification, where in this case the two classes has special meaning. Wherein, the two classes are called the target and the outlier/anomaly class. Since, the outlier data is not present readily with the dataset as the outlier might not have occurred in reality hence, containing little or no samples for such anomalies like machine malfunction. Use case of this training method helped in intrusion detection, as there exist the similar case where attacks or unauthorized access to the network are not frequently occurred resulting into detection of anomaly.

## Autoencoder
Autoencoders are the special type of artificial neural network which has the ability to learn the effective representation of the input data called as codings can be seen in figure given below. The coding layer has much less dimensions as compared to the input data, that makes autoencoders in performing dimensionality reduction. Moreover, autoencoders can be a feature detectors and has importance in case of unsupervised training of the model. Additionally, autoencoders can generate new instances similar to the training data. Working of autoencoders is very simple just by copying the input to their output. An autoencoder analyzes the input and converts them to the internal representation and again expands which looks similar to the input. An autoencoder consists of two parts: encoder helps in conversion to internal representation network given by equation, next to encoder there is decoder which converts the internal representation to the outputs seen in equation.
There are 4 hyperparameters that we need to set before training an autoencoder:
- Code size: number of nodes in the middle layer. Smaller size results in more compression.
- Number of layers: the autoencoder can be as deep as we like. In the figure above we have 2 layers in both the encoder and decoder, without considering the input and output.
- Number of nodes per layer: the autoencoder architecture we’re working on is called a stacked autoencoder since the layers are stacked one after another. Usually stacked 
<div align=center><img src=https://github.com/ghatoleyash/Deep-AutoEncoder-IDS/blob/master/autoencoder_architecture.png></div>

## Stacked Autoencoder
The stacked auto-encoders is a dense neural network version of Autoencoders as can be seen in the figure given below. It has been majorly used for feature learning. The input vectors are fed to the leftmost Autoencoder. Subsequently, the output representations are passed on to the next layers. The same procedure is repeated until all the auto-encoders are trained. The reconstructed input of the rightmost layer in figure is the output of the Stacked Autoencoders. The unsupervised training can explore huge instances of unlabeled data to prevail a good weight initialization for the neural network than conventional random initialization.
<div align=center><img src=https://github.com/ghatoleyash/Deep-AutoEncoder-IDS/blob/master/stacked_autoencoder_architecture.png></div>
<div align=center><img src=https://github.com/ghatoleyash/Deep-AutoEncoder-IDS/blob/master/sae_equations.png></div>
Where, X is the input vector, W is the weight vector, b relates bias to every node, H1 and H2 associated to the vector output of 1st and 2nd hidden layer respectively, E related to the output of coding layer, X' is reconstructed input vector
<div align=center><img src=https://github.com/ghatoleyash/Deep-AutoEncoder-IDS/blob/master/cost_function.png></div>
Where X is the input vector, X' is the reconstructed input vector, J is the cost function 

## Batch Normalization
There are two major problems that exist while training the deep neural network.
- Internal Covariance Shift: It occurs when there is a change in the input distribution to the network. As the input distribution changes, hidden layers try to adapt to the new distribution. This Internal covariance shift changes the denser regions of input distribution in successive batches during training, the weights modified into the network for the prior iteration are indifferent. This explains why Internal covariance shift can be such a problem that slows down the training process. Due to process slows down, algorithm takes a long time to converge to a global minima 
- Vanishing Gradient: While computing the gradients with the help of partial derivatives. The change that the gradient produced is very little as the model is trained deeper into the layer. This results into modification of weights present at initial layers very less in short the weights are not updated effectively to learn the complex pattern. Hence, the algorithm meets the saturation point from where no improvement in convergence of error function is shown.

Both problems can be solved using batch normalization to some extent. It is achieved through a normalization step that fixes the means and variances in the dense layers. Idea behind usage of normalized training sets with the help of various techniques such as min-max and standard scaler can be applied in batch normalization. In this, hidden layers are trained from the outputs received by activation function of the previous layers which can result into skewed distribution. Hence, to fix such issue normalizing the inputs generated inside the layers is also important so that weights are updated equally without any bias. It does the scaling to output of the layer, specifically performing standardization by considering the statistics such as mean and standard deviation per mini batch. In order to improve the weights which cannot be necessarily correspond to the standard-scaler. It is required to train the two new parameters which are scaling and shifting of the standardized layer inputs as part of the training process

<div align=center><img src=https://github.com/ghatoleyash/Deep-AutoEncoder-IDS/blob/master/bn_equations.png></div>
Where, m is the batch size, x is the vector of the batch B, mu is the mean of batch B, sigma_square is the variance, x'(i) is the normalized value with zero mean and unit variance, epsilon is the smoothing term

<div align=center><img src=https://github.com/ghatoleyash/Stacked-Autoencoder-based-IDS/blob/master/scaling%20_shifting_equation.png></div>
Where, gamma is the scaling parameter, beta is the shifting parameter


## Experimental Setup
In this section, we discuss details regarding the datasets, preprocessing, and evaluation measures used to assess the efficacy of deep autoencoder model.

#### Datasets
The dataset used for model evaluation is KDD Cup 1999 Dataset products. It includes 126208 records and total of 42 attributes. There are total of 33 continuous attributes and 9 continuous attributes. The attributes such as protocol_type defines the type of protocol used by the packet to get into the network and service which corresponds to network service on the destination, e.g. http, telnet, etc. 

#### Preprocessing
Contained duplicate values removal of those duplicate records and then all of the categorical variables were converted using the one-hot encoding which eventually made total 117 attributes for the dataset. Hence, the remaning continuous attributes were normalized using the standard scaler formula.

#### Validation and Testing
The model is trained based on single class since our model has to detect the intrusion which is very rare. The training set only included the labels with normal connections. But in the validation and test set both labels of the data are included. THe total dataset was split into 60% of training set, 20% of cross validation set and remaining 20% was test set.

#### Implementation details

Layer | Shape | Parameters
------|-------|-----------
Input Layer | 118 | 0
Encoder | 64 | 7616
Batch Normalization | 64 | 256
Coding Layer | 32 | 2080
Batch Normalization | 32 | 128
Decoder | 64 | 2112
Batch Normalization | 64 | 256
Output Layer | 118 | 7670

As the architecture given by the above table, where three hidden layers added which comprises of encoding, coding and decoding layer. ReLU was included as the activation function at hidden layers since, it provided the best results as compared to other functions such as tanh and sigmoid. Reason being, there is confined interval where sigmoid and tanh will have nonzero values. To be specific function's derivatives turned zero as the output of function reaches extreme end of left and right side, which is absurd to make backward pass. In addition, the model includes batch normalization at each of the hidden layers to reduce the problem mentioned earlier.
Mean squared error was used as error metric and adam optimizer was used while converging the cost function. The output layer was configured to be linear which was equivalent to the number of features given at the input layer. 

<div align=center><img src=https://github.com/ghatoleyash/Deep-AutoEncoder-IDS/blob/master/violin_plot_n.png></div>

The figure above outlines the effect on gradients with and without batch normalization. Layers 2 and 3 feel the involved normalized gradients as opposed to those without BN. First, weights initialized by the model were retrieved and then fetched after the training weights had been completed. Second, we measured the difference between the initial weights and the final weights so that we could obtain the total weight adjustment. Finally, a mean was determined for each of the nodes to which the gradients had been allocated which eventually gives us an overall gradient distribution for each node. Returning to the figure gradients at layer 2 without batch normalization, there were longer tails resulting in more altered nodes with gradients on the negative side compared to those with positive gradients. Lastly, linear function was worked on the output layer to create a representation replicated close to the input

#### Experimental Results
In this section, we analyze the results of experiments performed with deep autoencoder model.

![GitHub Logo](/recall&precisionVSthreshold_n.png)
![GitHub Logo](/confusion_matrix_n.png)

- threshold =  0.3
- Accuracy =  98.17%
- Recall =  95.22%
- Precision =  99.18%
- F1-Score =  97.15%


## References
- Géron, Aurélien. Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow: Concepts, tools, and techniques to build intelligent systems. O'Reilly Media, 2019.
- Tax, David Martinus Johannes. "One-class classification: Concept learning in the absence of counter-examples." (2002): 0584-0584.
- Chu, Xu, et al. "Data cleaning: Overview and emerging challenges." Proceedings of the 2016 International Conference on Management of Data. 2016.
- Ioffe, Sergey, and Christian Szegedy. "Batch normalization: Accelerating deep network training by reducing internal covariate shift." arXiv preprint arXiv:1502.03167 (2015).
- Rajendran, Sreeraj, et al. "Crowdsourced wireless spectrum anomaly detection." IEEE Transactions on Cognitive Communications and Networking 6.2 (2019): 694-703.
- Ling, Jie, and Chengzhi Wu. "Feature Selection and Deep Learning based Approach for Network Intrusion Detection." 3rd International Conference on Mechatronics Engineering and Information Technology (ICMEIT 2019). Atlantis Press, 2019.
