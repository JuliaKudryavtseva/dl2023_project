# Medical Image Classification with ResNets, DenseNets, Efficient Net and ViT
Medical image classification plays an increasingly important role in healthcare, especially in diagnosing, treatment planning and disease monitoring. However, the lack of large publicly available datasets with annotations means it is still very difficult, if not impossible, to achieve clinically relevant computer-aided detection and diagnosis (CAD). In recent years, deep learning models have been shown to be very effective at image classification, and they are increasingly being used to improve medical tasks. Thus, this project aims to explore the use of different convolutional neural network (CNN) architectures for medical image classification. Specifically, we will examine the performance of 6 different CNN models (ResNet-18, ResNet-152, DenseNet-121, DenseNet161, Efficient Net and ViT) on datasets of blood cell images and chest X-ray images.
### Prerequisites
### Datasets
We will use two datasets for our experiments:
1) [Blood Cell Images](https://www.kaggle.com/datasets/paultimothymooney/blood-cells): This dataset contains 12,500 augmented images of blood cells with 4 different cell types, namely Eosinophil, Lymphocyte, Monocyte, and Neutrophil. 
We use this dataset for the multi-class classification problem.
2) [Random Sample of NIH Chest X-ray Dataset](https://www.kaggle.com/datasets/nih-chest-xrays/sample?select=sample_labels.csv): This dataset contains 5,606 chest X-rays from random patients with 15 classes (14 diseases, and one for "No findings")
We use this dataset for the multi-label classification problem.
### Loss function
Multi-class classification refers to the categorization of instances into precisely one class from a set of multiple classes. So, the commonly used loss function is cross-entropy loss. \
Multi-label classification involves instances that can belong to multiple classes simultaneously. Binary cross-entropy loss is commonly employed in this scenario.
### Models
1)	ResNet, or Residual Network, is a deep convolutional neural network architecture that was introduced in 2015 by He et al. ResNets work by using residual connections to skip over layers in the network. This allows the network to learn more complex features without becoming too deep and overfitting to the training data.
2)	DenseNet, or Densely Connected Network, is another deep convolutional neural network architecture that was introduced in 2016 by Huang et al. DenseNets are similar to ResNets, but they use dense connections to connect all of the layers in the network. This allows the network to learn more global features and improve the accuracy of the model.
3)	EfficientNet is a family of convolutional neural network architectures that were introduced in 2019 by Tan et al. EfficientNets are designed to be efficient in terms of both accuracy and computational resources. They achieve this by using a combination of techniques, including compound scaling, squeeze-and-excitation blocks, and autoML.
4)	Vision Transformer, or ViT, is a deep learning model that was introduced in 2020 by Dosovitskiy et al. Vision Transformers are based on the transformer architecture, which was originally developed for natural language processing (NLP).
### Results
The results of our experiments are shown in the table below.
1)	For multi-class classification problem:
* ResNet18 and Efficient Net are the 2-most efficient models, as it performs well in all metrics. 
* DenseNet121 and DenseNet161 are unstable, accuracy and F1 are 0.87 and 0.91 correspondingly for both nets. 
* ViT showed SOTA results (accuracy: 0.89, F1: 0.89) with stable learning. 
2)	For multi-label classification problem:
* DenseNets and Efficient Net showed satisfied results with 0.52 accuracy and F1 score
* DenseNets and ResNets were unstable, while EfficientNet_b0 showed stable learning.
* ViT with metrics, as other algorithms and showed stable learning.
### Conclusion
Overall, Efficient Net is the most efficient models for both multi-class and multi-label classification problems. They perform well in all metrics and show stable learning. ViT performs as other algorithms in terms of metrics. However, there are significantly less params to train. That makes this architecture to be alternative for ConvNets in the future.
