# Non-linear neurons with human-like apical dendrite activations                                                                                    

## Summary

In order to classify linearly non-separable data, neurons are typically organized into multi-layer neural networks that are equipped with at least one hidden layer. Inspired by some recent discoveries in neuroscience, we propose a new neuron model along with a novel activation function enabling the learning of non-linear decision boundaries using a single neuron. We show that a standard neuron followed by the novel apical dendrite activation (ADA) can learn the XOR logical function with 100\% accuracy. Furthermore, we conduct experiments on five benchmark data sets from computer vision, signal processing  and natural language processing, i.e. MOROCO, UTKFace, CREMA-D, Fashion-MNIST, and Tiny ImageNet, showing that ADA and the leaky ADA functions provide superior results to Rectified Linear Units (ReLU), leaky ReLU, RBF and Swish, for various neural network architectures, e.g. one-hidden-layer or two-hidden-layer multi-layer perceptrons (MLPs) and convolutional neural networks (CNNs) such as LeNet, VGG, ResNet and Character-level CNN. We obtain further performance improvements when we change the standard model of the neuron with our pyramidal neuron with apical dendrite activations (PyNADA).

## Code

This repo provides the official implementation of "Non-linear neurons with human-like apical dendrite activations". The provided code can be used to reproduce resulys on Fashion-MNIST and MOROCO.

## Prerequisites
- numpy==1.15.4
- opencv_python==4.1.1.26
- scikit_image==0.15.0
- tensorflow_gpu==1.12.0 
- scikit_learn==0.22.1

## Citation

BibTeX:

    @article{georgescu2020non,
      title={Non-linear neurons with human-like apical dendrite activations},
      author={Georgescu, Mariana-Iuliana and Ionescu, Radu Tudor and Ristea, Nicolae-Catalin and Sebe, Nicu},
      journal={arXiv preprint arXiv:2003.03229},
      year={2020}
    }

## You can send your questions or suggestions to: 
raducu.ionescu@gmail.com

### Last Update:
Novermber 19th, 2021
