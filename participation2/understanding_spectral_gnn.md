# Understanding Spectral Graph Neural Network

## 1 Overview

- Essentially, CNNs have taken major strides in many fields  such as translation, object reconition, recommendation systems, etc. 

- CNNs are considered euclidian data
    - standard inner products
    - subtract one vector from other vector
    - apply matrices to vectors et.c

- We can do this because of translational equivariance and invariance

- **Translational equivaraince:** A function or operation is translationally equivariant if, when the input is translated (shifted), the output is also translated by the same amount.

Convolution layers in CNNs are translationally equivariant. If you shift an image (the input) and apply a convolutional filter, the feature map (output) will shift by the same amount, but the content of the features will remain the same. This is crucial for detecting patterns regardless of their location in the input.

- **Translational Invariance:** A function or operation is translationally invariant if, after translating the input, the output remains unchanged. This means that the function "ignores" the specific location of the input features and focuses on their presence or absence.

The max pooling operation in CNNs creates translational invariance to a degree. By downsampling the feature maps, the network becomes less sensitive to the exact location of features. For example, a CNN might recognize a face in an image no matter where the face appears, due to pooling layers.








