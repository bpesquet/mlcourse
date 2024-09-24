---
marp: true
---

<!-- Apply header and footer to first slide only -->
<!-- _header: "[![Bordeaux INP logo](../ensc_logo.jpg)](https://www.bordeaux-inp.fr)" -->
<!-- _footer: "[Baptiste Pesquet](https://www.bpesquet.fr)" -->

# Convolutional Neural Networks

<!-- Show pagination, starting with second slide -->
<!-- paginate: true -->

---

## Learning objectives

- Discover the general architecture of convolutional neural networks.
- Understand why they perform better than plain neural networks for image-related tasks.

---

## Architecture

---

### Justification

The visual world has the following properties:

- Translation invariance.
- Locality: nearby pixels are more strongly correlated
- Spatial hierarchy: complex and abstract concepts are composed from simple, local elements.

Classical models are not designed to detect local patterns in images.

[![Visual world](images/visual_world.png)](https://youtu.be/shVKhOmT0HE)

---

[![Topological structure](images/topological_structure.png)](https://youtu.be/shVKhOmT0HE)

[![From edges to objects](images/edges_to_objects.png)](https://youtu.be/shVKhOmT0HE)

---

### General CNN design

[![General CNN architecture](images/cnn_architecture.png)](https://developers.google.com/machine-learning/practica/image-classification/convolutional-neural-networks)

---

### The convolution operation

Apply a **kernel** to data. Result is called a **feature map**.

[![Convolution with a 3x3 filter of depth 1 applied on 5x5 data](images/convolution_overview.gif)](https://developers.google.com/machine-learning/practica/image-classification/convolutional-neural-networks)

---

![Convolution example](images/convolution_example.jpeg)

---

### Convolution parameters

- **Filter dimensions**: 2D for images.
- **Filter size**: generally 3x3 or 5x5.
- **Number of filters**: determine the number of feature maps created by the convolution operation.
- **Stride**: step for sliding the convolution window. Generally equal to 1.
- **Padding**: blank rows/columns with all-zero values added on sides of the input feature map.

---

### Preserving output dimensions with padding

[![Preserving output dimensions with padding](images/2d_convol.gif)](https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d)

---

#### Valid padding

Output size = input size - kernel size + 1

[![Valid padding](images/padding_valid.png)](https://youtu.be/shVKhOmT0HE)

---

#### Full padding

Output size = input size + kernel size - 1

[![Valid padding](images/padding_full.png)](https://youtu.be/shVKhOmT0HE)

---

#### Same padding

Output size = input size

[![Valid padding](images/padding_same.png)](https://youtu.be/shVKhOmT0HE)

---

### Convolutions inputs and outputs

[![Convolution inputs and outputs](images/conv_inputs_outputs.png)](https://youtu.be/shVKhOmT0HE)

---

### 2D convolutions on 3D tensors

- Convolution input data is 3-dimensional: images with height, width and color channels, or features maps produced by previous layers.
- Each convolution filter is a collection of *kernels* with distinct weights, one for every input channel.
- At each location, every input channel is convolved with the corresponding kernel. The results are summed to compute the (scalar) filter output for the location.
- Sliding one filter over the input data produces a 2D output feature map.

---

[![2D convolution on a 32x32x3 image with 10 filters](images/conv_image.png)](https://towardsdatascience.com/applied-deep-learning-part-4-convolutional-neural-networks-584bc134c1e2)

---

[![2D convolution over RGB image](images/2D_conv_over_rgb_image.png)](https://stackoverflow.com/a/44628011/2380880)

---

### Activation function

- Applied to the (scalar) convolution result.
- Introduces non-linearity in the model.
- Standard choice: ReLU.

---

### The pooling operation

- Reduces the dimensionality of feature maps.
- Often done by selecting maximum values (*max pooling*).

[![Max pooling with 2x2 filter and stride of 2](images/maxpool_animation.gif)](https://developers.google.com/machine-learning/practica/image-classification/convolutional-neural-networks)

---

#### Pooling result

[![Pooling result](images/pooling_result.png)](https://youtu.be/shVKhOmT0HE)

---

#### Pooling output

[![Pooling with a 2x2 filter and stride of 2 on 10 32x32 feature maps](images/maxpooling_image.png)](https://towardsdatascience.com/applied-deep-learning-part-4-convolutional-neural-networks-584bc134c1e2)

---

### Training process

Same principle as a dense neural network: **backpropagation** + **gradient descent**.

[Backpropagation In Convolutional Neural Networks](https://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/)

---

### Interpretation

- Convolution layers act as **feature extractors**.
- Dense layers use the extracted features to classify data.

![A convnet](images/convnet.jpeg)

---

[![Feature extraction with a CNN](images/representation_learning.png)](https://harishnarayanan.org/writing/artistic-style-transfer/)

---

[![Visualizing convnet layers on MNIST](images/keras_js_layers.png)](https://transcranial.github.io/keras-js/#/mnist-cnn)

---

## History

### Humble beginnings: LeNet5 (1988)

![LeNet5](images/lenet5.jpg)

---

[![Bell Labs demo](images/lecun_bell_labs.jpg)](https://www.youtube.com/embed/FwFduRA_L6Q)

---

### The breakthrough: ILSVRC

- [*ImageNet Large Scale Visual Recognition Challenge*](http://image-net.org/challenges/LSVRC/)
- Worldwide image classification challenge based on the [ImageNet](http://www.image-net.org/) dataset.

![ILSVRC results](images/ILSVRC_results.jpg)

---

### AlexNet (2012)

Trained on 2 GPU for 5 to 6 days.

![AlexNet](images/alexnet2.png)

---

### VGG (2014)

![VGG16](images/vgg16.png)

---

### GoogLeNet/Inception (2014)

- 9 Inception modules, more than 100 layers.
- Trained on several GPU for about a week.

![Inception](images/google_inception.jpg)

---

### Microsoft ResNet (2015)

- 152 layers, trained on 8 GPU for 2 to 3 weeks.
- Smaller error rate than a average human.

![ResNet](images/resnet_archi.png)

---

![Deeper model](images/deeper_model.jpg)

---

### Depth: challenges and solutions

- Challenges
  - Computational complexity
  - Optimization difficulties

- Solutions
  - Careful initialization
  - Sophisticated optimizers
  - Normalisation layers
  - Network design
