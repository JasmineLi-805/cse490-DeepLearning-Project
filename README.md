# Project: Use Saliency Map for Region Proposal

The project is implemented in a forked [Pytorch Image Model repository](https://github.com/JasmineLi-805/pytorch-image-models).
Since this README is specifically for the course project, I have created a separate repository for the project related portion.

VIDEO GOES HERE (probably): Record a 2-3 minute long video presenting your work. One option - take all your figures/example images/charts that you made for your website and put them in a slide deck, then record the video over zoom or some other recording platform (screen record using Quicktime on Mac OS works well). The video doesn't have to be particularly well produced or anything.

## Introduction

As introduced in the lectures, image classification networks such as R-CNN and YOLO have a region proposal phase that generates thousands of bounding boxes for later process. However, considering that background often consists of a great portion of an image, many of the bounding boxes are not useful in the classification task, thus there is inefficiency in this method.

In a paper by Yohanandan et al. (mentioned below), the authors suggest that locating the objects of interest could be done with a rather small network and low resolution input. This project would utilize the findings of this paper and try to replace the region proposal phase with a “salience map” generation, and train an image classification model end-to-end.


## Related Work

- Yohanandan et al., [Fast Efficient Object Detection Using Selective Attention](https://arxiv.org/abs/1811.07502): This is the inspiration of the project, where the author uses a convolutional network to identify the objects of interest in an image. Different from the paper, this project does not train the salience mapping on its own but attached a classifier after it to train the process end-to-end.
- Jang et al., [Gumbel Softmax](https://arxiv.org/abs/1611.01144): Gumbel softmax is used to generate a probability for each image in containing an object of interest. A higher probability means the classifier is more likely to identify an object of interest from the image.

## Approach

The implementation of this project uses the [Pytorch Image Model](https://github.com/rwightman/pytorch-image-models) Github Repo where it has a series of published neural networks implemented and a tested training/validation workflow. The following section describes the main changes and additional code added for this project.

### Dataset

This project was intended to be tested on the ImageNet dataset, however, due to limitations in computational power and time, a small subset of ImageNet, the [imagenette](https://github.com/fastai/imagenette) dataset, is used instead. It has a total of 10 classes and approximately 13.4k images for training and validation.

### Data Preprocessing

In the design of this project, each image is cropped into a few different smaller images to determine which part is the most significant in classification task. The cropped images would be downsized and turned into grayscale for the region proposal phase, while the original sized cropped image would be weighted on the results of regional proposal and used for classification.
To meet the goal of having different images of different resolutions, a new SalienceImageDataset is added to the collection of datasets in [dataset.py](https://github.com/JasmineLi-805/pytorch-image-models/blob/56bbba2d0466cd55f53f39b017c577bf183b17c2/timm/data/dataset.py#L161).

There are two types of transforms, `downsize_transform` and `original_transform`. The `downsize_transform` is used to create images for region proposal. It crops the images, applies grayscale, downsizes the cropped image, and finally pads the downsized image with 0s. The `original_transform` only crops the original images and outputs a stack of tensors. The results from the two transforms is then stacked into one 4 dimensional tensor. The FiveCrop function is used to crop the images.

The following diagram shows how an image of a dog is processed by the dataset.

![Dataset Example](dataset_example.png)

### Model

The model also consists of two parts. The first part is a stack of 3 convolutional layers with batch normalization to determine which image is the most useful in classification. The second part is ResNet18 for classification. The model is implemented in [scresnet.py](https://github.com/JasmineLi-805/pytorch-image-models/blob/master/timm/models/scresnet.py) in the Pytorch Image Model repository.

In model.forward, the downsized images first passes the convolutional layers, which outputs a probability using Gumbel Softmax. Then the original-sized cropped images are scaled by the probability and summed to create one 3xWxH image. Finally, ResNet18 would perform the classification task. In this design, it is expected for the regional proposal module to assign higher probability to images more helpful for the classification, and this can be trained with the backpropagation from the classification results.

The following diagram depicts a forward pass of the model.

![Network Example](network_example.png)

### Train and Evaluation Code

This project utilizes the existing [training](link) and [evalutation](link) script in the Pytorch Image Model repository. It supports distributed training and evaluation, and the hyperparameters can be set in the execution commands.


## Results

In this project, I decided to do a comparison study on how the size 

How did you evaluate your approach? How well did you do? What are you comparing to? Maybe you want ablation studies or comparisons of different methods.

You may want some qualitative results and quantitative results. Example images/text/whatever are good. Charts are also good. Maybe loss curves or AUC charts. Whatever makes sense for your evaluation.

## Discussion

You can talk about your results and the stuff you've learned here if you want. Or discuss other things. Really whatever you want, it's your project.