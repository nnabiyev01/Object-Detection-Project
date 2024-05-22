# **Machine Learning Project**.

## **Project:**
 Machine learning project in object detection (using AI models). More specifically we want to find the empty shelves (stocked or not stocked shelves) from a super Market. The dataset is in the following link (SKU dataset): https://drive.google.com/file/d/1iq93lCdhaPUN0fWbLieMtzfB1850pKwd/edit The dataset contains 12K of photos, but I have used a smaller part of them to make the task. 
 
 ## **Requirements:**
You need to work on Jupyter Notebook,

- I have explained the business case, the dataset, the code (understanding of what you do and why) and qualitative analysis of the results. You can also use pretrained models like I've got it here.

## **Documentation:**
Is attached int the repo in pdf.

## **Annotated Data:**
- inventory_images.zip contains all the ilabeled images.
- Open source tool was used to label the the data.

## **Trained Models**
- Trained models are also included here along with.

## **Notebooks**
There are total 4 notebooks attached;
1. 3 notebook containing all the each model separately trained
2. 1 notebook containing all the models in a single file.

#### Introduction:

With the coming of age of Artificial Intelligence, everyone wants to see how they can use their
data to solve their business problems. For example, some Brick-and-Mortar stores are silently
working on how they can stay competitive by predicting empty supermarket shelves. This leads
to the idea that the store can automatically track which items are being pulled off the shelves and
take the guesswork out stocking. With the advancements and neural networks and artificial
intelligence, I wanted to see if we could predict the empty spaces on supermarket shelves, given
a limited amount of data and no initial bounding boxes to train on. This problem is already
solved when we have bounding boxes to train on.

#### Our Project
The success of deep learning to solve complex problems is not hidden from anyone these days.
Deep learning plays an important role to automate problems in all walks of life. In this document,
I have used the different deep learning-based algorithm, to detect empty inventory in grocery
stores. Usually, when we go to a grocery store, and we see a shelf that doesn’t have the product we need,
then many customers will leave without asking the store workers if they have that item. Even if
the store had that item in their warehouse. This can cause the store to lose out on potential sales
for as long as the inventory remains empty. I have used machine learning models to help stores
replenish inventory quickly so that they don’t lose customers and sales.

#### Dataset Overview
The Sku110k dataset provides 11,762 images with more than 1.7 million annotated bounding
boxes captured in densely packed scenarios, including 8,233 images for training, 588 images for
validation, and 2,941 images for testing. There are around 1,733,678 instances in total. The
images are collected from thousands of supermarket stores and are of various scales, viewing
angles, lighting conditions, and noise levels. All the images are resized into a resolution of one
megapixel. Most of the instances in the dataset are tightly packed and typically of a certain
orientation

#### Methodology
We selected a collection of detection models and pre-trained them on the SKU-110K dataset
such as the EfficientDet D1 640x640, SSD MobileNet V1 FPN 640x640, and SSDResNet50 V1 FPN from TensorFlow 2 Detection Model Zoo and Detecto Module in Pytorch. These models are useful for initialization when training on our new datasets. By comparing
the performance of these models, we have concluded that SSD-ResNet50 delivers better
performance with respect to real-time detection. We trained our model based upon the SSDResNet50 V1 FPN Architecture. The entire workflow of the SSD-ResNet50 V1 FPN
Architecture is illustrated in Figure 3. SSD with the ResNet50 V1 FPN feature extractor in its
architecture is an object detection model that has been trained on the COCO 2017 dataset. A
Momentum optimizer with a learning rate of 0.04 was used for the region proposal and classification network, and the learning rate was reduced on the plateau. As shown in Figure
3, the Feature Pyramid Network (FPN) generates the multi-level features as inputs to the SSDResNet50 Architecture. The FPN is an extractor and provides the extracted feature maps layers
to an object detector. When the model localizes any small object, it draws an object boundary  3
box around it at each location. After training the model, the testing procedure was carried out by
providing the surgical videos as input to the trained model. Afterward, we used Tensorboard
which is a suitable feature of the TensorFlow Object Detection API. It allowed us to
continuously monitor and visualize several different training/evaluation metrics when our
model was being trained. As the final step, we obtained the output video containing the
labeled surgical instruments and the assessment results along with the log file. The generated
log file records the surgical assessment, the bounding box for each laparoscopic instrument, and
the center point of each laparoscopic instrument



---------------- ---------------- ------------- ------------------
**Author: ------**
------------ ----------------- -------------- ---------------