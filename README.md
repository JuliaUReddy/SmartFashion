# SmartFashion: Detailed Garment Analysis via Detection and Multi-Label Regression

## Overview

SmartFashion is a machine learning research project focused on detailed analysis of clothing garments through clothing body-part detection and multi-input, multi-output regression.

The system analyzes clothing images at the body-part level and predicts numerical appearance attributes for different regions of a garment. These attributes describe characteristics such as neck depth, shoulder breadth, chest fit, waist fit, hip fit, arm breadth, and garment length.

The long-term goal of SmartFashion is to help users find clothing that better matches their desired appearance. The completed system could eventually combine clothing analysis with human body analysis to recommend garments that help users achieve a desired visual proportion or style.

This repository focuses on the **clothing analysis component** of the SmartFashion research project.

---

## Problem Statement

Online clothing shopping can make it difficult for users to identify garments that match their desired fit and appearance.

Traditional clothing categories such as "shirt," "dress," or "jacket" do not fully describe the visual characteristics of a garment. Two garments in the same category may have significantly different shoulder structures, sleeve lengths, necklines, waist fits, or hip proportions.

SmartFashion explores a more detailed approach by representing garments through body-centric appearance attributes.

For example, a garment can be described using values representing:

- Narrow or broad shoulders
- Shallow or deep necklines
- Tight or loose chest fit
- Tight or loose waist fit
- Narrow or wide hip fit
- Short or long sleeves
- Tight or loose arm breadth
- Short or long garment lengths

These numerical representations provide a structured way to analyze and compare garments.

---

## Project Goals

The clothing analysis component of SmartFashion was developed with the following goals:

1. Detect important clothing body regions within garment images.
2. Extract individual clothing body-part regions from detected bounding boxes.
3. Develop a multi-input, multi-output regression model for detailed garment analysis.
4. Predict body-centric appearance regression values for individual clothing regions.
5. Evaluate model performance using regression metrics including Mean Absolute Error (MAE).
6. Explore perceived prediction accuracy using L1, L2, and L3 evaluation categories.

The eventual research goal is to combine clothing analysis with a separate human body analysis system to enable personalized clothing recommendations.

---

## My Contribution

My work focused on developing the **clothing analysis component** of the SmartFashion system.

My contributions included:

- Manually collecting and curating a dataset of **1,072 clothing images**.
- Preparing and labeling the clothing dataset using **Roboflow**.
- Developing the clothing body-part object detection component.
- Training and evaluating a **MobileNetV2-based object detection model**.
- Developing a multi-input, multi-output clothing regression model using TensorFlow.
- Using **EfficientNetV2-B0** as a pretrained CNN feature extractor for image-based features.
- Developing the data and image branches used by the regression model.
- Predicting 11 clothing body-centric appearance attributes.
- Evaluating regression performance using MAE.
- Analyzing regression predictions using L1, L2, and L3 perceived accuracy categories.

The human body analysis component was developed as a parallel research effort. The eventual SmartFashion system is intended to combine the clothing and human body analysis components.

---

# System Architecture

The SmartFashion clothing analysis pipeline consists of two primary stages:

1. **Clothing Body-Part Detection**
2. **Clothing Body-Part Regression**

The detection stage identifies important regions of a clothing image. The detected regions are then used by the regression system to predict detailed appearance attributes.

![SmartFashion System Architecture](Architecture/01_Smartfashion_System_Architecture.png)

---

# Dataset

A dataset of **1,072 clothing images** was manually collected from various clothing sources and prepared for model development.

The dataset was annotated using **Roboflow**.

The object detection component identifies eight clothing-related classes:

- Neck
- Shoulders
- Chest
- Waist
- Hips
- Legs
- Arms
- Whole

The dataset was divided into training, validation, and testing subsets for model development and evaluation.

### Dataset Example

![Clothing Body Parts Detection](Dataset/Clothing_Bodyparts_Detection.png)

### Clothing Regression Labeling

![Clothing Regression Labeling](Dataset/Clothing_Regression_Labeling.png)

### Regression Value Representation

![Clothing Regression Values](Dataset/Clothing_Regression_Values.png)

---

# Model Architecture

The SmartFashion clothing regression model uses a **multi-input, multi-output architecture**.

The model consists of two primary branches:

### 1. Image Branch

The image branch processes clothing body-part sub-images using a pretrained **EfficientNetV2-B0** CNN.

The pretrained network is used as a feature extractor to obtain visual features from each clothing body-part image.

### 2. Data Branch

The data branch processes numerical information associated with the detected clothing regions, including bounding-box information.

The outputs from the image and data branches are combined and passed through fully connected layers to generate the final regression predictions.

![SmartFashion Multibranch Regression Architecture](Architecture/02_SmartFashion_Multibranch_Regression_Architecture.png)

### EfficientNetV2 Feature Extractor

![EfficientNetV2 Feature Extractor](Architecture/03_EfficientNetv2_Feature_Extractor.png)

### Regression Data Branch

![Regression Data Branch Architecture](Architecture/04_Regression_DataBranch_Architecture.png)

### Regression Dense Layers

![Regression Dense Layers Architecture](Architecture/05_Regression_DenseLayers_Architecture.png)

---

# Clothing Body-Part Detection

The first stage of the SmartFashion clothing analysis pipeline is object detection.

The detection model identifies clothing body regions within an image using bounding boxes.

The eight detection classes are:

| Class |
|---|
| Neck |
| Shoulders |
| Chest |
| Waist |
| Hips |
| Legs |
| Arms |
| Whole |

A **MobileNetV2-based architecture** was used because of its relatively lightweight design and suitability for resource-constrained environments, including potential future mobile deployment.

The object detection model was trained using the curated clothing dataset.

### Detection Examples

![Detection Example 1](Results/ObjectDetection/01_ClothingBodyParts_Detection_Example1.png)

![Detection Example 2](Results/ObjectDetection/02_ClothingBodyParts_Detection_Example2.png)

### Detection Results

![Detection Summary](Results/ObjectDetection/03_ClothingBodyParts_Detection_Summary.png)

### COCO Evaluation Metrics

![COCO Evaluation Metrics 1](Results/ObjectDetection/04_COCO_Evaluation_Metrics1.png)

![COCO Evaluation Metrics 2](Results/ObjectDetection/05_COCO_Evaluation_Metrics2.png)

### MobileNetV2 Training Loss

![MobileNetV2 Training Loss](Results/ObjectDetection/06_MobileNetv2_Training_Loss.png)

---

# Clothing Body-Part Regression

Following clothing body-part detection, the detected clothing regions are used as inputs to the multi-input, multi-output regression model.

The regression model predicts **11 body-centric appearance attributes**.

Each regression value ranges from approximately **1.0 to 9.0**, where the interpretation depends on the specific clothing attribute.

A value closer to 1 generally represents a smaller, tighter, shorter, or shallower characteristic, while a value closer to 9 generally represents a broader, looser, longer, or deeper characteristic.

For clothing body parts that are not present in a garment, a value of **-1** is used to indicate a missing attribute.

## Regression Attributes

| # | Attribute |
|---|---|
| 1 | Neck Depth |
| 2 | Shoulder Breadth |
| 3 | Chest Breadth |
| 4 | Waist Breadth |
| 5 | Hips Breadth |
| 6 | Legs Breadth |
| 7 | Legs Length |
| 8 | Upper Arm Left Breadth |
| 9 | Upper Arm Right Breadth |
| 10 | Arm Left Length |
| 11 | Arm Right Length |

### Regression Scale Examples

- **Shoulder Breadth:** approximately 1.0 represents minimal shoulder structure, while higher values represent broader or puffier shoulders.
- **Chest Breadth:** approximately 1.0 represents a tightly fitted chest, while higher values represent a looser or baggier fit.
- **Waist Breadth:** approximately 1.0 represents a tightly fitted waist, while higher values represent a looser fit.
- **Hips Breadth:** approximately 1.0 represents a tightly fitted hip region, while higher values represent a looser, wider, or more voluminous fit.
- **Legs Length:** lower values represent shorter garments, while higher values represent longer garments.
- **Arm Length:** lower values represent shorter sleeves, while higher values represent longer sleeves.

---

# Regression Results

The regression model was evaluated using **Mean Absolute Error (MAE)** as the primary regression metric.

MAE measures the average absolute difference between the predicted regression value and the ground-truth value.

Because the regression values range from 1 to 9, additional perceived accuracy categories were used to analyze prediction performance.

- **L1 — Fair to Okay:** Prediction is within the most acceptable range of the ground truth.
- **L2 — Fair to Poor:** Prediction has a larger deviation from the ground truth.
- **L3 — Poor:** Prediction has a substantial deviation from the ground truth.

### Training Loss

![Regression Training Loss](Results/Regression/01_Regression_Training_Loss.png)

### Mean Absolute Error Results

![MAE Neck Shoulder](Results/Regression/02_MAE_Neck_Shoulder.png)

![MAE Chest Waist](Results/Regression/03_MAE_Chest_Waist.png)

![MAE Hips Legs Breadth](Results/Regression/04_MAE_Hips_LegsBreadth.png)

![MAE Legs Breadth Upper Left Arm Breadth](Results/Regression/05_MAE_LegsBreadth_UpperLeftArmBreadth.png)

![MAE Upper Right Arm Breadth Left Arm Length](Results/Regression/06_MAE_UpperRightArmBreadth_LeftArmLength.png)

![MAE Right Arm Length](Results/Regression/07_MAE_RightArmLength.png)

### Regression Summary

![Regression Summary](Results/Regression/08_Regression_Summary.png)

### Perceived Prediction Accuracy

![L1 Fair to Okay](Results/Regression/09_Regression_L1_FairToOkay.png)

![L2 Fair to Poor](Results/Regression/10_Regression_L2_FairToPoor.png)

![L3 Poor](Results/Regression/11_Regression_L3_Poor.png)

---

# Technologies Used

- **Python**
- **TensorFlow**
- **Google Colab**
- **Roboflow**
- **MobileNetV2**
- **EfficientNetV2-B0**
- **Convolutional Neural Networks (CNNs)**
- **Object Detection**
- **Multi-Input Multi-Output Regression**

---

# Repository Structure

```text
SmartFashion/
│
├── Architecture/
│   ├── 01_Smartfashion_System_Architecture.png
│   ├── 02_SmartFashion_Multibranch_Regression_Architecture.png
│   ├── 03_EfficientNetv2_Feature_Extractor.png
│   ├── 04_Regression_DataBranch_Architecture.png
│   └── 05_Regression_DenseLayers_Architecture.png
│
├── Dataset/
│   ├── Clothing_Bodyparts_Detection.png
│   ├── Clothing_Regression_Values.png
│   └── Clothing_Regression_Labeling.png
│
├── Notebooks/
│   ├── 01_Regression_Dataparsing.ipynb
│   ├── 02_ONEARM_NEWMediaPipeline_ObjectDetector.ipynb
│   └── 03_SmartFashion_MultiInput_MultiOutput_Regression.ipynb
│
└── Results/
|   ├── ObjectDetection/
|   └── Regression/
│
├── .gitignore
└── README.md
Note:
Large binary files, including the trained TensorFlow Lite model and full capstone report, are maintained separately and are not included in this repository due to repository size considerations.