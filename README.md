# Retinal_Disease_Detection
Pathology Detection on Medical Images

## Problem Statement
Can we distinguish diabetic retinopathy vs glaucoma

## Data ( https://www.kaggle.com/c/vietai-advance-course-retinal-disease-detection/overview
)

Explanation of the data set: The training data set contains 3435 retinal images that represent multiple pathological disorders. The patholgy classes and corresponding labels are: included in 'train.csv' file and each image can have more than one class category (multiple pathologies). The labels for each image are

-opacity (0), 
-diabetic retinopathy (1), 
-glaucoma (2),
-macular edema (3),
-macular degeneration (4),
-retinal vascular occlusion (5)
-normal (6)
The test data set contains 350 unlabelled images.

## Details
1. [Data Exploration and cleaning](./data_cleanse.ipynb) 
2. [Model building with architecture and hyper params](./hparam_tuning.ipynb)
3. [Prediction Outcome](./prediction_analysis.ipynb)
4. [Heatmap Visualizations](./outcome_visualization.ipynb)


