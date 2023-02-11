The attached code is a DeepLabV3+ binary segmentation algorithm for malignant prostate cancer glands. 

The model is already trained and your task is to (1) properly run inference in a new image set and (2) calculate relevant performance metrics

This code does not require a GPU, CPU inference has been tested on standard windows and linux systems

Please return the following within 1 week:
1. modified code
2. AI output masks
3. csv with performance metrics for each image

Environment requirements file is provided, alternatively to set up the env with the following:
pip install numpy
pip install fastai
pip install semtorch