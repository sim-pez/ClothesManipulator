## Introduction

This program extracts a feature from a cloth image and then applies some given manipulations. The output is the feature of the manipulated cloth image. After that, the image will be retrieved from the dataset.

### Example

We have the following image:

<p float="left" align="center">
  <img src="docs/cloth1.jpg" width="20%"  />
</p>

You can give an input to our model to remove the hood and the zip up from the image. The output will be:

<p float="left" align="center">
  <img src="docs/cloth2.jpg" width="20%"  />
</p>

### Our Contribution

This is a work based on amazon's [ADDE-M](https://github.com/amzn/fashion-attribute-disentanglement). The main difference is the support of multiple manipulations, because the original work only supports a single manipulation. 
The main advantage of our technique is data augmentation. In the amazon's method the query-target couples of training set are only with distance 1, but in our case the couples can be chosen from every image wich distance is < N

# Installation
1. Download [amazon's ADDE-M repo](https://github.com/amzn/fashion-attribute-disentanglement)
2. Clone our repo in the same folder
3. Install requirements with `pip install -r requirements.txt`

# Dataset
We used Shopping100k: [contact the author](https://sites.google.com/view/kenanemirak/home) of the dataset to get access to the images

After downloading that, you can create random couples using:

```
python3 f_dataset_gen.py
```

you can choose the maximum distance N of the couples inside the script

# Train
After created dataset (check section above), run:
```
python3 f_train.py
```

It is possibile to modify some parameters in `parameters.py`

# Evaluation
To evaluate the model use:
```
python3 f_eval.py
```

# Test Results
Here is a comparison between our model wrt amazon's one (ADDE-M) and other state-of-the-art models. We obtained a slightly better performances.

<p float="left" align="center">
  <img src="docs/results.png" width="50%"  />
</p>

