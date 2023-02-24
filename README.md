# Analysis of OCT Images to detect retinopathy ENESJ - IMO

## In this branch files
### [ReducedImagesTrain.ipynb](ReducedImagesTrain.ipynb)

Train models ([Xception](https://arxiv.org/abs/1610.02357), [ResNet50](https://arxiv.org/abs/1512.03385) and [OpticNet](https://ieeexplore.ieee.org/document/8999264)) with the [OCT2017](https://data.mendeley.com/datasets/rscbjbr9sj/2) dataset partitioning and training with fractions of the complete train set (from 1% to 100%) and evaluated performance with test and validation accuracy. All nets were trained with pretrained parameters (see sources) and with random weights.

**NOTE:** All runs were added through git commits, check them for more information.

### [Results.csv](Results.csv)
All trained models performance results.
Columns:
 - `model`: Name of the trained model.
 - `train set images`: Number of images used in the train set (all taken from the main train set).
 - `pretrained`: If the model parameters were pretrained or not.
 - `pretrained dataset`: If the model was pretrained, the dataset used to train it.
 - `epochs`: Number of epochs trained.
 - `batch size`: Batch size used while training.
 - `learning rate`: Learning rate used while training.
 - `optimizer`: Optimizer algorithm used while training.
 - `training time (seconds)`: Amount of time (in seconds) the model training process took.
 - `train accuracy`: Accuracy on the last training epoch over the train set.
 - `train loss`: Cross entropy loss on the last training epoch over the train set.
- `validation accuracy`: Accuracy on the last training epoch over the validation set.
 - `validation loss`: Cross entropy loss on the last training epoch over the validation set. 
 - `test accuracy`: Accuracy on the last training epoch over the test set.
