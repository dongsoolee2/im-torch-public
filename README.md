# im-torch

This ongoing project is to train Deep Learning (DL) models for optical imaging
data (or any other data) and to interpret PyTorch models using interpretability
algorithms including Integrated Gradients (IG) method.

Dongsoo Lee

Baccus Lab, Stanford University


### Input
The input is naturalistic video data projected to the retina

Size of the input tensor is: (Time, window (40 frames), x, y)

### Output
The output is neural physiological responses (calcium responses)

Size of the output tensor is: (Time, # of cells or ROIs)

### Model
Convolutional Neural Networks (CNN) models are defined in `models.py`.
