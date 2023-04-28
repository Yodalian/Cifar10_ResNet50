#############Data Preparation
The CIFAR-10 train set was split into train and validation sets with 45,000 and 5,000 images respectively.The data was preprocessed using data augmentation techniques, including resizing the images to 224x224, random flipping, and normalization.

#############Training porcess
I implemented my own ResNet50 architecture in resnet.50, which achieved an accuracy of approximately 86% on the validation set.Next, I used a pretrained ResNet50 model on the train set and validation set. The model was trained for 20 epochs with an initial learning rate of 0.001. The learning rate was reduced by half every 5 epochs.

#############Results
The trained model achieved a validation accuracy of 97%. To test the model, you can run the test.py script. After running the script, the 5 runs average accuracy was 96.83%.