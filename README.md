# Deep AUC Maximization (DAM) for Medical Image Classification
This repository contains the code and methodology for using Deep AUC Maximization (DAM) for medical image classification. DAM is a learning paradigm for deep neural networks that aims to maximize the Area Under the Curve (AUC) score of the model on a dataset.

## Introduction :
In this project, we aim to improve the application of DAM to small medical image datasets. Although DAM has shown significant success on large-scale datasets, it tends to overfit smaller training data. We focus on enhancing its generalization for small datasets in medical image classification.

## Methodology :
We used the LibAUC library and conducted experiments on seven medical image classification tasks derived from the MedMNIST website: BreastMNIST, PneumoniaMNIST, ChestMNIST, NoduleMNIST3D, AdrenalMNIST3D, VesselMNIST3D, and SynapseMNIST3D.

## Training Strategy :
We first combined the datasets and trained a ResNet-18 model on the 2D datasets (BreastMNIST, PneumoniaMNIST, ChestMNIST) with specific hyperparameters and data augmentation techniques. The model was then fine-tuned on individual datasets. For the 3D datasets, we followed a similar process but without data augmentation.

## Hyperparameter Selection :
The hyperparameters were chosen through a grid search over loss functions (CrossEntropyLoss, BCEWithLogitsLoss, and NLLLoss), optimizers (SGD, Adam, RMSprop), learning rates (0.1, 0.01, 0.001, 0.0001, 0.00001), and momentum values (0.9). The best hyperparameters for each dataset were obtained as follows:

**PneumoniaMNIST:** SGD optimizer, learning rate 0.001, momentum 0.9, CrossEntropyLoss function.

**BreastMNIST:** Adam optimizer, learning rate 0.001, momentum 0.9, CrossEntropyLoss function.

**ChestMNIST:** Adam optimizer, learning rate 0.001, momentum 0.9, BCEWithLogitsLoss function. 


**Nodulemnist3d:** SGD optimizer, learning rate 0.1, momentum 0.9, momentum 0.9, CrossEntropyLoss function.

**Adrenalmnist3d:** Adam optimizer, learning rate  1e-05, momentum 0.9, CrossEntropyLoss function.

**Vesselmnist3d:** RMSprop optimizer, learning rate 0.001, momentum 0.9, CrossEntropyLoss function.

**synapsemnist3d:**  Adam optimizer, learning rate  0.001, momentum 0.9, CrossEntropyLoss function.

## Results : 
**PneumoniaMNIST:** Training accuracy 0.9909, Validation AUC 0.9943, Validation accuracy 0.9733, Test AUC 0.9776, Test accuracy 0.9199

**BreastMNIST:** Training accuracy 0.9872, Validation AUC 0.9524, Validation accuracy 0.8846, Test AUC 0.9121, Test accuracy 0.8526 


For the 3D datasets:

**NoduleMNIST3D:** Training accuracy 90.8054%, Validation AUC 0.8724, Validation accuracy 82.8942%, Test AUC 0.8469, Test accuracy 81.7532%

**VesselMNIST3D:** Training accuracy 100.0000%, Validation AUC 0.8717, Validation accuracy 88.0208%, Test AUC 0.7835, Test accuracy 86.1257%

**AdrenalMNIST3D:** Training accuracy 87.7104%, Validation AUC 0.8396, Validation accuracy 81.6892%, Test AUC 0.8156, Test accuracy 79.5396%

**SynapseMNIST3D:** Training accuracy 84.7154%, Validation AUC 0.6003, Validation accuracy 70.6215%, Test AUC 0.5418, Test accuracy 71.8750%

## Discussion :
Our approach shows that DAM can effectively improve medical image classification in small datasets. However, the performance on 3D datasets isn't as high as on 2D datasets, indicating the need for more work to optimize for 3D medical image classification.

## Conclusion :
This work offers a new approach to enhancing the application of the DAM method on small medical image classification tasks. It shows promise, and future work may further improve its performance. The use of deep learning techniques in medical imaging holds significant potential for revolutionizing the field of radiology and improving patient care, even with small datasets.
