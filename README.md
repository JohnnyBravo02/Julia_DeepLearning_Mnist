﻿# Multilayer Perceptron with Julia and FluxML

### Goal
 _Design and train a high-performing **multilayer perceptron** in Julia and FluxML that accurately classifies MNIST handwritten digits in 10 classes_

### Design
<details>
  <summary>Neural Network Architecture</summary>
  
  - Input Layer Nodes: _784_
  - Hidden Layers: _3_
  - Hidden Layer Nodes: _[25, 25, 25]_
  - Output Layer Nodes: _10_
</details>

<details>
  <summary>Hyperparameters</summary>
  
  - Learning Rate ($\alpha$): _0.1_
  - Momentum ($\psi$): _0.0001_
  - Weight Decay ($\lambda$): _0.0004_
  - Batch Size: 250
</details>

<details>
  <summary>Training</summary>
  
  - Epochs: _1000_
  - Loss Function: _Cross Entropy_
  - Optimizer: _Gradient Descent ($\alpha$, $\psi$)_
  - Regularizer: _L2 (Weight Decay)_
</details>

### Training Metrics
<details>
  <summary>Loss Log</summary>

  At Epoch 1000
  
  Training Loss: _0.006_
  
  Validation Loss: _0.194_
  
  ![LossLog](https://github.com/JohnnyBravo02/Julia_DeepLearning_Mnist/assets/140975510/33919098-4143-49f2-a038-920594de867a)
</details>

<details>
  <summary>Accuracy Log</summary>

  At Epoch 1000
  
  Training Accuracy: _99.98%_
  
  Validation Accuracy: _95.84_

  ![AccuracyLog](https://github.com/JohnnyBravo02/Julia_DeepLearning_Mnist/assets/140975510/868824bd-6e07-4a1b-91e8-134b24f8227d)
</details>

### Test
Test Accuracy: _96.46%_
