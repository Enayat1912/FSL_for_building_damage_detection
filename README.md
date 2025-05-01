## Few-Shot Learning for Post-Earthquake Building Damage Detection
This repository contains code and resources used in the master's thesis titled: #Few-Shot Learning for Post-Earthquake Building Damage Assessment Using Metric-Based and Transfer Learning Methods

## Objective 
This study is concerned with developing a framework that uses FSL including metric-based 
methods, and transfer learning to perform multi-class post-earthquake building damage 
assessment. As we are dealing with multi-class problems in this study, the available dataset is 
highly imbalanced where one class has multiple times more samples than other classes. 
Additionally, as deep learning requires huge amounts of data, this study also faces the challenge 
of limited data. 
This project investigates how Few-Shot Learning (FSL) methods—particularly Prototypical Networks, and Transfer Learning using ResNet50 and EfficientNetB7—can be used to classify earthquake-induced building damage into four categories:

## No damage

## Minor damage

## Major damage

## Destroyed

The workflow combines remote sensing, deep learning, data augmentation, and geospatial post-processing techniques to generate damage predictions and visualization maps from satellite imagery.

| Model                      | Type                 
|----------------------------|---------------------- |
| Prototypical Networks      | Metric-based Learning | 
| ResNet50                   | Transfer Learning     | 
| EfficientNetB7             |Transfer Learning      | 

## Results
Best model: Prototypical Networks

Accuracy: ~63% on 4-class damage classification

Insight: ProtoNets performed best at detecting Destroyed buildings



## Data and Sources 
Data Source: [https://xview2.org/](https://xview2.org/)
Original code source: https://github.com/DIUx-xView/xView2_baseline 

Focus Area: Post-disaster satellite imagery from Mexico City Earthquake 2017

Image Resolution: 1024×1024 pixels

Classes: No Damage, Minor Damage, Major Damage, Destroyed

🔁 Workflow

![image](https://github.com/user-attachments/assets/54fc8b5b-2515-4c63-9e34-198f0fc34920)


## Dependencies
1.Python ≥ 3.8

2.TensorFlow / Keras

3. OpenCV

4. Shapely

5. NumPy / Pandas / Matplotlib

6. PIL (Pillow

## Citation
If you use this code or concept in your work, please cite:

Enayatullah Meskinyaar, Few-Shot Learning for Post-Earthquake Building Damage Assessment Using Metric-Based and Transfer Learning Methods, Master’s Thesis, NOVA Information Management School, Universidade Nova de Lisboa, 2025.
