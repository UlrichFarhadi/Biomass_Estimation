# RGB-D Camera Biomass Estimation using AI

This project aims to estimate biomass using artificial intelligence techniques, specifically employing Convolutional Neural Networks (CNN) with a ResNet backbone. The implementation is based on PyTorch and PyTorch Lightning. The approach involves a late fusion multimodal regression network, where depth and RGB data are processed through separate ResNet50 branches. The outputs of these branches, both 1000x1 layers, are concatenated and fed into the regression head. The final layer of the regression head produces the biomass prediction.

## Models

We have developed four distinct models utilizing the aforementioned approach, each focusing on estimating different parameters of lettuce growth:

1. Lettuce Fresh Weight
2. Lettuce Dry Weight
3. Lettuce Diameter
4. Lettuce Height

## Dataset

The dataset used for this project is sourced from the Tencent Online Lettuce Trait Estimation Challenge. It comprises a total of 341 images. Given the limited size of the dataset, we have employed data augmentation techniques to enhance its diversity and improve model generalization.

### Dataset Source

The dataset can be accessed from the [Tencent Online Lettuce Trait Estimation Challenge](http://www.autonomousgreenhouses.com/online_challenge/index), last accessed on October 20, 2022.


Feel free to explore the code, adapt it to your needs, and contribute to its development!

## License

This project is licensed under the [MIT License](LICENSE).

