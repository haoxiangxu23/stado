# Spatio-Temporal Action Detection with Occlusion

> Pytorch implementation of [Spatio-Temporal Action Detection with Occlusion]()
> Take [Actions as Moving Points](https://arxiv.org/abs/2001.04608) (MOC) as the backbone.

<br/>

## Overview  

&emsp; Present a new action tubelet detection framework, termed as **MovingCenter Detector (MOC-detector)**, by treating an action instance as a trajectory of moving points. MOC-detector is decomposed into three crucial head branches:

- (1) **Center Branch** for instance center detection and action recognition.
- (2) **Movement Branch** for movement estimation at adjacent frames to form moving point trajectories.
- (3) **Box Branch** for spatial extent detection by directly regressing bounding box size at the estimated center point of each frame.

<br/>

## Usage

### 1. Installation
Please refer to [Installation.md](readme/Installation.md) for installation instructions.

### 2. Dataset
Please refer to [Dataset.md](readme/Dataset.md) for dataset setup instructions.

### 3. Evaluation
You can follow the instructions in [Evaluation.md](readme/Evaluation.md) to evaluate our model and reproduce the results in original paper.

### 4. Train
You can follow the instructions in [Train.md](readme/Train.md) to train our models.

### 5. Visualization
You can follow the instructions in [Visualization.md](readme/Visualization.md) to get visualization results.

<br/>

## References

- Backbone codes from [MOC](https://github.com/MCG-NJU/MOC-Detector).
- Data augmentation codes from [ACT](https://github.com/vkalogeiton/caffe/tree/act-detector).
- Evaluation codes from [ACT](https://github.com/vkalogeiton/caffe/tree/act-detector).
- DLA-34 backbone codes from [CenterNet](https://github.com/xingyizhou/CenterNet).

  [MOC LICENSE](https://github.com/MCG-NJU/MOC-Detector/blob/master/LICENSE)
  [ACT LICENSE](https://github.com/vkalogeiton/caffe/blob/act-detector/LICENSE)
  [CenterNet LICENSE](https://github.com/xingyizhou/CenterNet/blob/master/LICENSE)
  See more in [NOTICE](NOTICE)

  <br/>

### Citation
If you find this code is useful in your research, please cite:

```bibtex
@InProceedings{li2020actions,
    title={Actions as Moving Points},
    author={Yixuan Li and Zixu Wang and Limin Wang and Gangshan Wu},
    booktitle={arXiv preprint arXiv:2001.04608},
    year={2020}
}
```
