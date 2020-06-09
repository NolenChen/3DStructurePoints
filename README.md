# Unsupervised Learning of Intrinsic Structural Representation Points


## Description
This repository contains the code for our cvpr 2020 paper: [Unsupervised Learning of Intrinsic Structural Representation Points](https://arxiv.org/abs/2003.01661)

<div align="center">
<img src="https://github.com/NolenChen/3DStructurePoints/blob/master/figs/teaser.png" width="70%" height="70%"><br><br>
</div>

 
## Environment setup

Current Code is tested on ubuntu16.04 with cuda9, python3.6, torch 1.1.0 and torchvision 0.3.0. 
We use a [pytorch version of pointnet++](https://github.com/erikwijmans/Pointnet2_PyTorch) in our pipeline.
```
pip install -r requirements.txt
cd pointnet2
python setup.py build_ext --inplace
```


## Dataset

The training and testing data for 3D semantic correspondence is provided by [LMVCNN](https://people.cs.umass.edu/~hbhuang/local_mvcnn/) and [bhcp](http://www.vovakim.com/projects/CorrsTmplt/doc_data.html) respectively, and you can download the preprocessed training data [here](https://drive.google.com/file/d/1MkUcFF4gbfhQLssPNd_MV9afOOar-Wxn/view?usp=sharing).
And [here](https://drive.google.com/open?id=1LexLVRwq13FIT-dfIuD1Ii9eBq_tQ0ph)'s the script for preprocessing the testing data.


## Train

```
cd train
python train_structure_points.py -data_dir PATH_TO_TRAINING_DATA -num_structure_points 16 -category plane -log_dir PATH_TO_LOG
```
* -data_dir: path to the preprocessed training data
* -num_structure_points: number of structure points
* -category: category to train on
* -log_dir: path to log dir

The trained model will be saved in PATH_TO_LOG/checkpoints/model

## Test

```
cd test
python test_structure_points.py -data_dir ../demo_data/plane -model_fname PATH_TO_TRAINED_MODEL -num_structure_points 16 -output_dir OUTPUT_PATH
```
* -model_fname: path to trained model
* -data_dir: path to the testing data
* -output_dir: output path.
* -num_structure_points: number of structure points, should be the same with training stage

The structure point will be outputed in off format, corresponding structure points will have same colors


## Citation
Please cite our paper if you find it useful in your research:

```
@article{chen2020unsupervised,
  title={Unsupervised Learning of Intrinsic Structural Representation Points},
  author={Chen, Nenglun and Liu, Lingjie and Cui, Zhiming and Chen, Runnan and Ceylan, Duygu and Tu, Changhe and Wang, Wenping},
  journal={arXiv preprint arXiv:2003.01661},
  year={2020}
}
```

If you have any questions, please contact chennenglun@gmail.com



