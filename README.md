# mech439_pybullet
Pybullet simulation framework for a MECH439 lecture

___
## Environments
* Windows 10/11, Ubuntu 20.04/22.04
* Python 3.8/3.10
___
## Requirements
1. Create conda environment
```shell
$ conda create -n <ENV_NAME> python==3.8    # create virtual environment
$ conda activate <ENV_NAME>
$ pip install -r requirements.txt           # install dependancies
$ conda install pinocchio -c conda-forge    # install pinocchio (Rigid Body Dynamics Library)
```