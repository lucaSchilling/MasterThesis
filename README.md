# Image Registration using Deep Learning
<p align="left">
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.8-blue.svg?style=flat" alt="MIT License"></a>
    <br />
    <a href="https://github.com/tensorflow/tensorflow"><img src="https://img.shields.io/badge/TensorFlow-FF6F00.svg?logo=tensorflow&logoColor=white" alt="Tensorflow"></a>
    <a href="https://github.com/pytorch/pytorch"><img src="https://img.shields.io/badge/PyTorch-EE4C2C.svg?logo=pytorch&logoColor=white" alt="PyTorch"></a>
    <br />
    <i><b>Author</b>:
        <a href="https://github.com/lucaSchilling">Luca S.</a>
    </i>
    <br />
    <hr />

## Requirements
 - Python 3.8
 - Poetry 1.1
 - cmake 3.2.1
 - git
 - build-essentials

## Setup (tested on Ubuntu 20.04 only)
1. Download & install the latest python 3.8 release from [here](https://www.python.org/downloads/mac-osx/) 

2. Download and install poetry as explained [here](https://python-poetry.org/docs/)

3. Download and compile SimpleElastix as explained [here](https://simpleelastix.readthedocs.io/GettingStarted.html#compiling-on-linux)

4. Run ``` poetry install -vvv ``` in the root of the repository to create a virtual environment and install all dependencies listed in the pyproject.toml. -vvv activates the debug mode which somehow is needed to install airlab as development dependency correctly.

5. From within poetry's venv console navigate to ```{BUILD_DIRECTORY}/SimpleITK-build/Wrapping/Python``` and execute ```python Packaging/setup.py install```
   
Hint: To enter poetry's venv u can use ```poetry shell``` if poetry is added correctly to your path
otherwise use ```source {path_to_venv}/bin/activate```.

On Ubuntu Poetry's venvs are located under ```~/.cache/pypoetry/virtualenvs/```

6. Run any of the run scripts... 
   
   6.1 ... using the command line
   ```
   poetry shell
   
   poetry run python validation.py

   ```
   6.2 ... using an IDE:
    - Select the poetry venv as interpreter and run/debug one of the scripts as usual