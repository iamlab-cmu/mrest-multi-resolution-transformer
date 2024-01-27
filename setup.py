import os
import sys
from setuptools import setup, find_packages

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))
    
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='mrest',
    version='0.0.0',
    packages=find_packages(),
    description='MResT: Multi-Resolution Sensing for Real-Time Control with Vision-Language Models',
    long_description=read('README.md'),
    author='Saumya Saxena, Mohit Sharma (CMU)',
    install_requires=[
        'click',
        'dill',
        'einops',
        'gdown>=4.4.0',
        'gspread',
        'gym==0.20.0',
        'hydra-core',
        'hydra-core>=1.1.1',
        'matplotlib',
        'moviepy',
        'omegaconf>=2.1.1',
        'pandas',
        'pillow>=9.0.1',
        'pip',
        'prettytable',
        'pyquaternion',
        'robosuite',
        'scipy',
        'shapely',
        'tabulate',
        'termcolor', 
        'torch>=1.7.1',
        'torchvision>=0.8.2',
        'transformers==4.25.1',
        'transforms3d',
        'wandb==0.13.7',
    ],
)
