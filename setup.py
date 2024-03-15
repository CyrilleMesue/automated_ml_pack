from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = "-e ."
def get_requirements(file_path) -> List[str]:
    """
    this function will return the list of requirements
    """

    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [package.replace("\n", "") for package in requirements]

        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements




setup(
name = "automatic_ml_pack",
version= "0.0.1",
author="Cyrille",
author_email="cyrillemesue@gmail.com",
description='This package is designed for swift and automated machine learning practice, catering to both classification and regression tasks. It facilitates model training, grid search application, and the preservation of the best model. Furthermore, it stores and visualizes the best scores attained by other models using commonly employed evaluation metrics.',
packages =  find_packages(),
install_requires=get_requirements("requirements.txt"),
python_requires='>=3.11'

)