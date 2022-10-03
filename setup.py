import platform
from setuptools import find_packages, setup, find_namespace_packages

setup(
    name="EmbryoFormer",
    version="v0.1",
    author="Tien-Phat Nguyen, Trong-Thang Pham, Hieu Le, ...",
    packages=find_namespace_packages(
        exclude=["dataset", "results", "notebooks", "script", "cfgs"]
    ),
    include_package_data=True,
    zip_safe=True,
    python_requires=">=3.6",
    install_requires=["numpy", "Pillow", "opencv-python", "tqdm", "yacs",],
)