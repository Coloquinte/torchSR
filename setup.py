import pathlib
from setuptools import setup


with open('README.md') as f:
    long_description = f.read()

setup(
    name="torchsr",
    version="1.0.4",
    description="Super Resolution Networks for pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Coloquinte/torchSR",
    author="Gabriel Gouvine",
    author_email="gabriel.gouvine_GIT@m4x.org.com",
    keywords=["superresolution", "pytorch", "edsr", "rcan", "ninasr"],
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
    ],
    packages=["torchsr", "torchsr.models", "torchsr.datasets", "torchsr.transforms"],
    include_package_data=True,
    install_requires=["torch>=1.6", "torchvision>=0.6"],
)

