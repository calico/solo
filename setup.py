import sys
from pathlib import Path
from setuptools import setup, find_packages

if sys.version_info < (3,):
    sys.exit("solo requires Python >= 3.6")

try:
    from solo import __author__, __email__
except ImportError:  # Deps not yet installed
    __author__ = __email__ = ""


setup(
    name="solo-sc",
    version="1.1",
    description="Neural network classifiers for doublets",
    long_description=Path("README.md").read_text("utf-8"),
    long_description_content_type="text/markdown",
    url="http://github.com/calico/solo",
    download_url="https://github.com/calico/solo/archive/1.0.tar.gz",
    author=__author__,
    author_email=__email__,
    license="Apache",
    python_requires=">=3.7",
    install_requires=[
        l.strip() for l in Path("requirements.txt").read_text("utf-8").splitlines()
    ],
    packages=find_packages(exclude="testdata"),
    entry_points=dict(
        console_scripts=["solo=solo.solo:main", "hashsolo=solo.hashsolo:main"],
    ),
    classifiers=[
        "Environment :: Console",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
