import sys
from pathlib import Path
from setuptools import setup, find_packages

if sys.version_info < (3,):
    sys.exit('solo requires Python >= 3.6')

try:
    from solo import __author__, __email__
except ImportError:  # Deps not yet installed
    __author__ = __email__ = ''


setup(
    name='solo-sc',
    version='0.2',
    description='Neural network classifiers for doublets',
    long_description=Path('README.md').read_text('utf-8'),
    long_description_content_type="text/markdown",
    url='http://github.com/calico/solo',
    download_url='https://github.com/calico/solo/archive/0.1.tar.gz',
    author=__author__,
    author_email=__email__,
    license='Apache',
    python_requires='>=3.6',
    install_requires=["ConfigArgParse==0.14.0",
                      "cycler==0.10.0",
                      "decorator==4.4.0",
                      "joblib==0.13.2",
                      "mock==3.0.5",
                      "natsort==6.0.0",
                      "networkx==2.2",
                      "numexpr==2.6.9",
                      "pandas==0.24.2",
                      "patsy==0.5.1",
                      "pyparsing==2.4.0",
                      "python-dateutil==2.8.0",
                      "pytz==2019.1",
                      "seaborn==0.9.0",
                      "six==1.12.0",
                      "tables==3.5.1",
                      "tqdm==4.32.1",
                      "numba==0.45.0",
                      "numpy>=1.16.4",
                      "scanpy==1.4.5.1",
                      "scvi==0.6.0",
                      "dataclasses==0.6",
                      "leidenalg==0.7.0",
                      "pytest==5.2.1",
                      "prompt-toolkit==2.0.9",
                      "torch==1.0.1"
                      ],
    packages=find_packages(),
    entry_points=dict(
        console_scripts=['solo=solo.solo:main',
                         'hashsolo=solo.hashsolo:main'],
    ),
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
)
