import sys
import yaml

from pathlib import Path

from setuptools import setup, find_packages

if sys.version_info < (3,):
    sys.exit('loner requires Python >= 3.6')

try:
    from scnym import __author__, __email__
except ImportError:  # Deps not yet installed
    __author__ = __email__ = ''


def required_packages():
    deps = []
    with open("environment.yml", 'r') as stream:
        for dep in yaml.safe_load(stream)['dependencies']:
            if isinstance(dep, dict):
                for pip in dep['pip']:
                    deps.append(pip)
            else:
                deps.append(dep)
    return deps


setup(
    name='loner',
    version='0.1',
    description='Neural network classifiers for doublets',
    long_description=Path('README.md').read_text('utf-8'),
    url='http://github.com/calico/loner',
    author=__author__,
    author_email=__email__,
    license='Apache',
    python_requires='>=3.6',
    install_requires=required_packages(),
    packages=find_packages(),
    entry_points=dict(
        console_scripts=['scnym=loner.loner:main'],
    ),
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
    ],
)
