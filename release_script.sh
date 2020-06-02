#!/bin/bash

rm -rf dist
source activate solo-sc
python setup.py sdist
twine upload dist/*
