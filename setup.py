# -*- coding: utf-8 -*-

# Learn more: https://github.com/JeremyBowyer/pyca

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='pyca',
    version='0.1.0',
    description='Data Analysis Tool',
    long_description=readme,
    author='Jeremy Bowyer',
    author_email='jeremybowyer@gmail.com',
    url='https://github.com/JeremyBowyer/pyca,
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)