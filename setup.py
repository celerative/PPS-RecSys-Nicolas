from setuptools import setup, find_packages, dist

__version__ = '0.1'

with open('requirements.txt') as req:
    install_requires = req.read().splitlines()

setup(
    name='pyrecsys',
    version=__version__,
    packages=find_packages(exclude=['tests']),
    install_requires=install_requires,
)
