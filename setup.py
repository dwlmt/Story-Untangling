from setuptools import setup, find_packages

# Use requirements text to manage the dependencies.
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='Story-Untangling',
    version='0.0.1',
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=requirements,
    url='',
    license='',
    author='David Wilmot',
    author_email='david.wilmot@ed.ac.uk',
    description='Text story experiments for understanding based on predictive expectations from model output.',
)
