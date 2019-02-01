from setuptools import find_packages, setup


setup(
    name='Teras',
    version='0.1.4',
    author='Hiroki Teranishi',
    author_email='teranishihiroki@gmail.com',
    description='framework for deep learning applications',
    url='https://github.com/chantera/teras',
    license='MIT',
    install_requires=['numpy>=1.11.0'],
    packages=find_packages(),
)
