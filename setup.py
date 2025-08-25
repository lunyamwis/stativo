from setuptools import setup, find_packages

setup(
    name='stativiz',
    version='1.0.18',
    author='Martin Luther Bironga',
    description='Stativiz is a library for statistical analysis, visualization and modeling.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/lunyamwis/stativo.git',
    packages=find_packages(),
    install_requires=[
        'requests',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)