import setuptools
import pathlib


setuptools.setup(
    name='ivre',
    version='1.0.0',
    description='A benchmark for testing the ability of agents to learn to reason and resolve uncertainty.',
    url='https://sites.google.com/view/ivre/home',
    long_description=pathlib.Path('README.md').read_text(),
    long_description_content_type='text/markdown',
    packages=['src'],
    package_data={},
    install_requires=[
        'numpy', 'scipy', 'matplotlib', 'pandas', 'scikit-learn', 'tianshou', 'gym', 'pillow', 'tqdm', 'statsmodels', 'causality'
    ],
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Games/Entertainment',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
