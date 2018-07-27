from setuptools import setup, find_packages
from os import path


VERSION = '0.1.2'

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name='program-synthesis',
    version=VERSION,
    description='NEAR Program Synthesis: models, tools, and datasets for program synthesis tasks',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(exclude=["*_test.py", "test_*.py"]),
    author='NEAR Inc and Contributors',
    author_email='contact@near.ai',
    install_requires=[
        'boto3',
        'cached-property',
        'ipython',
        'gym',
        'numpy',
        'torchfold',
        'ply',
        'pylru',
        'pyparsing',
        'pytest',
        'pytest-timeout',
        'pytest-xdist',
        'python-Levenshtein',
        'prompt_toolkit',
        'sortedcontainers',
        'tensorflow',
        'tqdm',
    ],
    project_urls={
        'Source': "https://github.com/nearai/program_synthesis",
    },
    python_requires='>=3.5',
)
