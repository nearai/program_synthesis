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
        'boto3~=1.7.10',
        'cached-property~=1.4',
        'ipython~=5.5.0',
        'gym~=0.10.5',
        'numpy~=1.13.0',
        'torchfold~=0.1.0',
        'ply~=3.8',
        'pylru~=1.0.9',
        'pyparsing~=2.2.0',
        'pytest~=3.5',
        'pytest-timeout~=1.2',
        'pytest-xdist~=1.22',
        'python-Levenshtein~=0.12.0',
        'prompt_toolkit~=1.0.15',
        'tensorflow~=1.5.0',
        'tqdm~=4.23.4',
    ],
    project_urls={
        'Source': "https://github.com/nearai/program_synthesis",
    },
    python_requires='~=3.5',
)
