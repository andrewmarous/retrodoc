from setuptools import setup, find_packages

setup(
    name='retrodoc',
    version='0.1.1',
    packages=find_packages(),
    entry_points={
        'console_scripts': [
            'retrodoc=main.main:main',
        ],
    },
    author='Andrew Marous',
    author_email='andrewmarous@gmail.com',
    description='CLI application that generates industry-standard specifications for a file using ChatGPT 3.5'

    # Include other package metadata as needed
)
