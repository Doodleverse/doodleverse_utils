from setuptools import setup, find_packages
from pathlib import Path

DESCRIPTION = 'Imports into the Doodleverse.'

exec(open('doodleverse_utils/version.py').read())

setup(
    name="doodleverse_utils",
    version=__version__,
    author="Daniel Buscombe",
    author_email="dbuscombe@gmail.com",
    description=DESCRIPTION,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: GIS",
        ],
    keywords=[
        "doodleverse",
        "image segmentation",
        "remotesensing",
        "gis",
        "deep learning",],
    python_requires="<3.13",
    project_urls={
        "Issues": "https://github.com/Doodleverse/doodleverse_utils/issues",
        "GitHub":"https://github.com/Doodleverse/doodleverse_utils",
    },
)