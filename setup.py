import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="TwoSampleHC", 
    version="0.0.9",
    author="Alon Kipnis",
    author_email="alonkipnis@gmail.com",
    description="Two-sample Higher Criticism",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/alonkipnis/TwoSampleHC",
    download_url="https://github.com/alonkipnis/TwoSampleHC",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires = [
    'numpy>=1.6',
    'scipy>=1.2',
    'pandas>=0.24'
]
)