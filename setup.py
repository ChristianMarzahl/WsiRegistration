import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="qt-wsi-registration", 
    version="0.0.3",
    author="Christian Marzahl",
    author_email="christian.marzahl@gamil.com",
    description="Robust quad-tree based registration on whole slide images",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ChristianMarzahl/WsiRegistration",
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy', #>=1.19.5
        'opencv-python', # >=4.5.1.48
        'openslide-python', #>=1.1.2
        'matplotlib', #>=3.3.4
        'sklearn', #>=0.24.1
        'probreg', #>=0.3.1
        'pillow' # >=8.1.0
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)