from setuptools import setup, find_packages

setup(
    name="enso_analysis",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "xarray",
        "matplotlib",
        "scipy",
    ],
    entry_points={
        'console_scripts': [
            'enso-analysis=enso_analysis.__main__:main',
        ],
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A toolkit for analyzing El Niño-Southern Oscillation (ENSO) patterns",
    keywords="enso, climate, meteorology, oceanography",
    url="https://github.com/yourusername/enso-analysis",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Atmospheric Science",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.8",
)
