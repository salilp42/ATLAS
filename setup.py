from setuptools import setup, find_packages

setup(
    name="atlas-medical-diffusion",
    version="0.1.0",
    description="ATLAS: Medical Diffusion Model with Clinical Feature Preservation",
    author="Salil Patel",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "wandb>=0.15.0",
        "pydicom>=2.3.0",
        "tqdm>=4.65.0",
        "scikit-learn>=1.2.0",
        "pillow>=9.5.0"
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Medical Science Apps."
    ]
)
