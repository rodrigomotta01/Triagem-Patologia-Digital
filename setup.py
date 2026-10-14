from setuptools import setup, find_packages

setup(
    name="pathoshift",
    version="0.1.0",
    description="PathoShift Challenge: Triagem Robusta em Patologia Digital",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "Pillow>=10.0.0",
        "opencv-python>=4.8.0",
        "scikit-image>=0.21.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
        "joblib>=1.3.0",
    ],
    entry_points={
        "console_scripts": [
            "pathoshift=pathoshift.cli:main",
        ],
    },
)

