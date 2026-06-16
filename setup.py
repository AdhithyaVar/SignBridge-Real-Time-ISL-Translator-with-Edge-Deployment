# =============================================================================
# setup.py — Package installation configuration
# Install (editable): pip install -e .
# =============================================================================

from setuptools import setup, find_packages

setup(
    name="signbridge",
    version="1.0.0",
    description="Real-time Indian Sign Language (ISL) translator with edge deployment",
    author="SignBridge Team",
    python_requires=">=3.9",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=2.1.0",
        "opencv-python>=4.8.0",
        "mediapipe>=0.10.7",
        "onnx>=1.15.0",
        "onnxruntime>=1.16.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "pandas>=2.0.0",
        "PyYAML>=6.0.1",
        "pyttsx3>=2.90",
        "tqdm>=4.65.0",
    ],
    entry_points={
        "console_scripts": [
            "signbridge-collect  = scripts.collect_data:main",
            "signbridge-train    = scripts.train:main",
            "signbridge-evaluate = scripts.evaluate:main",
            "signbridge-demo     = scripts.run_demo:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Accessibility",
    ],
)
