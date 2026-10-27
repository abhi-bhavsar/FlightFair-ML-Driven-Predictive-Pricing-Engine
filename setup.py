from setuptools import find_packages, setup

setup(
    name="flightfair",
    version="1.0.0",
    author="Abhishek Bhavsar",
    description="ML-Driven Predictive Pricing Engine for Airline Tickets",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    py_modules=["app", "streamlit_app"],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "mlflow>=2.10.0",
        "dvc>=3.0.0",
        "flask>=3.0.0",
        "flask-cors>=4.0.0",
        "pydantic>=2.0.0",
        "waitress>=3.0.0",
        "streamlit>=1.30.0",
        "openpyxl>=3.1.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "python-dotenv>=1.0.0",
        "pytest>=7.4.0",
    ],
    entry_points={
        "console_scripts": [
            "flightfair-train=src.pipeline.train_pipeline:main",
            "flightfair-serve=app:main",
        ],
    },
)
