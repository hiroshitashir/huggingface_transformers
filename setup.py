from setuptools import setup, find_packages

setup(
    name="summary_bot",
    version="0.0.1",
    description="A package for HuggingFace natural language processing",
    url="https://github.com/hiroshitashir/huggingface_transormers",
    author="Hiroshi Tashiro",
    author_email="hiroshitashir@gmail.com",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
)
