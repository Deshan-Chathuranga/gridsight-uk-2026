from setuptools import setup, find_packages

setup(
    name="gridsight",
    version="1.0.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        # dependencies are pinned in requirements.txt
    ],
    python_requires=">=3.11",
)
