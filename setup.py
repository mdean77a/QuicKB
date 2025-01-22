from setuptools import setup, find_packages

setup(
    name="quickb",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=open("requirements.txt").read().splitlines(),
    entry_points={
        'console_scripts': [
            'quickb=main:main',
        ],
    },
    python_requires='>=3.9',
)