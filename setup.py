#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("CHANGELOG.rst") as changelog_file:
    changelog = changelog_file.read()

requirements = ["numpy", "pandas", "matplotlib", "seaborn", "pingouin"]

setup_requirements = [
    "pytest-runner",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="Andre Rendeiro",
    author_email="afrendeiro@gmail.com",
    python_requires=">=3.5",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Extensions of seaborn plots for biology",
    install_requires=requirements,
    license="GNU General Public License v3",
    long_description=readme + "\n\n" + changelog,
    include_package_data=True,
    keywords="seaborn_extensions",
    name="seaborn_extensions",
    packages=find_packages(
        include=["seaborn_extensions", "seaborn_extensions.*"]
    ),
    setup_requires=setup_requirements,
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/afrendeiro/seaborn_extensions",
    version="0.1.0",
    zip_safe=False,
)
