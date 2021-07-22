#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages


def parse_requirements(req_file):
    """Parse requirements.txt files."""
    reqs = open(req_file).read().strip().split("\n")
    reqs = [r for r in reqs if not r.startswith("#")]
    return [r for r in reqs if "#egg=" not in r]


REQUIREMENTS_FILE = "requirements.txt"
DEV_REQUIREMENTS_FILE = "requirements_dev.txt"
README_FILE = "README.md"

# Requirements
requirements = parse_requirements(REQUIREMENTS_FILE)
requirements_dev = parse_requirements(DEV_REQUIREMENTS_FILE)

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("CHANGELOG.rst") as changelog_file:
    changelog = changelog_file.read()

setup(
    name="seaborn_extensions",
    package="seaborn_extensions",
    packages=find_packages(),
    use_scm_version={
        "write_to": "seaborn_extensions/_version.py",
        "write_to_template": '__version__ = "{version}"\n',
    },
    author="Andre Rendeiro",
    author_email="afrendeiro@gmail.com",
    python_requires=">=3.5",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    description="Extensions of seaborn plots for biology",
    license="GNU General Public License v3",
    long_description=readme,
    # long_description_content_type
    include_package_data=True,
    keywords="seaborn_extensions",
    setup_requires=["setuptools_scm"],
    install_requires=requirements,
    tests_require=requirements_dev,
    extras_require={"dev": requirements_dev},
    url="https://github.com/afrendeiro/seaborn_extensions",
    project_urls={
        "Bug Tracker": "https://github.com/afrendeiro/seaborn_extensions/issues",
        "Documentation": "https://seaborn-extensions.readthedocs.io",
        "Source Code": "https://github.com/afrendeiro/seaborn_extensions",
    },
    zip_safe=False,
)
