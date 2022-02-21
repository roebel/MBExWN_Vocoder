import os
import sys
import re

from setuptools import setup, find_packages
from pkg_resources import parse_version
from distutils.command.sdist import sdist

# get _pysndfile version number
for line in open("MBExWN_Voc/__init__.py") :
    if line.startswith("mbexwn_version"):
        _mbexwn_nvoc_version_str = re.split('[()]', line)[1].replace(',','.',3).replace(',','-',1).replace('"','').replace(' ','')
        break

if sys.argv[1] == "get_version":
    print(parse_version(_mbexwn_nvoc_version_str))
    sys.exit(0)

setup(
    name="MBExWN_Voc",
    version=_mbexwn_nvoc_version_str,
    description="MBExWN Universal Mel Inverter for Speech and Singing Voiced",
    include_package_data=True,
    long_description=open("README.md").read(),
    author="Axel Roebel",
    author_email="axel.roebel@ircam.fr",
    packages=find_packages(where=".", exclude=("waveglow_model",)),
    install_requires=open("requirements.txt").read().splitlines(),
    scripts=[
        "bin/resynth_mel.py",
        "bin/generate_mel.py",
        "bin/view_mel.py",
    ],
)
