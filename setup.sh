#!/bin/bash
# Check for root permission
if [[ ${UID} -ne 0 ]]
then
  echo “Please execute with root privileges.”
  exit 1
fi
# Install java
sudo add-apt-repository ppa:webupd8team/java
sudo apt-get install oracle-java8-installer
# Install python and python pip
sudo apt install python
sudo apt install python-pip
# Install support linux graphic library
sudo apt install imagemagick-6.q16
sudo apt install graphicsmagick-imagemagick-compat
sudo apt install imagemagick-6.q16hdri
# Install python graphic library Matplotlib and pandas
pip install matplotlib
pip install pandas
