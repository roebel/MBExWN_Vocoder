#! /bin/bash

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
cd $DIR/..
wget https://nubo.ircam.fr/index.php/s/RHexyqmbTEWLS6w/download -O SIIConv_pretrained_MBExWN_models.zip

unzip  SIIConv_pretrained_MBExWN_models.zip
rm -f SIIConv_pretrained_MBExWN_models.zip