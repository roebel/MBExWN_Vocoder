#! /bin/bash

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
cd $DIR/..
zip -r SIIConv_pretrained_MBExWN_models.zip ./MBExWN_NVoc/models/MBExWN_SIIConv_V71g*
