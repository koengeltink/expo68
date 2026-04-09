#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR

source /home/expo68/expo-project/opencv-env/bin/activate

python3 $SCRIPT_DIR/multi_gui.py
read -p "Press "enter" to close window"


