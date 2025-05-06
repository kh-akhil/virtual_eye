#!/bin/bash

VENV_DIR="env"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Virtual Enviroment"
    python3.11 -m venv $VENV_DIR
fi

os_name=$(uname)

if [ "$os_name" == "Linux" ] || [ "$os_name" == "Darwin" ]; then
    echo "Activating Virtual Enviroment"
    source $VENV_DIR/bin/activate
else
    echo "Activating Virtual Enviroment"
    $VENV_DIR/Scripts/activate.bat
fi

pip install --upgrade pip
pip install pyttsx3

if [ "$os_name" == "Linux" ]; then
    echo "Installing teresseract-ocr"
    sudo apt install tesseract-ocr
else
    pip install pytesseract
fi

echo "All dependencies installed"
echo "Starting the program"
python read_book.py