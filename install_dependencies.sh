#!/bin/bash

create_and_activate_venv() {
    
    if [ ! -d "env" ]; then
        echo "Creating virtual environment..."
        python3 -m venv env
    else
        echo "Virtual environment already exists!"
    fi

    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        source env/bin/activate
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        source env/bin/activate
    elif [[ "$OSTYPE" == "cygwin" || "$OSTYPE" == "msys" ]]; then
        source env/Scripts/activate
    else
        echo "Unsupported OS. Exiting..."
        exit 1
    fi
}

pip install --upgrade pip
pip install opencv-python "numpy<2" ultralytics pyttsx3 google-generativeai python-dotenv gpiozero SpeechRecognition paddlepaddle paddleocr

echo "All required modules have been installed successfully!"
