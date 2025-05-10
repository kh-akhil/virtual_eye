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
pip install opencv-python tensorflow matplotlib

echo ""
echo "All Dependies are installed. Running model file..."

python currency_detection_model.py 2>error.log
status=$?

if [ $status -ne 0 ]; then
    if grep -q "certificate verify failed" error.log; then
        echo "SSL certificate error detected"
        case "$os_name" in 
            ubuntu|linux)
                echo "Installing ca-certificates"
                sudo apt-get install ca-certificates
                ;;
            Darwin)
                echo "Installing ca-certificates"
                /Applications/Python\ 3.11/Install\ Certificates.command
                ;;
            *)
                echo "Please install ca-certificates manually."
                exit 1
                ;;
        esac
        echo "Re-running the model file..."
        python currency_detection.py
    else
        echo "An error occurred while running the model file. Please check error.log for details."
        exit 1
    fi
else
    echo "Model file executed successfully."
fi

echo ""
echo "Running main file"
python currency_detection_new.py