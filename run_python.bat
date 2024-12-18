@echo off
REM Define the name of the virtual environment
set ENV_NAME=my_new_env

REM Define the path to the project root directory where 'pikan' is located
set PROJECT_DIR=.

REM Create a new virtual environment
python -m venv %ENV_NAME%

REM Activate the virtual environment
call %ENV_NAME%\Scripts\activate.bat

REM Install dependencies from requirements.txt
pip install -r requirements.txt

REM Add the project directory to PYTHONPATH so the 'pikan' package can be found
set PYTHONPATH=%PROJECT_DIR%;%PYTHONPATH%

REM Check if the Python file is provided as an argument
if "%1"=="" (
    echo Usage: %0 ^<python_file^>
) else (
    REM Execute the Python file
    python %1
)

REM Deactivate the virtual environment
deactivate
