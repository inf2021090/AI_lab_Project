@echo off

REM Define variables
set VENV_NAME=myenv
set REQUIREMENTS=requirements.txt

REM Display help message
if [%1]==[/?] goto :Help
if [%1]==[--help] goto :Help
goto :Start

:Help
echo Usage: make [COMMAND]
echo.
echo Commands:
echo   install         Installs dependencies
echo   setup           Sets up a new virtual environment and activates it
echo   activate        Activates the virtual environment
echo   update          Updates the virtual environment and requirements
echo   add             Adds new dependency to requirements.txt
echo   clean           Cleans the virtual environment
echo.
echo Example usage:
echo   make install
echo   make setup
echo   make activate
echo   make update
echo   make add requests
echo   make clean
echo.
goto :End

:Start
REM Check command
if [%1]==[] goto :Help
if [%1]==[install] goto :Install
if [%1]==[setup] goto :Setup
if [%1]==[activate] goto :Activate
if [%1]==[update] goto :Update
if [%1]==[add] goto :AddDependency
if [%1]==[clean] goto :Clean
goto :Help

:Install
echo Installing dependencies...
python -m venv %VENV_NAME%
call %VENV_NAME%\Scripts\activate.bat
pip install -r %REQUIREMENTS%
echo Install complete.
goto :End

:Setup
echo Setting up virtual environment...
python -m venv %VENV_NAME%
call %VENV_NAME%\Scripts\activate.bat
echo Setup complete.
goto :End

:Activate
echo Activating virtual environment...
call %VENV_NAME%\Scripts\activate.bat
echo Environment activated.
goto :End

:Update
echo Updating virtual environment...
call %VENV_NAME%\Scripts\activate.bat
pip install -r %REQUIREMENTS%
echo Update complete.
goto :End

:AddDependency
if [%2]==[] goto :HelpAdd
echo Adding dependency %2 to requirements.txt...
echo %2 >> %REQUIREMENTS%
echo Dependency added.
goto :End

:HelpAdd
echo Usage: make add [DEPENDENCY]
echo.
echo Example usage:
echo   make add numpy
echo   make add seaborn
echo.
goto :End

:Clean
echo Cleaning virtual environment...
call %VENV_NAME%\Scripts\deactivate.bat
rmdir /s /q %VENV_NAME%
echo Clean complete.
goto :End

:End

