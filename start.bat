@echo off
chcp 65001 >nul 2>&1
setlocal EnableDelayedExpansion
title MovCap - Visual-Inertial Motion Capture System

set "PROJECT_DIR=%~dp0"
cd /d "%PROJECT_DIR%"

:MENU
cls
echo ================================================================
echo   MovCap - Visual-Inertial Motion Capture System
echo ================================================================
echo.
echo   [1] Launch GUI (Recommended)
echo   [2] Install ^& Check Environment
echo   [3] Generate ChArUco Calibration Board
echo   [4] Calibrate Cameras
echo   [5] Record MoCap Session
echo   [6] Process Recorded Data
echo   [7] Run Tests
echo   [8] Edit Configuration
echo   [0] Exit
echo.
echo ================================================================
set /p "CHOICE=Select option: "

if "%CHOICE%"=="1" goto LAUNCH_GUI
if "%CHOICE%"=="2" goto CHECK_ENV
if "%CHOICE%"=="3" goto GEN_BOARD
if "%CHOICE%"=="4" goto CALIBRATE
if "%CHOICE%"=="5" goto RECORD
if "%CHOICE%"=="6" goto PROCESS
if "%CHOICE%"=="7" goto TEST
if "%CHOICE%"=="8" goto EDIT_CONFIG
if "%CHOICE%"=="0" goto END
goto MENU

:: ================================================================
:: Launch GUI
:: ================================================================
:LAUNCH_GUI
cls
echo ================================================================
echo   Launching MovCap GUI
echo ================================================================
echo.

:: Auto-setup if venv doesn't exist
if not exist "venv\Scripts\activate.bat" (
    echo [INFO] First-time setup: creating virtual environment...
    call :AUTO_SETUP
    if !errorlevel! neq 0 (
        echo [ERROR] Setup failed. Run option 2 to check environment.
        pause
        goto MENU
    )
)

call venv\Scripts\activate.bat 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Virtual environment not found. Run option 2 first.
    pause
    goto MENU
)

echo [INFO] Launching GUI...
python -m scripts.gui --config config\default.yaml

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] GUI exited with errors.
    pause
)
goto MENU

:: ================================================================
:: Auto Setup (called from GUI launch)
:: ================================================================
:AUTO_SETUP
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Install Python 3.10+ first.
    exit /b 1
)

python -m venv venv
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment.
    exit /b 1
)

call venv\Scripts\activate.bat
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 pip install -r requirements.txt

echo [OK] Setup complete.
exit /b 0

:: ================================================================
:: Environment Check & Install
:: ================================================================
:CHECK_ENV
cls
echo ================================================================
echo   Environment Check
echo ================================================================
echo.

:: Check Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found in PATH.
    echo.
    echo Please install Python 3.10+ from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    goto MENU
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set "PYVER=%%v"
echo [OK] Python version: %PYVER%

:: Check Python version >= 3.10
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do (
    set "PYMAJOR=%%a"
    set "PYMINOR=%%b"
)
if %PYMAJOR% LSS 3 (
    echo [ERROR] Python 3.10+ required, found %PYVER%.
    pause
    goto MENU
)
if %PYMAJOR% EQU 3 if %PYMINOR% LSS 10 (
    echo [ERROR] Python 3.10+ required, found %PYVER%.
    pause
    goto MENU
)

:: Check/create venv
if not exist "venv\Scripts\activate.bat" (
    echo.
    echo [INFO] Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        goto MENU
    )
    echo [OK] Virtual environment created.
) else (
    echo [OK] Virtual environment found.
)

:: Activate venv
call venv\Scripts\activate.bat

:: Check pip
python -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Upgrading pip...
    python -m ensurepip --upgrade >nul 2>&1
)
for /f "tokens=2 delims= " %%v in ('python -m pip --version 2^>^&1') do echo [OK] pip version: %%v

:: Install dependencies
echo.
echo [INFO] Checking dependencies...
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [WARN] Some packages may have failed. Retrying...
    pip install -r requirements.txt
)

:: Check CUDA availability
echo.
python -c "import torch; cuda='available' if torch.cuda.is_available() else 'NOT available'; print(f'[INFO] CUDA: {cuda}')" 2>nul
if %errorlevel% neq 0 (
    echo [WARN] Could not check CUDA status.
)

:: Verify key imports
echo.
echo [INFO] Verifying imports...
python -c "import cv2; print(f'[OK] OpenCV {cv2.__version__}')" 2>nul
if %errorlevel% neq 0 echo [ERROR] OpenCV import failed.
python -c "import ultralytics; print(f'[OK] Ultralytics {ultralytics.__version__}')" 2>nul
if %errorlevel% neq 0 echo [ERROR] Ultralytics import failed.
python -c "import numpy; print(f'[OK] NumPy {numpy.__version__}')" 2>nul
if %errorlevel% neq 0 echo [ERROR] NumPy import failed.
python -c "import scipy; print(f'[OK] SciPy {scipy.__version__}')" 2>nul
if %errorlevel% neq 0 echo [ERROR] SciPy import failed.
python -c "import filterpy; print(f'[OK] FilterPy {filterpy.__version__}')" 2>nul
if %errorlevel% neq 0 echo [ERROR] FilterPy import failed.
python -c "import yaml; print('[OK] PyYAML')" 2>nul
if %errorlevel% neq 0 echo [ERROR] PyYAML import failed.
python -c "import serial; print(f'[OK] PySerial {serial.VERSION}')" 2>nul
if %errorlevel% neq 0 echo [ERROR] PySerial import failed.
python -c "import tkinter; print('[OK] Tkinter (GUI)')" 2>nul
if %errorlevel% neq 0 echo [ERROR] Tkinter import failed.

:: Check calibration files
echo.
if exist "config\calibration\intrinsic.yaml" (
    echo [OK] Intrinsic calibration found.
) else (
    echo [WARN] No intrinsic calibration. Run calibration first.
)
if exist "config\calibration\extrinsic.yaml" (
    echo [OK] Extrinsic calibration found.
) else (
    echo [WARN] No extrinsic calibration. Run calibration first.
)

echo.
echo ================================================================
echo   Environment check complete.
echo ================================================================
pause
goto MENU

:: ================================================================
:: Generate ChArUco Board
:: ================================================================
:GEN_BOARD
cls
echo ================================================================
echo   Generate ChArUco Calibration Board
echo ================================================================
echo.

call venv\Scripts\activate.bat 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Virtual environment not found. Run option 2 first.
    pause
    goto MENU
)

if not exist "config\calibration" mkdir "config\calibration"
python -m scripts.calibrate --generate-board --output config\calibration\ --config config\default.yaml
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Board generation failed.
) else (
    echo.
    echo [OK] Board saved to config\calibration\charuco_board.png
    echo [TIP] Print this on A4 paper at 100%% scale, mount on rigid board.
    start "" "config\calibration\charuco_board.png"
)
pause
goto MENU

:: ================================================================
:: Calibrate Cameras
:: ================================================================
:CALIBRATE
cls
echo ================================================================
echo   Camera Calibration
echo ================================================================
echo.
echo   Requirements:
echo   - 3 USB cameras connected
echo   - Printed ChArUco board
echo.
echo   Process:
echo   - Show the board to all cameras
echo   - Move it to different positions/angles
echo   - System collects frames automatically
echo.

set /p "DUR=Collection duration in seconds [30]: "
if "%DUR%"=="" set "DUR=30"

call venv\Scripts\activate.bat 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Virtual environment not found. Run option 2 first.
    pause
    goto MENU
)

echo.
echo [INFO] Starting calibration (%DUR%s)...
echo [INFO] Press 'q' in the camera window to stop early.
echo.

if not exist "config\calibration" mkdir "config\calibration"
python -m scripts.calibrate --config config\default.yaml --output config\calibration\ --duration %DUR%

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Calibration failed. Check camera connections.
) else (
    echo.
    echo [OK] Calibration complete.
    if exist "tools\visualization\preview_calib.py" (
        echo.
        set /p "PREVIEW=Preview calibration results? (y/n) [n]: "
        if /i "!PREVIEW!"=="y" (
            python tools\visualization\preview_calib.py
        )
    )
)
pause
goto MENU

:: ================================================================
:: Record MoCap Session
:: ================================================================
:RECORD
cls
echo ================================================================
echo   Record MoCap Session
echo ================================================================
echo.

call venv\Scripts\activate.bat 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Virtual environment not found. Run option 2 first.
    pause
    goto MENU
)

:: Check calibration
if not exist "config\calibration\intrinsic.yaml" (
    echo [ERROR] No calibration data found.
    echo Run option 4 to calibrate cameras first.
    pause
    goto MENU
)

:: Output filename
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do set "DT=%%I"
set "TIMESTAMP=%DT:~0,8%_%DT:~8,6%"
set "DEFAULT_OUT=recordings\session_%TIMESTAMP%.bvh"

set /p "OUTFILE=Output BVH path [%DEFAULT_OUT%]: "
if "%OUTFILE%"=="" set "OUTFILE=%DEFAULT_OUT%"

:: Create output directory
for %%F in ("%OUTFILE%") do if not exist "%%~dpF" mkdir "%%~dpF"

set /p "DUR=Duration in seconds (0=continuous) [0]: "
if "%DUR%"=="" set "DUR=0"

set /p "NOVIS=Disable real-time visualization? (y/n) [n]: "

set "EXTRA_ARGS="
if /i "%NOVIS%"=="y" set "EXTRA_ARGS=--no-visual"

echo.
echo [INFO] Starting recording...
echo [INFO] Press Ctrl+C to stop.
echo.

python -m scripts.record --config config\default.yaml --calibration config\calibration\ --output "%OUTFILE%" --duration %DUR% %EXTRA_ARGS%

if %errorlevel% neq 0 (
    echo.
    echo [WARN] Recording ended with errors.
) else (
    echo.
    echo [OK] Recording complete.
)
pause
goto MENU

:: ================================================================
:: Process Recorded Data
:: ================================================================
:PROCESS
cls
echo ================================================================
echo   Process Recorded Data
echo ================================================================
echo.
echo Available raw data files:
echo.

call venv\Scripts\activate.bat 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Virtual environment not found. Run option 2 first.
    pause
    goto MENU
)

set "COUNT=0"
for %%F in (recordings\*_raw.json) do (
    set /a COUNT+=1
    echo   [!COUNT!] %%~nxF
)
if %COUNT%==0 (
    echo   (no raw data files found in recordings\)
    echo   Record a session first (option 5) or export demo data from GUI.
    pause
    goto MENU
)

echo.
set /p "INPUT=Input JSON file path: "
if "%INPUT%"=="" goto MENU

if not exist "%INPUT%" (
    echo [ERROR] File not found: %INPUT%
    pause
    goto MENU
)

:: Generate output name
for %%F in ("%INPUT%") do set "OUTNAME=%%~nF"
set "OUTNAME=%OUTNAME:_raw=%"
set "OUTFILE=recordings\%OUTNAME%_smoothed.bvh"

echo.
echo [INFO] Processing %INPUT%...
echo [INFO] Output: %OUTFILE%
echo.

python -m scripts.process --config config\default.yaml --input "%INPUT%" --output "%OUTFILE%"

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Processing failed.
) else (
    echo.
    echo [OK] Processing complete.
    echo [OK] Output: %OUTFILE%
)
pause
goto MENU

:: ================================================================
:: Run Tests
:: ================================================================
:TEST
cls
echo ================================================================
echo   Run Tests
echo ================================================================
echo.

call venv\Scripts\activate.bat 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] Virtual environment not found. Run option 2 first.
    pause
    goto MENU
)

echo [INFO] Running pytest...
echo.
pytest tests\ -v
echo.
pause
goto MENU

:: ================================================================
:: Edit Configuration
:: ================================================================
:EDIT_CONFIG
cls
echo ================================================================
echo   Configuration Files
echo ================================================================
echo.
echo   [1] config\default.yaml     (main settings)
echo   [2] config\skeleton_model.yaml (skeleton hierarchy)
echo   [0] Back
echo.
set /p "CFG_CHOICE=Select: "

if "%CFG_CHOICE%"=="1" (
    start notepad "config\default.yaml"
) else if "%CFG_CHOICE%"=="2" (
    start notepad "config\skeleton_model.yaml"
)
goto MENU

:: ================================================================
:END
echo.
echo Goodbye.
endlocal
exit /b 0
