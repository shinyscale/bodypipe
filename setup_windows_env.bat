@echo off
REM bodypipe Windows environment setup
REM No Developer Command Prompt needed — pytorch3d is vendored.

REM Initialize conda in this shell (not on PATH by default)
call C:\ProgramData\miniconda3\condabin\conda.bat activate base

set ENV_PREFIX=F:\conda_envs\bodypipe

echo.
echo === Step 1: Create conda env on F: ===
if exist "%ENV_PREFIX%\python.exe" (
    echo Env already exists, skipping create.
) else (
    conda create --no-default-packages --prefix %ENV_PREFIX% python=3.12 -y
)

echo.
echo === Step 2: Activate env ===
call conda activate %ENV_PREFIX%
python --version
echo Using: %ENV_PREFIX%

echo.
echo === Step 3: PyTorch nightly cu128 (Blackwell sm_120) ===
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
if errorlevel 1 goto :error

echo.
echo === Step 4: Core deps ===
pip install PySide6 PyOpenGL numpy scipy opencv-python requests
if errorlevel 1 goto :error

echo.
echo === Step 5: SMPL-X body model stack ===
pip install smplx trimesh
if errorlevel 1 goto :error

echo.
echo === Step 6: Smoke test ===
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.is_available()}')"
python -c "import PySide6; print('PySide6 OK')"
python -c "from OpenGL import GL; print('PyOpenGL OK')"

echo.
echo === Done! To launch bodypipe: ===
echo.
echo   call C:\ProgramData\miniconda3\condabin\conda.bat activate %ENV_PREFIX%
echo   cd F:\GVHMR\bodypipe
echo   python main.py
echo.
exit /b 0

:error
echo.
echo !!! Step failed. See error above. !!!
exit /b 1
