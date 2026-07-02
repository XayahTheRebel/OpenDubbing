# MuseTalk 1.5 one-shot setup script for Windows + Conda.
# Run from PowerShell as Administrator if necessary.

$ErrorActionPreference = "Stop"

$MuseTalkRoot = "C:\MuseTalk"
$CondaEnv = "MuseTalk"
$Cuda = "cu118"  # Change to cu121 if your GPU supports CUDA 12.x

function Test-Command {
    param([string]$Command)
    $null = Get-Command $Command -ErrorAction SilentlyContinue
}

if (-not (Test-Command conda)) {
    throw "conda not found. Please install Anaconda or Miniconda first."
}
if (-not (Test-Command git)) {
    throw "git not found. Please install Git first."
}
if (-not (Test-Command ffmpeg)) {
    Write-Warning "ffmpeg not found on PATH. Please install FFmpeg and add it to PATH before running OpenDubbing."
}

Write-Host "Creating Conda environment '$CondaEnv' with Python 3.10..."
conda create -n $CondaEnv python=3.10 -y
conda run -n $CondaEnv pip install --upgrade pip

if (-not (Test-Path $MuseTalkRoot)) {
    Write-Host "Cloning MuseTalk repository to $MuseTalkRoot..."
    git clone https://github.com/TMElyralab/MuseTalk.git $MuseTalkRoot
}
else {
    Write-Host "MuseTalk directory already exists at $MuseTalkRoot. Skipping clone."
}

Write-Host "Installing PyTorch ($Cuda)..."
conda run -n $CondaEnv pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url "https://download.pytorch.org/whl/$Cuda"

Write-Host "Installing MuseTalk requirements..."
conda run -n $CondaEnv pip install -r "$MuseTalkRoot\requirements.txt"

Write-Host "Installing MMLab stack..."
conda run -n $CondaEnv pip install --no-cache-dir -U openmim
conda run -n $CondaEnv mim install mmengine
conda run -n $CondaEnv mim install "mmcv==2.0.1"
conda run -n $CondaEnv mim install "mmdet==3.1.0"
# chumpy can fail to build on Windows inside build isolation; install it first.
conda run -n $CondaEnv pip install chumpy --no-build-isolation
conda run -n $CondaEnv mim install "mmpose==1.1.0"

Write-Host "Downloading model weights (huggingface_hub from requirements.txt)..."
$Models = "$MuseTalkRoot\models"
New-Item -ItemType Directory -Force -Path "$Models\musetalk", "$Models\musetalkV15", "$Models\syncnet", "$Models\dwpose", "$Models\face-parse-bisent", "$Models\sd-vae-ft-mse", "$Models\whisper" | Out-Null

conda run -n $CondaEnv huggingface-cli download TMElyralab/MuseTalk --local-dir $Models --include "musetalk/musetalk.json" "musetalk/pytorch_model.bin"
conda run -n $CondaEnv huggingface-cli download TMElyralab/MuseTalk --local-dir $Models --include "musetalkV15/musetalk.json" "musetalkV15/unet.pth"
conda run -n $CondaEnv huggingface-cli download stabilityai/sd-vae-ft-mse --local-dir "$Models\sd-vae-ft-mse" --include "config.json" "diffusion_pytorch_model.bin"
conda run -n $CondaEnv huggingface-cli download openai/whisper-tiny --local-dir "$Models\whisper" --include "config.json" "pytorch_model.bin" "preprocessor_config.json"
conda run -n $CondaEnv huggingface-cli download yzd-v/DWPose --local-dir "$Models\dwpose" --include "dw-ll_ucoco_384.pth"
conda run -n $CondaEnv huggingface-cli download ByteDance/LatentSync --local-dir "$Models\syncnet" --include "latentsync_syncnet.pt"
conda run -n $CondaEnv huggingface-cli download ManyOtherFunctions/face-parse-bisent --local-dir "$Models\face-parse-bisent" --include "79999_iter.pth" "resnet18-5c106cde.pth"

Write-Host "Verifying Conda environment..."
$probe = conda run -n $CondaEnv python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available())"
Write-Host $probe

Write-Host "Done. MuseTalk 1.5 is installed at $MuseTalkRoot inside Conda env '$CondaEnv'."
Write-Host "You can now use it in OpenDubbing with providers.face.name = 'musetalk'."
