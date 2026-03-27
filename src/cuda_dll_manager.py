"""Automatic CUDA/cuBLAS DLL downloader for GPU acceleration.

This module handles downloading NVIDIA CUDA runtime libraries on demand
when GPU transcription is requested. This avoids bundling potentially
copyrighted NVIDIA binaries directly in the repository.

CUDA libraries are downloaded from NVIDIA's official distribution packages
via pip (nvidia-cublas-cu12, nvidia-cuda-runtime-cu12, etc.)
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional


def _get_cuda_dll_path() -> Optional[Path]:
    """Get the path to CUDA DLLs if they exist."""
    # Check common CUDA installation paths on Windows
    cuda_paths = [
        Path(os.environ.get("CUDA_PATH", "")) / "bin",
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.x/bin"),
        Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.x/bin"),
    ]

    for cuda_path in cuda_paths:
        if cuda_path.exists():
            return cuda_path

    # Check if NVIDIA packages were installed via pip in venv
    if hasattr(sys, "_base_executable"):
        venv_path = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
        if venv_path.exists():
            # nvidia-cublas-cu12 puts DLLs in nvidia/cublas/lib
            for subdir in ["cublas", "cuda_runtime", "cudart"]:
                dll_path = venv_path / subdir / "lib"
                if dll_path.exists():
                    return dll_path

    return None


def _find_cublas_dlls() -> list[Path]:
    """Find all available cuBLAS DLL files."""
    dlls = []
    cuda_path = _get_cuda_dll_path()

    if cuda_path and cuda_path.exists():
        dlls.extend(cuda_path.glob("cublas*.dll"))
        dlls.extend(cuda_path.glob("cudart*.dll"))
        dlls.extend(cuda_path.glob("nvJit*.dll"))

    return dlls


def ensure_cuda_dlls(force_download: bool = False) -> bool:
    """Ensure CUDA DLLs are available for GPU computation.

    This function checks if CUDA/cuBLAS DLLs are available and offers
    to download them via pip if not present and GPU is requested.

    Args:
        force_download: If True, always download even if DLLs exist.

    Returns:
        True if CUDA DLLs are now available, False otherwise.
    """
    # Check if DLLs already exist
    if not force_download and _find_cublas_dlls():
        return True

    # Check if NVIDIA packages are available via pip
    try:
        import nvidia.cublas
        import nvidia.cuda_runtime
        return True
    except ImportError:
        pass

    return False


def download_cuda_dlls(verbose: bool = True) -> bool:
    """Download and install CUDA runtime libraries via pip.

    This installs the nvidia-cublas-cu12 and nvidia-cuda-runtime-cu12
    packages which include the necessary DLLs for GPU acceleration.

    Args:
        verbose: If True, print progress messages.

    Returns:
        True if download/install succeeded, False otherwise.
    """
    if verbose:
        print("Downloading NVIDIA CUDA runtime libraries...")

    try:
        # Install NVIDIA CUDA runtime and cuBLAS packages
        # These are officially distributed via pip by NVIDIA
        packages = [
            "nvidia-cublas-cu12",
            "nvidia-cuda-runtime-cu12",
            "nvidia-cudnn-cu12",
        ]

        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet"] + packages,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            if verbose:
                print(f"Warning: CUDA download failed: {result.stderr}")
            return False

        if verbose:
            print("CUDA libraries installed successfully.")

        return True

    except Exception as e:
        if verbose:
            print(f"Error downloading CUDA libraries: {e}")
        return False


def setup_gpu_if_needed(verbose: bool = True) -> tuple[bool, str]:
    """Set up GPU environment, downloading CUDA if necessary.

    This is the main entry point for GPU setup. It checks if GPU is
    available and ready, and automatically downloads CUDA libraries
    if they're missing.

    Args:
        verbose: If True, print progress messages.

    Returns:
        Tuple of (success, message)
    """
    # First check if GPU + CUDA already available
    try:
        import torch
        if torch.cuda.is_available():
            return True, "GPU already available"
    except ImportError:
        pass

    # Try to find existing CUDA DLLs
    if _find_cublas_dlls():
        return True, "CUDA DLLs found"

    # Try to download CUDA libraries
    if download_cuda_dlls(verbose=verbose):
        # Verify installation
        try:
            import torch
            if torch.cuda.is_available():
                return True, "GPU setup complete"
        except ImportError:
            pass

    return False, "GPU not available - will use CPU"


# Auto-setup on import - but only if explicitly requested via env var
if os.environ.get("YTT_AUTO_DOWNLOAD_CUDA", "").lower() in ("1", "true", "yes"):
    setup_gpu_if_needed(verbose=True)
