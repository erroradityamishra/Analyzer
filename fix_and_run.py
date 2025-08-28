import subprocess
import sys
import os
import urllib.request
import zipfile
import tempfile

def download_portable_python():
    """Download and setup portable Python"""
    print("🔧 Setting up portable Python environment...")
    
    # Create a portable directory
    portable_dir = os.path.join(os.getcwd(), "portable_python")
    if not os.path.exists(portable_dir):
        os.makedirs(portable_dir)
    
    # Download embeddable Python
    python_url = "https://www.python.org/ftp/python/3.11.9/python-3.11.9-embed-amd64.zip"
    zip_path = os.path.join(portable_dir, "python.zip")
    
    print("📥 Downloading Python embeddable...")
    try:
        urllib.request.urlretrieve(python_url, zip_path)
        print("✅ Download completed")
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return None
    
    # Extract Python
    print("📂 Extracting Python...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(portable_dir)
        os.remove(zip_path)
        print("✅ Extraction completed")
    except Exception as e:
        print(f"❌ Extraction failed: {e}")
        return None
    
    # Download get-pip.py
    pip_url = "https://bootstrap.pypa.io/get-pip.py"
    pip_path = os.path.join(portable_dir, "get-pip.py")
    
    print("📥 Downloading pip installer...")
    try:
        urllib.request.urlretrieve(pip_url, pip_path)
        print("✅ Pip installer downloaded")
    except Exception as e:
        print(f"❌ Pip download failed: {e}")
        return None
    
    # Enable pip in python._pth
    pth_file = os.path.join(portable_dir, "python311._pth")
    if os.path.exists(pth_file):
        with open(pth_file, 'a') as f:
            f.write("\nLib\\site-packages\n")
    
    python_exe = os.path.join(portable_dir, "python.exe")
    return python_exe

def install_packages_with_portable_python(python_exe):
    """Install required packages"""
    print("📦 Installing required packages...")
    
    # Install pip first
    try:
        subprocess.run([python_exe, "get-pip.py"], cwd=os.path.dirname(python_exe), check=True)
        print("✅ Pip installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Pip installation failed: {e}")
        return False
    
    # Install packages
    packages = [
        "streamlit",
        "pandas",
        "numpy", 
        "plotly",
        "seaborn",
        "matplotlib",
        "scikit-learn",
        "scipy",
        "requests",
        "google-generativeai"
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.run([python_exe, "-m", "pip", "install", package], check=True)
            print(f"✅ {package} installed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
    
    return True

def run_streamlit_app(python_exe):
    """Run the Streamlit application"""
    print("🚀 Starting Streamlit application...")
    
    try:
        # Run streamlit
        subprocess.run([python_exe, "-m", "streamlit", "run", "app.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to run Streamlit: {e}")
        return False
    
    return True

def main():
    print("🔧 PYTHON ENVIRONMENT REPAIR & APPLICATION LAUNCHER")
    print("=" * 60)
    print("Issue detected: Corrupted Python importlib module")
    print("Solution: Creating portable Python environment")
    print("=" * 60)
    
    # Download and setup portable Python
    python_exe = download_portable_python()
    if not python_exe:
        print("❌ Failed to setup portable Python")
        return
    
    # Install packages
    if not install_packages_with_portable_python(python_exe):
        print("❌ Failed to install packages")
        return
    
    # Run the application
    print("\n🎯 LAUNCHING DATA SCIENCE APPLICATION")
    print("=" * 60)
    print("Your application will open in your browser at:")
    print("🌐 http://localhost:8501")
    print("=" * 60)
    
    run_streamlit_app(python_exe)

if __name__ == "__main__":
    main()
