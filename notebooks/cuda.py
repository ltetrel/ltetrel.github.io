# %% [markdown]
'''
# How to install NVIDIA GPU driver and CUDA on ubuntu
'''
# %% [markdown]
'''
This post is intended to describe how to install NVIDIA GPU driver and CUDA on ubuntu.  
I have myself a GTX 560M with ubuntu 18.04.
'''
# %% [markdown]
'''
## Requirements

Before installing, we will first need to find the different version that your GPU support in the following order (in parenthesis my versions):
1. The GPU driver version (390.132)
2. CUDA version (9.0)
3. OS version (ubuntu 18.04) (and optionnally `gcc` version)

### GPU driver version

Got to [this page](https://www.nvidia.com/Download/index.aspx) and fill in all the information, and click on search to find the latest driver version your GPU support.

From here we will download the driver installer that we will use later.

Right-click on `Download` and `copy the link location`.
Back on your computer, paste the link to download the installer with `wget`:
```bash
wget https://www.nvidia.com/content/DriverDownload-March2009/confirmation.php?url=/XFree86/Linux-x86_64/390.132/NVIDIA-Linux-x86_64-390.132.run&lang=us&type=TITAN
```

### CUDA version

CUDA compatibility depends on your driver version you got in the previous section, check [here](https://docs.nvidia.com/deploy/cuda-compatibility/index.html#binary-compatibility) the compatibility matrix to know chich CUDA you can use.

### OS version

Check if your ubuntu version is compatible with this CUDA version.
You should select the right documentation [there](https://docs.nvidia.com/cuda/archive/)
and check the section `system-requirements`, for example https://docs.nvidia.com/cuda/archive/9.0/cuda-installation-guide-linux/index.html#system-requirements.

>**Warning**  
>If your ubuntu version is not compatible with CUDA, you should install the good gcc version.
>After [cheking the gcc version](https://docs.nvidia.com/cuda/archive/9.0/cuda-installation-guide-linux/index.html#system-requirements), you can install it and make it your default one:
>```bash
>sudo apt-get install build-essential gcc-6
>sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 6
>```

## Installation

Now that we know all the required version, we can start installing the softwares.

First, we make sure to have the build dependencies installed:
```bash
sudo apt-get install build-essential
```

Install the GPU driver by running the installer (we downloaded it previously):
```bash
sudo bash NVIDIA-Linux-x86_64-390.132.run
```

To download the CUDA toolkit installer, select the good release there:
https://developer.nvidia.com/cuda-toolkit-archive

Fill in the different informations, and at the end select `runfile (local)`:
https://developer.nvidia.com/cuda-90-download-archive?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1704&target_type=runfilelocal

>**Note**  
>In my case, my CUDA version does not support ubuntu 18.04 so I selected the newest distribution which is 17.04.

Now you can download and run the executable to install the CUDA toolkit.
```bash
wget https://developer.nvidia.com/compute/cuda/9.0/Prod/local_installers/cuda_9.0.176_384.81_linux-run
sudo bash cuda_9.0.176_384.81_linux-run
```

>**Warning**  
>After running the CUDA installer, you will be asked if you want to install the NVIDIA Accelerated Graphics Driver for Linux-x86_64. Of course don't do it, we already installed it previously.

You can now follow the [post-installation actions](https://docs.nvidia.com/cuda/archive/9.0/cuda-installation-guide-linux/index.html#post-installation-actions).
The most important is to add the CUDA paths to the environment variable (system wide), so depending on your CUDA version:
```bash
echo "export PATH=/usr/local/cuda-9.0/bin:/usr/local/cuda/bin\${PATH:+:\${PATH}}" | sudo tee -a /etc/profile.d/myenvvars.sh
echo "export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64:/usr/local/cuda/lib64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}" | sudo tee -a /etc/profile.d/myenvvars.sh
```
Optionnally check the cuda toolkit samples to test if CUDA is working.

To help the compiler when linking, add cuda libraries to the library path and check:
```bash
echo "/usr/local/cuda-9.0/lib64" | sudo tee /etc/ld.so.conf.d/cuda-9.0.conf
sudo ldconfig -v
```

Finally, reboot your computer:
```bash
sudo reboot
```
'''
# %% [markdown]
'''
# Tags
'''
# %% [markdown]
'''
Software-Development; GPU
'''