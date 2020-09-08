# %% [markdown]
'''
# Build your own media server
'''
# %% [markdown]
'''
Streaming and sharing media are gaining popular adoption, specifically using popular web media platform that everyone knows: [Netflix](https://www.netflix.com/), [YouTube](https://www.youtube.com/) or [Spotify](https://www.spotify.com/ca-en/).
They all exists thanks to the same technology, media encoding/decoding.

Streaming has many advantages, maybe you want to share your personnal pictures with your family (while being in controll of the server), access music with a nice front-end, or easilly watch your movies anywhere on your house.
But did you knew that is was possible to make your own media server for FREE. 

You may have heard of [Plex](https://www.plex.tv/) which is a popular and free media server. But you need to understand that 1) it is not open-source and 2) Plex collects and stores information about you (check the [user agreement page](https://www.plex.tv/en-ca/about/privacy-legal/privacy-preferences/)). Let me introduce you instead [Jellyfin](https://jellyfin.org/), which does not store anything about you, and is trully free and open-source. Yes, you will mostly have lot of headaches, things will not work as smoothly as Plex and their UI is worst but at least your privacy should be safe.

>**Disclaimer**  
>I am not promoting and don't support downloading and distributing copyrighted content digitally, without copyright holder permission.
'''
# %% [markdown]
'''
## tl;dr
1. Install jellyfin by following [here](https://jellyfin.org/docs/general/administration/installing.html).
2. Enable hardware video with the [NVIDIA ffmpeg documentation](https://developer.nvidia.com/ffmpeg)
'''
# %% [markdown]
'''
## Installing Jellyfin

<img src="imgs/jellyfin/jellyfin.png" alt="drawing" width="300"/>

The following implies that you already installed an [ubuntu 18.04 server OS](http://cdimage.ubuntu.com/ubuntu/releases/18.04/release/) on a computer.

All the `Jellyfin` installation procedure is described [here](https://jellyfin.org/docs/general/administration/installing.html).

First, you will need to install HTTPS transport for APT if you haven't already:
```bash
sudo apt install apt-transport-https
```

Enable the Universe repository to obtain all the FFMpeg dependencies:
```bash
sudo add-apt-repository universe
```

Import the GPG signing key (signed by the `Jellyfin` Team):
```bash
wget -O - https://repo.jellyfin.org/ubuntu/jellyfin_team.gpg.key | sudo apt-key add -
```

Add a repository configuration at `/etc/apt/sources.list.d/jellyfin.list`:
```bash
echo "deb [arch=$( dpkg --print-architecture )] https://repo.jellyfin.org/ubuntu $( lsb_release -c -s ) main" | sudo tee /etc/apt/sources.list.d/jellyfin.list
```

>**Note**  
>Supported releases are xenial, bionic, cosmic, disco, and eoan.

Update APT repositories:
```bash
sudo apt update
```

Install `Jellyfin`:
```bash
sudo apt install jellyfin
```

Note the ip of your device:
```bash
ip a
```
You can see your IP adress by checking under `enp2s0`, it should be on the form `192.168.X.X`.

Congratulations!  
You have now access to your `Jellyfin` server through the web on any device (tablet, smartphone etc..).
Open a browser and type `http://<your_ip>:8096` to access it.

>**Note**  
>The server is running on your LAN network thanks to your router, it is not accessible from outside of your home Wi-Fi.
>If you want to make your server available anywhere, check the [reverse proxy setup](https://jellyfin.org/docs/general/networking/nginx.html#nginx).

Optionnally, you can check the status of the server using:
```bash
sudo service jellyfin status
```

If you want to start/stop the `Jellyfin` server:
```bash
sudo service jellyfin stop
sudo service jellyfin start
```

## Enabling hardware encoding

<img src="imgs/jellyfin/computer_hardware.jpg" width="400"/>

By default, `Jellyfin` will use software for video encoding/decoding. It uses the CPU and can be slow because it is not optimized for your hardware. This is why it is interresting to enable hardware acceleration for transcoding your medias.  
The latter will explain you how to enable hardware acceleration on your GPU NVIDIA card.

First, check if you have an NVIDIA compatible card:
```bash
lshw -class display
```

Basically, FERMI and newer cards allow NVDEC (hardware decoding), KEPLER and newer allows for NVENC (hardware encoding)
https://developer.nvidia.com/video-encode-decode-gpu-support-matrix.

>**Note**  
>Hardware accelaration is of course also possible on CPUs. You should check the official documentation for [CPU hardware acceleration](https://jellyfin.org/docs/general/administration/hardware-acceleration.html) (Intel, AMD etc..).

### Installing FFmpeg

#### Prerequesites

[FFmpeg](https://ffmpeg.org/) is a popular open-source multimedia streaming software, mostly used to do video encoding and decoding. There is a high chance that all the company I mentionned in the beginning are using it.

We will compile our own FFmpeg to make the best used of our GPU, but you will firt need to make sure that you have CUDA installed.
If not, [check my post on installing cuda](cuda_install.md) for your ubuntu system.

We also need to make sure that the build tools are available on the system:
```bash
sudo apt-get update
sudo apt-get install build-essential pkg-config
```

All the following is partially based from the [NVIDIA ffmpeg documentation](https://developer.nvidia.com/ffmpeg).

Download the `ffnvcodec` git repository:
```bash
git clone https://git.videolan.org/git/ffmpeg/nv-codec-headers.git
cd nv-codec-headers
```

You should now check which version (git branch) of the codec is compatible with your NVIDIA driver version [on the github repo](https://github.com/FFmpeg/nv-codec-headers/tree/master). Checkout to the repo (for me it is `sdk/8.0`) and build it:
```bash
git checkout sdk/8.0
cd nv-codec-headers && sudo make install && cd ..
```

#### Compiling FFmpeg

Download the latest `ffmpeg` source code, by cloning the corresponding git repository:
```bash
git clone https://git.ffmpeg.org/ffmpeg.git
cd ffmpeg
```

Use the following configure command:
```bash
./configure --prefix=/usr/lib/ffmpeg --target-os=linux --disable-doc --disable-ffplay --disable-shared \
--disable-libxcb --disable-vdpau --disable-sdl2 --disable-xlib --enable-gpl --enable-version3 --enable-static \
--enable-libx264 --enable-libx265 --arch=amd64 --enable-cuda-nvcc --enable-cuvid --enable-nvenc --enable-nonfree \
--enable-libnpp --extra-cflags=-I/usr/local/cuda/include --extra-ldflags=-L/usr/local/cuda/lib64
```

>**Warning**  
>It is possible that the `nv-codec-headers` version does not match the `ffmpeg` version.
>You should checkout to a `ffmpeg` version that is compatible. By comparing the commits date, I had to change Fmpeg to `v4.0` :
>```bash
>git checkout n4.0
>```

>**Note**  
>You can add `--disable-nvenc` if your GPU is not compatible with hardware encoding (FERMI architecture).

You are now ready to compile FFMpeg (should take few minutes!):
```bash
make -j 2
```

And copy the binaries into the appropriate folder,
```bash
sudo cp ffmpeg /usr/lib/ffmpeg/bin/
sudo cp ffprobe /usr/lib/ffmpeg/bin/
```

#### Verifying installation

You can first check if the nvidia codecs were properly installed,

```bash
ffmpeg -codecs | grep h264
```
You should see `h264_cuvid` and `h264_nvenc` in the output.

To test that the decoding works on GPU, run this benchmark on a sample `test.mp4` video:
```bash
ffmpeg -v debug -y -hwaccel cuvid -c:v h264_cuvid -i test.mp4 -benchmark -f null -
```

The CPU usage should be really low, and you should the encode/decode cores of the GPU running at 100\% (check encoder and decoder stats):
```bash
nvidia-smi -i 0 -q -d UTILIZATION -lms 500
```

To enable hardware encoding on `jellyfin`, on the admin dashboard select transcoding.
Depending on the architecture and the GPU, you will not support every format.
Check the box on jellifyn app depending on https://developer.nvidia.com/video-encode-decode-gpu-support-matrix 
and http://developer.download.nvidia.com/assets/cuda/files/NVIDIA_Video_Decoder.pdf

To further optimize your server usage (the worst being transcoding), you should use a codec and a container that is supported with almost any browser.
The ideal is the `H.264 8Bit` video (libx264), `AAC` audio, `mp4` container with `SubRip Text` (srt) subtitles codecs.
Check [here](https://jellyfin.org/docs/general/clients/codec-support.html) for information on what the codes that browsers supports.
'''
# %% [markdown]
'''
# Tags
'''
# %% [markdown]
'''
Computer-Science; Media-Server; Open-Science
'''