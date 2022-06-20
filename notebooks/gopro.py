# %% [markdown]
'''
# Access the GoPro videos without the app
'''
# %% [markdown]
'''
This post will guide you step by step on how to access your video files over LAN wifi without the official app, from your GoPro directly to your computer.
'''
# %% [markdown]
'''
## Introduction

GoPro is a well known action camera founded by Woodman Labs. As many of big manufacturers, they like to build their own eco-system to keep their clients inside it
(so client won't easilly switch product because they are used to the frontend).
This is a pure marketing technic that personally I hate, and make simple things as copying files from one device to another a living hell (hi Apple!).

In this case, we have videos captured from a GoPro, and we want to transfer them to our computer.
The easiest way is to download the GoPro software or app, and download from there.
If you are like me and don't like to install too useless softwares on your computer to keep it clean (so it is not a garbage), follow this guide!

## Requirements

Before you start make sure you have a stable wifi, and enable it on your desktop.
Since we will use wifi to connect to the GoPro devide, if you are not using ethernet you will lose your internet connection.
Optionnally, if you want to automatically scrap all the content on your desktop, you will need to install [wget](https://manpages.ubuntu.com/manpages/bionic/man1/wget.1.html).

```bash
sudo apt install wget
```

### GoPro in server mode

If you have the app installed, they ask you to put your GoPro ready in application mode.
What happens is that your GoPro is actually creating a local web server, then the app access this server and let you download the files.
Instead, we will bypass the app and access the GoPro server directly!

1. Power the GoPro, so it will not shut down during the entire process.
2. Make sure that you enabled remote connections under `settings/connection/remote connections/yes`
3. Go into application mode with `settings/connections/connect a peripheral/gopro application`

>**Warning**  
>Pluging your GoPro is important because when the device is put into application mode, it won't put itself on sleep. Be also carefull that the camera (and mostly the lens!) doesn't get too hot.

### Downloading the video files

Now we want to access the server to be able to download files.
Place the GoPro near your desktop, then:

1. On you desktop, connect to the new wifi named `GPXXXXXXXX` (with 8 digits).
2. You will find the credentials for the wifi under `settings/connections/camera info`.

When you are connected, you should be able to access locally your file at http://10.5.5.9:8080/videos/DCIM/100GOPRO/.
Voila! You have access to your files and you can play them on your browser, or manually download them.

>**Note**  
>The actual important video files are the `*.MP4`, the other files `*.LRV` are a specific GoPro format used by their app for previews.

### Scrap the files

Of course, downloading the files one by one is a time consuming process.
Ideally you would like to get all the videos directly inside a folder:

```bash
cd my/folder
wget -A .MP4 -r --no-parent http://10.5.5.9:8080/videos/DCIM/100GOPRO/
```

>**Note**  
>Normally it should download at relatively high speed (>5MB/s), but it depends of the speed of your LAN network.

### (optionnally) Transcode the media

There is a high chance that you will not be able to play the media on your browser. This is because GoPro uses HEVC format (which is the update from h264) by default, and most browser does not support this format.
If you need more information about the HEVC format, check [this amazing post](https://developer.mozilla.org/en-US/docs/Web/Media/Formats/Video_codecs) from mozilla, but to summarize,
it allows a more efficient compression compare to its previous version.

To disable HEVC, you can do that under the GoPro seetings in `general/video compression/h.264 and HEVC`.
If you have you files already in HEVC, you can still transcode it to h.264 using [ffmpeg](https://ffmpeg.org/).
As an example if you [have already compiled ffmpeg with nvidia]({% link _posts/2000-01-01-jellyfin.html %}#Installing-FFmpeg):
```bash
ffmpeg -vsync 0 -hwaccel cuda -hwaccel_output_format cuda -i GXxxxxxx.MP4 -map 0:a:0 -c:a copy -map 0:v:0 -c:v:0 h264_nvenc -preset slow -b:v 36M GXxxxxxx_h264.mp4
```

To guide you on which options to use for compression, check [the ffmpeg wiki](https://trac.ffmpeg.org/wiki/Encode/H.264).

'''
# %% [markdown]
'''
## To go further
'''
# %% [markdown]
'''
Because the GoPro acts as a server, there is nothing stopping you accessing the media on another device than your desktop.
For example you could view the videos on your tv with a kodi device, remotely and without buying the hdmi cable.
'''
# %% [markdown]
'''
# Tags
'''
# %% [markdown]
'''
Software-Development; Media-Server
'''