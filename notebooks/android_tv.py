# %% [markdown]
'''
# Use your Android TV without a Google account
'''
# %% [markdown]
'''
How to avoid registering to a Google account on your Android smart TV.
'''
# %% [markdown]
'''
## tl;dr
1. Install android developper tools on your PC
2. Enable dev mode on your TV and connect to it
3. Download .apk files and install them using `adb`
'''
# %% [markdown]
'''
## Introduction

Forced registration is a [dark pattern](https://en.wikipedia.org/wiki/Dark_pattern) that forces users to register to an account to use any kind of services.
Google is known for that kind of practices and they use it quite a lot on any android-powered devices (like your smart TV).

People are not necessarily aware that it is possible to completely avoid registering to Google to use your TV and android features, especially installing app.

## Requirements

To use this tutorial, you will need:
* A tv running android-tv OS (mine has Android TV 11 with kernel 4.19) connected to internet (ideally via ethernet)
* Another computer that will connect remotely to your TV, also connected to your network

The following instructions are for LINUX, but should be somehow similar to Windows.

### Install android developper tools

We first need to install the [Android Debug Bridge](https://developer.android.com/tools/adb) (adb) on your host PC.

```bash
sudo apt-get -y install android-tools-adb
```

### Enable developper mode on your TV

In the Android settings, find the section where you have your Android build information, and push `OK` a few times.

<img src="imgs/android_tv/dev_mode.png" width="400"/>

Quite a weird way to enable dev-mode, it reminds me the old days when putting some cheat codes on my GameBoy games...

If correct, there should be a message saying "you are now a developper".
There should be a new menu item "Developper options" where you can enable "USB debugging".

If any troubles check on [this website](https://docs.nvidia.com/gameworks/content/developertools/mobile/nsight_tegra/nsight_tegra_unlock_devmode.htm?ref=evanw.com).

## Install android apps

Android developers package their application uising the `.apk` file format, make sure to always select the 32-bit versions (armeabi-v7a versions) if you are not using an NVIDIA shield.

Where to find them ? There are a lot of website that provide repositories to distribute them but I am personnally using [apkmirror.com](https://www.apkmirror.com).
Always prefer developper mirrors or github/gitlab releases page (which is common), for example [SmartTube](https://github.com/yuliskov/SmartTube/releases/) does that.

### Sideload app to your TV

The action of installing an app directly to a device is called "side-loading".
After you have downloaded a bunch of apk files on your PC, it is time to install them!

>**Warning**  
>Always proceed with caution when downloading files from an untrusted source!

First identify the IP adress of your TV, it should be somehwere in **Settings > Network > About**.

Then, connect your TV with `adb` and check that the connection is successfull:
```bash
adb connect <IP>
adb devices
```

Accept the USB debugging prompt that will appear on your Android TV.

<img src="imgs/android_tv/usb-debugging-promt.jpg" width="400"/>

You can now sideload the package!
```bash
adb install /path/to/apk/file
```
'''
# %% [markdown]
'''
## To go further
'''
# %% [markdown]
'''
A whole new world is now open to you, you are not limited to only installing apk but you can actually customize your TV now!
For example, you can access and send files to the TV storage with `adb shell` (I use that to backup my kodi library environment).
This is quite usefull if you don't want to install additionnal apps on your TVs.
'''
# %% [markdown]
'''
# Tags
'''
# %% [markdown]
'''
Software-Development; Computer-Science; Media-Server
'''