---
layout: post
title: Setting up a Razer Core X E-GPU to support CUDA on a Framework Laptop with Ubuntu 22.04
date: 2023-02-22 16:17:20 -0300
description: You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. # Add post description (optional)
img: yearofthelinuxgaminglaptop.png # Add image post (optional)
fig-caption: # Add figcaption (optional)
tags: [linux, nvidia, egpu, framework, FOSS]
---


## Introduction (Don't Try This on a Partition You Are Afraid to Break)

After some time, I was finally able to set up my egpu with my Framework laptop running Ubuntu 22.04. I tried previously with Ubuntu 20.04, following instructions which suggested manually changing Xorf.conf, and I nearly broke my partition, so I wouldn't recommend doing that. I will share many exact versions of the software I used here, as well as the steps I took, in the hope that this is helpful to others. Take these instructions with a grain of salt, because they are very software version dependent, but since I was able to get through the woods of nvidia Linux driver setup, it felt appropriate to leave some path markers. **⚠As a word of warning, don't try this on a partition you are afraid to break.** Make sure all your data is backed up. Graphics driver setup can break your boot sequence to an extent that its difficult for a non-expert user to fix it. You probably could still get your data out, but the path might be pretty technical so don't risk it. Another small note, if you are also a framework user, this is with an 11th gen intel mainboard. I have no idea if this works with 12th gen, I can only attest to it working on my 11th gen Framework.


## Software and Hardware Versions

OS: Ubuntu 22.04.1 LTS x86_64 \
Kernel: 5.15.0-58-generic \
CPU: 11th Gen Intel i7-1185G7 (8) \
GPU: NVIDIA GeForce GTX 1070 Ti  \
GPU: Intel TigerLake-LP GT2 \
Nvidia Driver Version: 525.85.12 \
CUDA Version: 12.0 \
EGPU Enclosure: Razer Core X 

## Thunderbolt Authorization


So you should power on your Razer Core X(with GPU inside) then connect it to your Framework with the thunderbolt cable before you boot the Framework. Then search open thunderbolt device management from the start menu, and set the status to unlocked so the device will allow access to the Razer Core X. This probably has security implications with respect to the secuurity of your usb-c ports. 

The Razer core X should be listed as an authorized device.

## Nvidia Driver Install

**⚠I'll repeat my warning here; don't try this on a partition you are afraid to break** 

First I checked what drivers were recommended for the connected 1070ti using the command. 

````
ubuntu-drivers devices
````

I picked the recommended driver, so that's what I recommend trying.

I recommend installing the recommended driver using apt. For me it was driver 525, but it may be different for you.
````
sudo apt-get install nvidia-driver-XXX
````

At this stage, don't try to manually switch the display system over the the graphics card, the apt install is sufficient.

*You should reboot after the nvidia driver install, if your boot is broken, probably time to give up and purge nvidia drivers*


## egpu-switcher

The next step is cloning and installing egpu-switcher, a tool designed to allow the same device to boot with or without the egpu connected. An important caveat is
that a design assumption of this software is that you will not unplug, or plug in the egpu after booting, it should have the same connection status for the duration of any
single uptime.

I followed the instructions for Manual installation in the README (it was not difficult, but you do have to install go first which you can do with apt)

[https://github.com/hertg/egpu-switcher](https://github.com/hertg/egpu-switcher)



When you run the final setup command

````
sudo egpu-switcher enable

````

It will prompt you for which device to designate as the egpu; you should see something like this;

````
Found 2 possible GPU(s)...

1: 	Vendor 8086 Device 9a49 (i915)
2: 	Vendor 10de Device 1b82 (nvidia)

Which one is your external GPU? [1-2]: 

````
You want to select the device labeled "nvidia"


## Sanity Checks
Once you are done with the egpu-switcher setup, you probably will want to shut down your laptop then reboot it first with the egpu enclosure plugged in then second without it plugged in. You want to make sure the boot works in both cases, and when you run:

````
nvidia-smi
````
on the boot without the egpu plugged in, you should see a message like:

````
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.

````
And when you run it on the boot with the egpu plugged in, you should see something like:

````
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  On   | 00000000:04:00.0 Off |                  N/A |
|  0%   41C    P5    12W / 180W |    270MiB /  8192MiB |      4%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|    0   N/A  N/A      1882      G   /usr/lib/xorg/Xorg                176MiB |
|    0   N/A  N/A      2193      G   /usr/bin/gnome-shell               76MiB |
+-----------------------------------------------------------------------------+

````

You should also repeat this sanity check after you have set up the cuda drivers. You don't want your laptop to unexpectedly break when you are away from your egpu.

You can also run the command:

````
sudo lshw -c video
````

If you booted without the egpu plugged in, you should see something like:

````
*-display                 
       description: VGA compatible controller
       product: TigerLake-LP GT2 [Iris Xe Graphics]
       vendor: Intel Corporation
       physical id: 2
       bus info: pci@0000:00:02.0
       logical name: /dev/fb0
       version: 01
       width: 64 bits
       clock: 33MHz
       capabilities: pciexpress msi pm vga_controller bus_master cap_list rom fb
       configuration: depth=32 driver=i915 latency=0 mode=2256x1504 resolution=2256,1504 visual=truecolor xres=2256 yres=1504
       ...
````

and if you booted with the egpu plugged in, you should see something like:

````
*-display                 
        description: VGA compatible controller
        product: TigerLake-LP GT2 [Iris Xe Graphics]
        vendor: Intel Corporation
        physical id: 2
        bus info: pci@0000:00:02.0
        logical name: /dev/fb0
        version: 01
        width: 64 bits
        clock: 33MHz
        capabilities: pciexpress msi pm vga_controller bus_master cap_list rom fb
        configuration: depth=32 driver=i915 latency=0 mode=2256x1504 resolution=2256,1504 visual=truecolor xres=2256 yres=1504
        ...
*-display
        description: VGA compatible controller
        product: GP104 [GeForce GTX 1070 Ti]
        vendor: NVIDIA Corporation
        physical id: 0
        bus info: pci@0000:04:00.0
        version: a1
        width: 64 bits
        clock: 33MHz
        capabilities: pm msi pciexpress vga_controller bus_master cap_list rom
        configuration: driver=nvidia latency=0
        ...
````

## Cuda Driver Setup

I followed the instructions here:

[https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

Fortunately, nothing broke.

## Checking Cuda Functionality With PyTorch

````
python3 -m venv env
source env/bin/activate
python3 -m pip install torch
python3
````

then from inside the python interpreter command line interface
````
import torch
torch.cuda.is_available()
````

If you see `True`, thats a pretty good sign, you can probably look up other ways to check, personally, I tried running inference on [whisper](https://github.com/openai/whisper) locally.

## Parting Words

Hopefully, this can be the year of the Linux gaming(deep neural network training) laptop. We should aim for an era where Linux is a first class citizien with respect to support for consumer scale high performance computing such as for gaming and (relatively speaking) small scale deep learning. End users who care little for technical setup should be able to have an operating system that respects their freedom, without having to become an expert in high performacne computing card driver installation and debugging. The ease-of-use gap between Windows and Linux has been dissapointing for a long time. I think Michael Hertig and any other egpu-switcher developers deserve a special thanks because egpu-switcher does the heavy lifting here, and the developers were kind enough to release it as free and open source software. 


## Feedback

As a final note, if you have corrections or want to add notes from your own experience, please add them as a PR on the source repository for this blog:

[https://github.com/balisujohn/balisujohn.github.io/](https://github.com/balisujohn/balisujohn.github.io/)