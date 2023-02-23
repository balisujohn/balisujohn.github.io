---
layout: post
title: Setting up a Razer Core X E-GPU to support CUDA on a Framework Laptop with Ubuntu 22.04
date: 2023-02-22 16:17:20 -0300
description: You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. # Add post description (optional)
img: relay.png # Add image post (optional)
fig-caption: # Add figcaption (optional)
tags: [linux, nvidia, egpu, framework, FOSS]
---


## Introduction

After some time, I was finally able to set up my egpu with my Framework laptop running Ubuntu 22.04. I tried previously with Ubuntu 20.04, following instructions which suggesting manually changing Xorf.conf, and I nearly broke my partition, so I wouldn't recommend doing that. I will share as many exact versions of the software I used here, as well as the steps I took, in the hope that this is helpful to others. Take these instructions with a grain of salt, because they are very software version dependent, but since I was able to get through the woods of nvidia Linux driver setup, it felt appropriate to leave some path markers. *As a word of warning, don't try this on a partition you are afraid to break.* Make sure all your data is backed up, or that it is a new partition. Graphics driver setup can break your boot sequence to an extent that its difficult for a non-expert user to fix it. 


## Software and Hardware Versions

OS: Ubuntu 22.04.1 LTS x86_64 \
Kernel: 5.15.0-58-generic \
CPU: 11th Gen Intel i7-1185G7 (8) @ \
GPU: NVIDIA GeForce GTX 1070 Ti  \
GPU: Intel TigerLake-LP GT2 \
Nvidia Driver Version: 525.85.12 \
CUDA Version: 12.0 \
EGPU Enclosure: Razer Core X 

## Thunderbolt Authorization

*I'll repeat my warning here; don't try this on a partition you are afraid to break*

So you should power on your Razer Core X then connect it to your Framework with the thunderbolt cable before you boot the Framework. Then search open thunderbolt device management from the start menu, and set the status to unlocked so the device will allow access to the Razer Core X. This probably has security implications with respect to the secuurity of your usbc ports. \

The Razer core X should be listed as an authorized device.

## Nvidia Driver Install

First I checked what drivers were recommended for the connected 1070ti using the command. 

````
ubuntu-drivers devices
````

I picked the recommended driver, so that's what I recommend trying.

*You should reboot after the nvidia driver install, if your boot is broken, probably time to give up and purge nvidia drivers*


## egpu-switcher

The first step is cloning and installing egpu switcher, a tool designed to allow the same device to boot with or without the egpu connected. An important caveat is
that a design assumption of this software is that you will not unplug, or plug in the egpu after booting, it should have the same connection status for the duration of any
single uptime.

I followed the instructions for Manual installation in the README (it was not difficult, but you do have to install go first which you can do with apt)

https://github.com/hertg/egpu-switcher

When you run the command

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


## Sanity checks
Once you are done with the egpu-switcher setup, to probably will want to shut down your laptop then reboot it first with the egpu enclosure plugged in then second without it plugged in. You want to make sure the boot works in both cases, and when you run:

````
nvidia-smi
````
on the boot without the egpu plugged in, you should see a message like:

````
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.

````
And when you run it on the boot without the egpu plugged in, you should see something like:

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

## Cuda Driver Setup

I followed the instructions here:

https://developer.nvidia.com/cuda-downloads

Somehow, nothing broke.

## Checking cuda functionality with PyTorch

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

If you see true, thats a pretty good sign, you can probably look up other ways to check, personally, I tried running inference on whisper locally.

## Parting Words

Hopefully, this can be the year of the Linux gaming(deep neural network training) laptop. We should aim for an era where Linux is a first class citizien with respect to support for consumer scale high performance computing such as for gaming and (relatively speaking) small scale deep learning. End users who care little for technical setup should be able to have an operating system that respects their freedom, without having to become an expert in high performacne computing card driver installation and debugging. The ease-of-use gap between Windows and Linux has been dissapointing for a long time. I think Michael Hertig and any other egpu switcher developers deserve a special thanks because egpu-switcher program does the heavy lifting here, and the developers were kind enough to release this as free and open source software. 

As a final note, if you have corrections or want to add notes from your own experience, please add them as a PR on the source repository for this blog:

https://github.com/balisujohn/balisujohn.github.io/