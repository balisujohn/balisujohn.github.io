---
layout: post
title: How to connect to GlobalProtect VPN Efficiently on Linux
date: 2021-10-08 16:17:20 -0300
description: You’ll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. # Add post description (optional)
img: relay.png # Add image post (optional)
fig-caption: # Add figcaption (optional)
tags: [linux, vpn]
---
For my first blog-post I'm going to present an extremely simple script that I've been using to make connecting to the vpn a bit easier. 

If you're a CS student or researcher at UW - Madison that uses a Linux daily driver, this post will be the most relevant to you. Even if you don't find the specific script in this blog-post useful, you can apply the techniques from this blog-post to create your own quality-of-life enhancing scripts on linux.

This blog-post assumes you already have the globalprotect client for linux installed. 

The first step is to create a scripts folder in your home directory.

````
cd ~
mkdir scripts
````

Inside your newly-created scripts folder, create a file titled vpn.bash, and copy the following script into the file.

````
#!/bin/bash

if [ "$1" == "--main" ]
then 
globalprotect connect --portal uwmadison.vpn.wisc.edu
elif [ "$1" == "--cs" ]
then 
globalprotect connect --portal compsci.vpn.wisc.edu
elif [ "$1" == "--off" ]
then 
globalprotect disconnect
fi
````

Once you've set up vpn.bash as specified, two steps are required make this script easily accessible from anywhere.

First, change directory into the scripts folder, then run the command

````
chmod +x ./vpn.bash
````
This will allow vpn.bash to be executed as a script

Then enter your home directory, and open .bashrc with the editor of your choice.

Add the following line:

````
alias vpn='~/scripts/vpn.bash'
````

Save your changes to .bashrc, then open a new terminal. You should now have access to the commands
````
vpn --main
````
,
````
vpn --cs
````
and 
````
vpn --off
````
from your terminal, regardless of what directory you are in. 

This script is very simple and provides nothing more than a set of aliases to other commands, but with a bit of bash or python scripting you can create aliases to powerful custom scripts. 

If you have feedback about this blog-post, feel free to email me at mylastname at wisc dot edu. Alternatively, feel free to directly pull request any corrections to the repository. This whole blog is released under a GPL3.0 license.