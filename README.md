# Jetson nano configuration & getting ready for development

NVIDIA announced the Jetson Nano Developer Kit at the 2019 NVIDIA GPU Technology Conference (GTC), a $99 computer available for embedded designers, researchers, and DIY makers, delivering the power of modern AI in a compact, easy-to-use platform with full software programmability. Jetson Nano delivers 472 GFLOPS of compute performance with a quad-core 64-bit ARM CPU and a 128-core integrated NVIDIA GPU. It also includes 4GB LPDDR4 memory in an efficient, low-power package with 5W/10W power modes and 5V DC input, as shown in figure 1. An exciting piece of technology. I was a very early developer on the Jetson platform with the first generation the TK1. I started deploying CV edge solutions in 2003 and I witnessed the incredible evolution of CV and embedded solutions. Jetson nano is one of the most efficient embedded solutions (Cost, Power, Functionalities) for CV in the world. This repo's objective is to speed up the process of configuring the Jetson-nano and start developing some innovative and great applications. I will share only what worked, I tested and what I learned. And step by step learning process. My next step from here will be to move to Jetpack 4.3 & eventually Jetpack 4.4. I will also focus on Deepstream, TRT and TLT. I will be also exploring [Mediapipe](https://github.com/google/mediapipe/)
![Jetson nano](https://github.com/T-DevH/jetson-nano-tfdev/blob/master/images/nano.jpg)

## Hardware
- Jetson nano
- HDMA screen
- USB keyboard and mouse
- Wifi, complete kit
- Makeronics Developer Kit for Jetson Nano (Wifi included)
- SD card reader

## Build & Configuration
- Jetpack 4.2
- Python 3.6
- SciPy v1.3.3
- Tensorflow 1.13.1
- Keras 2.3.0
- Opencv 4.1.2
- dlib 19.19.0

I have specific reasons why I am using Tensorflow 1.13.1 with Jetpack 4.2 and its more to do with compatibility and stability with TensorRT. OpenCV is built from source with CUDA support. However, dlib doesn't support Jetson nano. I identified a work around and I will share my finding.

## Step 1: Flash NVIDIA’s Jetson Nano Developer Kit .img to a microSD
NVIDIA JetPack bundles most of the developer tools required on the Jetson platform, including system profiler, graphics debugger, and the CUDA Toolkit
- L4T R32.2 (K4.9)
- Ubuntu 18.04 LTS aarch64
- CUDA 10.0.326
- cuDNN 7.5.0.66
- TensorRT 5.1.6.1
- VisionWorks 1.6
- OpenCV 3.3.1
- Nsight Systems 2019.4
- Nsight Graphics 2019.2
- SDK Manager 0.9.13

Download Jetpack 4.2 from here. I am using my mac and a microSD card reader. You will need to download and install BalenaEtcher a disk image flashing tool. Insert the microSD into the card reader, and then plug the card reader into a USB port on your computer. Start balenaEtcher and proceed to flash. Once flashing is completed, eject and you are ready to move to step 2.

## Step 2: Boot Jetson Nano with the microSD
Insert the microSD into the Jetson Nano, connect thescreen, keyboard, mouse. Apply power. Insert the power plug of your power adapter into your Jetson Nano (use the J48 jumper if using a 20W barrel plug supply). You will see the NVIDIA + Ubuntu 18.04 desktop, you should configure your wired or wireless network settings, langage and basic linux configuration including setting a password.

## Step 3: Set SSH
You have two options here, you can use the Jetson nano terminal to continue the configuration or use SSH. Let's st an SSH cession for convinience. 
From the nano, start a terminal cession and enter follwing commands:
```
$ cd ~
$ whoami
tarik-dev
$ ifconfig
wlan0: flags=4163<UP,BROADCAST,RUNNING,MULTICAST>  mtu 1500
        inet 17.0.0.88  netmask 255.255.255.0  broadcast 10.0.0.255
        inet6 2601:647:5a00:7d50:bdc8:edf4:6e5d:268f  prefixlen 64  scopeid 0x0<global>
        inet6 2601:647:5a00:7d50:4537:1484:4c5a:eacd  prefixlen 64  scopeid 0x0<global>
        inet6 2601:647:5a00:7d50:4c27:4824:ba02:4a14  prefixlen 64  scopeid 0x0<global>
        inet6 2601:647:5a00:7d50:3db:bcdb:bdcd:c918  prefixlen 64  scopeid 0x0<global>
        inet6 2601:647:5a00:7d50::dad  prefixlen 128  scopeid 0x0<global>
        inet6 fe80::efa4:3feb:ece3:62de  prefixlen 64  scopeid 0x20<link>
        ether 90:61:ae:5c:d8:e3  txqueuelen 1000  (Ethernet)
        RX packets 2350277  bytes 3290716941 (3.2 GB)
        RX errors 0  dropped 0  overruns 0  frame 0
        TX packets 193279  bytes 27324343 (27.3 MB)
        TX errors 0  dropped 0 overruns 0  carrier 0  collisions 0
```
My IP address is 17.0.0.88; On my remote machine, my mac in this case, I ssh to my jetson nano and I enter the nano password set when 
the nano was configured.
```
$ ssh tarik-dev@17.0.0.88
tarik-dev@17.0.0.88's password: 
Welcome to Ubuntu 18.04.4 LTS (GNU/Linux 4.9.140-tegra aarch64)

 * Documentation:  https://help.ubuntu.com
 * Management:     https://landscape.canonical.com
 * Support:        https://ubuntu.com/advantage
This system has been minimized by removing packages and content that are
not required on a system that users do not log into.

To restore this content, you can run the 'unminimize' command.

3 packages can be updated.
0 updates are security updates.


1 updates could not be installed automatically. For more details,
see /var/log/unattended-upgrades/unattended-upgrades.log
*** System restart required ***
Last login: Thu Apr 16 11:53:41 2020 from 10.0.0.132
tarik-dev@tarikdev-desktop:~$ 
```
Lets move to the next step now.

## Step 4: Update your system
We will set the Jetson nano to use maximum power capacity.
```
$ sudo nvpmodel -m 0
$ sudo jetson_clocks
```
Update system level packages before starting the next step and installing software.
```
$ sudo apt-get update && sudo apt-get upgrade
```

## Step 5: Install system-level dependencies
We will start by installing development tools
```
$ sudo apt-get install git cmake
$ sudo apt-get install libatlas-base-dev gfortran
$ sudo apt-get install libhdf5-serial-dev hdf5-tools
$ sudo apt-get install python3-dev
$ sudo apt-get install nano locate
```
Let's install now SciPy prerequisites and a system level Cython library. 
```
sudo apt-get install libfreetype6-dev python3-setuptools
$ sudo apt-get install protobuf-compiler libprotobuf-dev openssl
$ sudo apt-get install libssl-dev libcurl4-openssl-dev
$ sudo apt-get install cython3
```
I will be installing few XML tools for working with TensorFlow Object Detection (TFOD) API projects:
```
$ sudo apt-get install libxml2-dev libxslt1-dev
```

## Step 6: Update CMake
We will need a newer version of Cmake in order to compile OpenCV. Lets download and extract Cmake.
```
$ wget http://www.cmake.org/files/v3.13/cmake-3.13.0.tar.gz
$ tar xpvf cmake-3.13.0.tar.gz cmake-3.13.0/
& let's compile now
$ cd cmake-3.13.0/
$ ./bootstrap --system-curl
$ make -j4
```
Next step will be to update the bash profile:
```
$ echo 'export PATH=/home/nvidia/cmake-3.13.0/bin/:$PATH' >> ~/.bashrc
$ source ~/.bashrc
```

## Step 7: Install OpenCV system-level dependencies and other development dependencies
We will install now OpenCV dependecies on our system beginning with tools needed to build and compile OpenCV with parallelism:
```
$ sudo apt-get install build-essential pkg-config
$ sudo apt-get install libtbb2 libtbb-dev
```
Codecs and useful image libraries:
```
$ sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev
$ sudo apt-get install libxvidcore-dev libavresample-dev
$ sudo apt-get install libtiff-dev libjpeg-dev libpng-dev
```
We install now a selection of GUI libraries
```
$ sudo apt-get install python-tk libgtk-3-dev
$ sudo apt-get install libcanberra-gtk-module libcanberra-gtk3-module
```
The last task for this step is to install Video4Linux (V4L) in case we are using USB webcams and install a library for FireWire cameras:
```
$ sudo apt-get install libv4l-dev libdc1394-22-dev
```

## Step 8: Set up Python virtual environments on your Jetson Nano
The main purpose of Python virtual environments is to create an isolated environment for Python projects. 
This means that each project can have its own dependencies, regardless of what dependencies every other project has. 
Imagine that you have an application which is fully developed and you do not want to make any change to the libraries it is using 
but at the same time you start developing another application which requires the updated versions of those libraries. 
What will you do ? It is where virtualenv comes into play. 
It creates isolated environments for your python application and allows you to install python libraries in that isolated environment 
instead of installing them globally. Let's install Python package management tool, pip:
```
$ wget https://bootstrap.pypa.io/get-pip.py
$ sudo python3 get-pip.py
$ rm get-pip.py
```
And then we’ll install next tools for managing virtual environments, virtualenv and virtualenvwrapper
```
$ sudo pip install virtualenv virtualenvwrapper
```
The virtualenvwrapper tool is not fully installed until you add information to your bash profile. 
Go ahead and open up your ~/.bashrc with the nano ediitor: We will need to add this the the bash profile in order for the 
virtualenvwrapper tool to be installed
```
$ nano ~/.bashrc
```
Copy and paste the following to the bash file:
```
# virtualenv and virtualenvwrapper
export WORKON_HOME=$HOME/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
source /usr/local/bin/virtualenvwrapper.sh
```
Save and exit the file and then load the bash profile to finish the virtualenvwrapper installation
```
$ source ~/.bashrc
```
We are done, both virtualenv and virtualenvwrapper are installed.

## Step 9: Create my "nanocv" virtual environement
The main command to manage virtualenv are
- mkvirtualenv: Create a Python virtual environment
- lsvirtualenv: List virtual environments installed on your system
- rmvirtualenv: Remove a virtual environment
- workon: Activate a Python virtual environment
- deactivate: Exits the virtual environment taking you back to your system environemnt 

I will create now a virtual environement for my next step, in my case I named it nanocv.
```
$ mkvirtualenv nanocv -p python3
```
Let's activate my nanocv virtual environemnt
```
tarik-dev@tarikdev-desktop:~$ workon nanocv
(nanocv) tarik-dev@tarikdev-desktop:~$ 
```
For the remaining of this instalation/configuration it's **important to be on the virtual environemnt**.

## Step 10: Install the Protobuf Compiler
An efficient instalation of `protobuf` and `libprotobuf` is critical in order for `Tensorflow` to operate efficiently. Issues with `Protubuf`
will lead to performance degradation. More details in nvidia developer forum discussing this topic. Let's download and install the protobuf compiler. 
```
$ wget https://raw.githubusercontent.com/jkjung-avt/jetson_nano/master/install_protobuf-3.6.1.sh
$ sudo chmod +x install_protobuf-3.6.1.sh
$ ./install_protobuf-3.6.1.sh
```
It will take at least an hour to compile, once completed we will need to install inside our virtual envoronemnt:
```
$ workon nanocv 
$ cd ~
$ cp -r ~/src/protobuf-3.6.1/python/ .
$ cd python
$ python setup.py install --cpp_implementation
```
An important observation here, we are not using generic precompiled binaries via pip but more optimized compilation for 
nano using setup.py.

## Step 11: Install TensorFlow, Keras, NumPy, and SciPy
Lets make sure we are in the virtual environment:
```
$ workon nanocv 
```
Let's start by installing NumPy and Cython:
```
$ pip install numpy cython
```
Let's test the instalation
```
$ python
>>> import numpy
```

The next step is to install SciPy, we will use setup.py
```
$ wget https://github.com/scipy/scipy/releases/download/v1.3.3/scipy-1.3.3.tar.gz
$ tar -xzvf scipy-1.3.3.tar.gz scipy-1.3.3
$ cd scipy-1.3.3/
$ python setup.py install
```
We are ready now to install `Tensorflow`. I had to do some research on this and it will be productive to consult the follwing links:
- https://forums.developer.nvidia.com/t/converting-tf-2-0-saved-model-for-tensorrt-on-jetson-nano/107472 
- https://forums.developer.nvidia.com/t/inferencing-with-a-custom-made-keras-tensorflow-model-on-jetson-nano/112316/2 
- https://jkjung-avt.github.io/jetpack-4.3/ 

I will be using Tensorflow 1.13 along with Jetpack 4.2. I have enough evidence it will work. 

My next step after this repo is to evaluate Jetpack 4.3 with associated TensorRT and test comaptibility with Tensorflow 2.0 on Jetson nano. 
Let's install Tensorflow:
```
$ pip install --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v42 tensorflow-gpu==1.13.1+nv19.3
```
**IMPORTANT**: Here is a potential issue. I tried to optimize a model with TensorRT, to my surprize I had the following error:
```
WARNING: Tensorflow:TensortRT mismatch. Compiled against version 5.0.6, but loaded 5.1.6. Things may not work 
```
Please refer to Jupyter note book example: Detection. I was able to use TRT successfuly. 

Let's install  Keras:
```
$ pip install keras
```

## Step 12: Install the TensorFlow Object Detection API
Make sure you are in your `nanocv` virtual environment. We will install NVIDIA's `tf_trt_models`. 
The models are sourced from the TensorFlow models repository and optimized using TensorRT for Jetson.
```
$ cd ~
$ workon nanocv
```
And we clone the models from Tensorflow repo
```
git clone https://github.com/tensorflow/models
```
Now, we install the COCO API
```
$ cd ~
$ git clone https://github.com/cocodataset/cocoapi.git
$ cd cocoapi/PythonAPI
$ python setup.py install
```
We compile the `Protobuf` libraries used by the TFOD API
```
$ cd ~/models/research/
$ protoc object_detection/protos/*.proto --python_out=.
```
We will need to configure a script `setup.sh`. 
This script will be needed each time we use the TFOD API for deployment on the Jetson Nano.
```
$ nano ~/setup.sh
```
and insert the follwing lines, save and exit
```
#!/bin/sh
export PYTHONPATH=$PYTHONPATH:/home/`whoami`/models/research:\
/home/`whoami`/models/research/slim
```
We are ready to install `tf_trt_models`

## Step 13: Install NVIDIA’s `tf_trt_models`
Let's make sure we are in our virtual environment
```
$ workon nanocv
```
Let's clone the optimized nvidia optimized models and execute the instalation script
```
$ cd ~
$ git clone --recursive https://github.com/NVIDIA-Jetson/tf_trt_models.git
$ cd tf_trt_models
$ ./install.sh
```
**IMPORTANT**. If you are in `virtualenv` make sure your edit `install.sh` before you execute the script, 
and make sure you remove --user. The problem is documented [here](https://stackoverflow.com/questions/61336332/tensorflow-trt-models-installation-problem/61338313#61338313)

```
....
echo $PROTOC
$PROTOC object_detection/protos/*.proto --python_out=.
$PYTHON setup.py install --user
popd

pushd $MODELS_DIR/research/slim
echo $PWD
echo "Installing slim library"
$PYTHON setup.py install --user
popd

echo "Installing tf_trt_models"
echo $PWD
$PYTHON setup.py install --user
```

## Step 14: Install OpenCV 4.1.2

Let’s download the OpenCV source code from GitHub:
```
 $ workon nanocv    
 $ cd ~
 $ wget -O opencv.zip https://github.com/opencv/opencv/archive/4.1.2.zip
 $ wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.1.2.zip
 ```
Extract the files and renaming the directories for simplicity:
```
 $ unzip opencv.zip
 $ unzip opencv_contrib.zip
 $ mv opencv-4.1.2 opencv
 $ mv opencv_contrib-4.1.2 opencv_contrib
 ```
Lets move to OpenCV directory, and create a build directory:
```
 $ cd opencv
 $ mkdir build
 $ cd build
 ```
While making sure we are in the build directory `opencv/build` and in the virtual environment enter `CMake` command:
```
 $ cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D WITH_CUDA=ON \
      -D CUDA_ARCH_PTX="" \
      -D CUDA_ARCH_BIN="5.3,6.2,7.2" \
      -D WITH_CUBLAS=ON \
      -D WITH_LIBV4L=ON \
      -D BUILD_opencv_python3=ON \
      -D BUILD_opencv_python2=OFF \
      -D BUILD_opencv_java=OFF \
      -D WITH_GSTREAMER=ON \
      -D WITH_GTK=ON \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      -D BUILD_EXAMPLES=OFF \
      -D OPENCV_ENABLE_NONFREE=ON \
      -D OPENCV_EXTRA_MODULES_PATH=/home/`whoami`/opencv_contrib/modules ..  
```

Compilation process with Make, this type will take around 2.5 hours:

```
$ make -j4
```
![Compiling with Make](https://github.com/T-DevH/jetson-nano-tfdev/blob/master/images/Screenshot%20from%202020-04-17%2010-37-59.png)

Finish installation with install command:
```
$ sudo make install
```
Let's create a symbolic link from OpenCV’s installation directory to the virtual environment:

```
$ cd ~/.virtualenvs/py3cv4/lib/python3.6/site-packages/
$ ln -s /usr/local/lib/python3.6/site-packages/cv2/python-3.6/cv2.cpython-36m-aarch64-linux-gnu.so cv2.so
```

Note: I had some trouble here and I end up renaming the file cv2.cpython-36m-aarch64-linux-gnu to cv2.so first then linked the instalation.

```
$ ln -s /usr/local/lib/python3.6/site-packages/cv2/python3.6/cv2.so cv2.so
```
OpenCV is installed.

## Step 15: Install other libraries via pip

Let's install the following packages for machine learning, image processing, and plotting:
```
$ pip install matplotlib scikit-learn
$ pip install pillow imutils scikit-image
```
The next step is to install dlib. Dlib is a modern C++ toolkit containing machine learning algorithms and tools for creating 
complex software in C++. It is used in both industry and academia in a wide range of domains including robotics, embedded devices, 
mobile phones, and large high performance computing environments. It is principally a C++ library, however, you can use a number of 
its tools from python applications. For more details I suggest to consult the Python API. 
The challenge we face right now is the fact that dlib does not support the Nano’s GPU. 😰. A pip install won't include CUDA capability.

```
$ pip install dlib
```
By consulting this nvidia devtalk forum I foung a workaround. And here is a solution: You can download and extract files from source:

```
$ wget http://dlib.net/files/dlib-19.16.tar.bz2
$ tar jxvf dlib-19.16.tar.bz2
```
Before compiling let's access and edit a specific file `cudnn_dlibapi.cpp`:

```
$ cd dlib-19.17
$ nano dlib/cuda/cudnn_dlibapi.cpp
```
Let's search for the follwing line code `forward_algo = forward_best_algo` and comment it out:

```
dnn_prefer_fastest_algorithms()?CUDNN_CONVOLUTION_FWD_PREFER_FASTEST:CUDNN_CONVOLUTION_FWD_NO_WORKSPACE,
                 std::numeric_limits<size_t>::max(),
                 &forward_best_algo));
-                forward_algo = forward_best_algo;
+                //forward_algo = forward_best_algo;
                 CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize( 
                 context(),
                 descriptor(data),
```

Save the file, close the editor, and go back to the Terminal window. Next, run these commands to compile and install dlib:

```
$ sudo python3 setup.py install
```

I tested the dlib install(WORKED) but decided not to go for it for this build and install dlib with out CUDA support. 
I plan to put more time to do more testing with dlib on Jetson nano.

Let's install now Flask, a Python micro web server; and Jupyter, a web-based Python environment:

```
$ pip install flask jupyter
```

And finally installing XML tool for the TFOD API, and progressbar

```
$ pip install lxml progressbar2
```

We are done 😀

# Testing

## Tensorflow & Keras
```
$ workon nanocv
$ python
>>> import tensorflow
>>> import keras
>>> print(tensorflow.__version__)
1.13.1
>>> print(keras.__version__)
2.3.0
```
## Tensorflow Object Detection/Classification and TensorRT
I included two Jupyter notebooks from `tf_trt_models` for object detection and classification. 
- [Detection](https://github.com/T-DevH/Jetson-nano-step1/blob/master/detection.ipynb)
- [Classification](https://github.com/T-DevH/Jetson-nano-step1/blob/master/classification.ipynb)

## OpenCV & Dlib
```
$ workon nanocv
$ python
>>> import cv2
>>> import imutils
>>> image = cv2.imread("Galactica.jpg")
>>> image = imutils.resize(image, width=400)
>>> message = "OpenCV Jetson-Nano AI!"
>>> font = cv2.FONT_HERSHEY_SIMPLEX
>>> _ = cv2.putText(image, message, (30, 130), font, 0.7, (0, 255, 0), 2)
>>> cv2.imshow("Penguins", image); cv2.waitKey(0); cv2.destroyAllWindows()
```

## Camera
```
$ workon nanocv
$ python test_camera_nano.py
```
![camera test](https://github.com/T-DevH/jetson-nano-tfdev/blob/master/images/cameratest.png)
## References & useful links
- [Jetson Nano Brings AI Computing to Everyone](https://devblogs.nvidia.com/jetson-nano-ai-computing/)
- [Pyimagesearch](https://www.pyimagesearch.com/category/embedded/)
- [Opencv](https://opencv.org/)
- [Dlib](http://dlib.net/)
- [Building Dlib on nano](https://forums.developer.nvidia.com/t/issues-with-dlib-library/72600/46)
