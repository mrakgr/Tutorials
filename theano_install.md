#Theano Windows Install Guide.

A year ago when I was starting out with ML, I tried installing Theano for the first time and did not have much luck. Now that I have some experience with Linux, things went much more smoothly. The trouble with Theano's installation guide is that it is fairly massive, overwhelming and even outdated – the exact opposite of a good guide. Actually, I had more luck ignoring the specific instructions and following my intuition.

Time of writing this guide: 5/13/2016.

Prerequisites:

Rapid Environment Editor: http://www.rapidee.com/en/about

For environment variable editing.

Visual Studio 2013:  https://www.visualstudio.com/en-us/downloads/download-visual-studio-vs.aspx

The VS2015 C++ compiler is not currently compatible with Nvidia compiler.  Cuda 8.0 SDK which has not been released at the time of writing might be compatible with VS2015, so check if the latest Cuda SDK is VS2015 compatible and get VS2015 instead if it is.

As Theano calls `nvcc` (Nvidia C++ Cuda compiler) from the command line and `nvcc` then calls `cl` (Microsoft C++ compiler) also from the command line, add `C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\bin` or the like to `PATH` so `nvcc` can find it.

CUDA 7.5 SDK: https://developer.nvidia.com/cuda-downloads

Python Anaconda: https://www.continuum.io/downloads#_windows

2.7 is preferred.

Cmake: https://cmake.org/

Get the installer version. This one is necessary to make the VS solution files for the `libgpuarray` backend.

Git For Windows: https://git-scm.com/download/win

Add the optional Linux tools to path during installation or manually add `C:\Program Files\Git\usr\bin` to PATH assuming you installed Git there.

MingWpy: Install it using `pip install -i https://pypi.anaconda.org/carlkl/simple mingwpy` from the command line after installing Anaconda. Add `C:\Users\UserName\Anaconda2\share\mingwpy\bin` to the system-wide path using the Rapid Environment editor. Add  `C:\UserName\UserName\Anaconda2\share\mingwpy\include` to `CPATH` assuming you installed Anaconda there. Note that it is `CPATH` and not `PATH`. Create the environment variables if they are missing. And of course, change the `UserName` to your own user name.

Do not install libpython or mingw using pip as the Theano instructions say. MingWpy is a later distribution that also works fine.

1) From the command line, clone Theano wherever you want using `git clone https://github.com/Theano/Theano`. Do not download the zip as that can lead to missing subrepository errors.

2) In the Theano directory that you cloned, install Theano into your Python distribution by typing `python setup.py install` from the command line.

3) Now you should be able to type `import theano` inside Python and have it execute without error.

To add GPU support, create THEANO_FLAGS an environment variable with the following values “device=gpu,floatX=float32,nvcc.fastmath=True,lib.cnmem=0.75,warn_float64=warn,allow_gc=False”. The values have to be colon separated, if you use semi-colons (or put them in their own separate entries) it will not work.

To check what all those options do, [see here](http://deeplearning.net/software/theano/library/config.html).

4) For much better performance on convolutional nets, sign up for the Nvidia developer program and get [cuDNN](https://developer.nvidia.com/cudnn). After installing it it make sure that `cudnn64_5.dll` if using v5 or a different dll if using another version is in `PATH`. To make Theano use cuDNN, the easiest way is to imitate the Linux installation – copy `cudnn.lib` into the Cuda library directory and `cudnn.h` into the Cuda include directory.

On my system they are “C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\lib\x64” and “C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v7.5\include” respectively.

5) Optionally, you might also want to get OpenBLAS for much better performance on the CPU as well.

I am going to describe the hard way of installing it first for educational purposes.

First we need the `make` tool for the Makefiles. Go into your MingWpy bin directory - “C:\Users\UserName\Anaconda2\share\mingwpy\bin” or the like – and find the `mingw32-make.exe` file. Copy it to the same directory and rename the copy to `make.exe`. Now you should be able to run Linux style makefiles on Windows with MingW.

Get the latest stable OpenBLAS source [from here](https://sourceforge.net/projects/openblas/files). Extract it to something like, “OpenBLAS-2.18-source”. From the command line, just run `make` and it should build automatically. It will take 10m or so to finish.

After it is done, copy the `libopenblas.dll` and `libopenblas.dll.a` to `C:\OpenBlas` and add that directory to path. You are not done yet.

[Download](https://sourceforge.net/projects/openblas/files/v0.2.15/) `mingw64_dll.zip` and extract `libgcc_s_seh-1.dll`, `libgfortran-3.dll` and `libquadmath-0.dll` to the same directory where you put `libopenblas.dll` and `libopenblas.dll.a`.

To finish the installation, add the following option to `THEANO_FLAGS` - `blas.ldflags=-L"C:\OpenBlas" -lopenblas`. It should look like this now - `device=gpu,floatX=float32,blas.ldflags=-L"C:\OpenBlas" -lopenblas,nvcc.fastmath=True,lib.cnmem=0.75,warn_float64=warn,allow_gc=False`.

The easier way of installing OpenBLAS by avoiding the build step is to just download the Windows binary in the `0.2.15` folder where you also got the `mingw64_dll.zip` file. But the build step is easy enough if you have a working MingW distro and Linux command line tools from Git. It is essential that you be able to do it. Also by building the library yourself, you get optimizations specific to your CPU.

6) The final part of the Theano installation is to install the `libgpuarray` backend. At the time writing, the new backend has a heap corruption error on Windows, but I will describe how to install it here for when it gets fixed.

First, clone it using `git clone https://github.com/Theano/libgpuarray.git`.

The fire up the Cmake GUI and select the source directory like so perhaps - `C:/libgpuarray`. Copy paste the source directory to build directory and add `/Build` to it, like so for example `C:/libgpuarray/Build`.

Specify the VS 2015 Win64 (if you have it) or VS 2013 Win64 generator and use the default native compiler.

Ignore the error messages, they are not like compiler errors, and just press `Generate`. If everything went well, there should be a `libgpuarray.sln` file in the Build directory. Run it and in VS build the solution in Release mode. In the `C:\libgpuarray\lib\Release` there should now be `gpuarray.dll` a few static library files. Add this directory to path.

Lastly, add `cuda0` to the `DEVICE` environment variable and you should be able to do `import pygpu` and then `pygpu.test()` to run the tests in Cuda mode. `GPUARRAY_DEVICE` variable does nothing unlike what the documentation states.

It is also possible to generate a MingW makefile using Cmake and then build the library using `make` from the command line similar to for OpenBlas. A word of warning is that the library will look for `gpuarray.dll` while the MingW will compile it as `libgpuarray.dll` instead. The `lib` prefixes will have to be removed manually from the `.dll` and `.a` files.

To activate the new `libgpuarray` backend in Theano, change `device=gpu` to `device=cuda`. This is also undocumented at the time of writing. It might be better to wait until the heap corruption bug gets resolved.

7) I am writing the above from memory, so if you find an omission tell me and I will update it.

Extras:

- Note on the `.bashrc` file. When I first tried Theano a year ago, I was positively mystified about what that thing was doing as it looked like some kind of Linux configuration file that I knew Windows would not touch. Now with much more experience, I know that it is in fact a Linux configuration file that Windows will not touch. In Linux such files are used to add variables to the environment, but do nothing on Windows. Do not bother with it. It is highly likely that the Theano documentation is wrong here and Theano itself is not checking the file on Windows.

- It is remarkable how similar Linux and Windows environments for path variables are. They are pretty much identical. And unlike Windows, Linux does not store extra information in the registry as it does not even have it. All it uses are local configuration files. Keeping this bit of knowledge in mind is quite important when it comes to adapting Linux first programs on Windows. Often, one just needs to add the right flag variables to the environment and the location of the binaries to `PATH`.

- If you have multiple MingW or Python distributions, the first one in path will get used. To find out if you have multiple Pythons for example, type `where python` in the command line. Then if changing the select priority is needed, just move the entry upwards in the environment using the Rapid Environment Editor.
