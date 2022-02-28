# Taste CUDA

## Used tools
* [CUDA Toolkit 11.6](https://developer.nvidia.com/cuda-11-6-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local)
* [VS2017 Express](https://visualstudio.microsoft.com/ko/vs/express)
* [MinGW-W64 10.3.0](https://github.com/brechtsanders/winlibs_mingw)

## Compile
```dos
msvc_env.cmd
setpath.cmd
build.cmd
```

## Trouble shooting

### `nvcc fatal: Microsoft Visual Studio configuration file 'vcvars64.bat' could not be found for installation..`
* Copy file
    * vcvarsx86_amd64.bat -> vcvars64.bat
    * Path: C:\Program Files (x86)\Microsoft Visual Studio\2017\WDExpress\VC\Auxiliary\Build
