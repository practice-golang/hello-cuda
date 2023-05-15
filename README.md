# Taste CUDA

<details>
<summary>MSVC 2017</summary>

## Used tools
* [CUDA Toolkit 11.6](https://developer.nvidia.com/cuda-11-6-0-download-archive?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local)
* [VS2017 Express](https://visualstudio.microsoft.com/ko/vs/express)
* [MinGW-W64 10.3.0](https://github.com/brechtsanders/winlibs_mingw)

## Compile
```powershell
msvc_env_2017.cmd
setpath_cuda11.cmd
build.ps1
```

## Trouble shooting

### `nvcc fatal: Microsoft Visual Studio configuration file 'vcvars64.bat' could not be found for installation..`
* Copy file
    * vcvarsx86_amd64.bat -> vcvars64.bat
    * Path: C:\Program Files (x86)\Microsoft Visual Studio\2017\WDExpress\VC\Auxiliary\Build
</details>

## Used tools
* [CUDA Toolkit 12.1](https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64)
* [VS 2022 Community](https://visualstudio.microsoft.com/vs)
* [MinGW 12.2.0](https://github.com/brechtsanders/winlibs_mingw/releases/tag/12.2.0-16.0.0-10.0.0-ucrt-r5)

## Compile
```powershell
msvc_env_2022.ps1
setpath_cuda12.ps1
build.ps1
```
