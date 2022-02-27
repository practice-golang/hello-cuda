# Taste CUDA

## Used tools
* CUDA Toolkit v11.6
* VS2017 Express

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
