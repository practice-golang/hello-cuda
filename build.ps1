cd cu-hello
./compile.ps1
cd ..

gendef ./hello.dll
dlltool -dllname ./hello.dll --def ./hello.def --output-lib ./libhello.a

go build
