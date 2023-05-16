# <# c practice #>
# cd msvc-hello
# ./compile.ps1
# cd ..

# gendef ./hello.dll
# dlltool -dllname ./hello.dll --def ./hello.def --output-lib ./libhello.a

# gcc -I./msvc-hello -L. hello-c_main/main.c -o main.exe -lhello


<# cgo practice #>
cd cu-hello
./compile.ps1
cd ..

gendef ./hello.dll
dlltool -dllname ./hello.dll --def ./hello.def --output-lib ./libhello.a

go build
