# <# c practice #>
# cd msvc-hello
# ./compile.ps1
# cd ..

# gendef ./hello.dll
# dlltool -k -d ./hello.def -l ./libhello.a

# gcc -I./msvc-hello hello-c_main/main.c -o main.exe -static -L. -lhello


<# cgo practice #>
cd cu-hello
./compile.ps1
cd ..

gendef ./hello.dll
dlltool -k -d ./hello.def -l ./libhello.a

cd cu-wrapper
./compile.ps1
cd ..

go build -ldflags "-w -s"
