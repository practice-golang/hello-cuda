@REM cl as cgo not working
@REM go env -w CC=cl.exe

cd cu-calc
call compile.cmd
cd ..

gendef calc.dll
dlltool  -dllname calc.dll --def calc.def --output-lib libcalc.a

go build
