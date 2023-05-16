g++ -I../cu-hello -c -o ../wrapper.o wrapper.cpp
ar rcs ../libwrapper.a ../wrapper.o
