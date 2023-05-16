#include <iostream>
#include <cstring>

#include "hello.h"
#include "wrapper.h"

int say_hello() {
    sayHello();

    return 0;
}

int free_mem() {
    freeMem();

    const char* output = "Hello from C++";
    std::cout << output << std::endl;

    return 0;
}
