#include <iostream>
#include <cstring>

#include "hello.h"
#include "wrapper.h"

void* new_container() {
    return getContainer();
}

int say_hello(void* container) {
    sayHello(container);

    return 0;
}

int free_mem(void* container) {
    freeMem(container);

    const char* output = "Hello from C++";
    std::cout << output << std::endl;

    return 0;
}
