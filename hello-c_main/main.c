#ifdef _WIN32
#include <windows.h>
#define sleep(x) Sleep(x * 1000)
#else
#include <unistd.h>
#endif

#include "hello.h"

int main(void) {
    while (1) {
        void* container = getContainer();

        sayHello(container);
        freeMem(container);

        sleep(2);
    }

    return 0;
}
