package main // import "hello-cuda"

// #cgo CFLAGS: -I./cu-hello
// #cgo LDFLAGS: -static -L. -lstdc++ -lhello
// #include "hello.h"
import "C"
import (
	"fmt"
	"time"
)

func main() {
	fmt.Println("Begin:")

	for {
		r, err := C.sayHello()
		if err != nil {
			panic(err)
		}
		fmt.Println("sayHello return: ", int(r))

		C.freeMem()

		time.Sleep(2 * time.Second)
	}
}
