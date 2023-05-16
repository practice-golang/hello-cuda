package main // import "hello-cuda"

// #cgo CFLAGS: -Icu-wrapper -Icu-hello
// #cgo LDFLAGS: -static -L. -lwrapper -lhello -lstdc++
// #include "wrapper.h"
import "C"
import (
	"fmt"
	"time"
)

func main() {
	fmt.Println("Begin:")

	for {
		r, err := C.say_hello()
		if err != nil {
			panic(err)
		}
		fmt.Println("sayHello return: ", int(r))

		C.free_mem()

		time.Sleep(2 * time.Second)
	}
}
