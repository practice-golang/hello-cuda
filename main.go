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
		go func() {
			container := C.new_container()

			r, err := C.say_hello(container)
			if err != nil {
				panic(err)
			}
			fmt.Println("sayHello return: ", int(r))

			C.free_mem(container)

			time.Sleep(2 * time.Second)
		}()
		time.Sleep(7 * time.Second)
	}
}
