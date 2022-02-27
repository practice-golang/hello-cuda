package main // import "hello-cuda"

// #cgo CFLAGS: -I./include
// #cgo LDFLAGS: -L. -lcalc
// #include <calc.cuh>
import "C"
import "fmt"

func main() {
	fmt.Println("^_^")

	r, err := C.sayHello()
	if err != nil {
		panic(err)
	}
	fmt.Println(int(r))
}
