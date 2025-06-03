package main

import (
	"fmt"
	"os"

	"github.com/OzzyKampha/gotsetlinmachine/examples/binary"
	"github.com/OzzyKampha/gotsetlinmachine/examples/mnist"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Please specify which example to run: binary, mnist")
		fmt.Println("Usage: go run main.go [binary|mnist]")
		os.Exit(1)
	}

	switch os.Args[1] {
	case "binary":
		fmt.Println("Running binary classification example...")
		binary.RunBinaryExample()
	case "mnist":
		fmt.Println("Running MNIST classification example...")
		mnist.RunMNISTExample()
	default:
		fmt.Printf("Unknown example: %s\n", os.Args[1])
		fmt.Println("Available examples: binary, mnist")
		os.Exit(1)
	}
}
