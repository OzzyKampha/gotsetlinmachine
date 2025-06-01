package main

import (
	"fmt"
	"os"

	"github.com/OzzyKampha/gotsetlinmachine/examples/binary"
	"github.com/OzzyKampha/gotsetlinmachine/examples/multiclass"
)

func main() {
	if len(os.Args) < 2 {
		fmt.Println("Please specify which example to run: binary or multiclass")
		fmt.Println("Usage: go run main.go [binary|multiclass]")
		os.Exit(1)
	}

	switch os.Args[1] {
	case "binary":
		fmt.Println("Running binary classification example...")
		binary.RunBinaryExample()
	case "multiclass":
		fmt.Println("Running multiclass classification example...")
		multiclass.RunMulticlassExample()
	default:
		fmt.Printf("Unknown example: %s\n", os.Args[1])
		fmt.Println("Available examples: binary, multiclass")
		os.Exit(1)
	}
}
