.PHONY: all test clean build examples

all: test build

test:
	go test -v ./...

test-coverage:
	go test -v -coverprofile=coverage.out ./...
	go tool cover -html=coverage.out

build:
	go build -v ./...

clean:
	go clean
	rm -f coverage.out

examples:
	go run examples/main.go binary
	go run examples/main.go multiclass

benchmark:
	go test -bench=. -benchmem ./...

lint:
	golangci-lint run

deps:
	go mod tidy
	go mod verify

.DEFAULT_GOAL := all 