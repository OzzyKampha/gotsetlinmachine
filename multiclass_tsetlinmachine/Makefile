.PHONY: all test clean lint build run-example

all: test lint build

test:
	go test -v -race -coverprofile=coverage.txt -covermode=atomic ./...

lint:
	golangci-lint run

build:
	go build -o bin/example ./cmd/example

run-example:
	go run cmd/example/main.go

clean:
	rm -rf bin/
	rm -f coverage.txt

deps:
	go mod download
	go mod tidy

bench:
	go test -bench=. -benchmem ./...

vet:
	go vet ./...

fmt:
	go fmt ./... 