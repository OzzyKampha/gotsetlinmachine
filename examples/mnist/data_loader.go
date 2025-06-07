package main

import (
	"compress/gzip"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"

	"github.com/petar/GoMNIST"
)

var mnistMirrorBase = "https://ossci-datasets.s3.amazonaws.com/mnist/"

var mnistFiles = []string{
	"train-images-idx3-ubyte.gz",
	"train-labels-idx1-ubyte.gz",
	"t10k-images-idx3-ubyte.gz",
	"t10k-labels-idx1-ubyte.gz",
}

func downloadFile(url, dest string) error {
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return fmt.Errorf("bad HTTP response: %s", resp.Status)
	}

	out, err := os.Create(dest)
	if err != nil {
		return err
	}
	defer out.Close()

	_, err = io.Copy(out, resp.Body)
	return err
}

func extractGzipFile(gzPath string) error {
	outPath := strings.TrimSuffix(gzPath, ".gz")

	f, err := os.Open(gzPath)
	if err != nil {
		return err
	}
	defer f.Close()

	gzr, err := gzip.NewReader(f)
	if err != nil {
		return fmt.Errorf("gzip open failed for %s: %w", gzPath, err)
	}
	defer gzr.Close()

	outFile, err := os.Create(outPath)
	if err != nil {
		return err
	}
	defer outFile.Close()

	_, err = io.Copy(outFile, gzr)
	return err
}

func EnsureMNISTReady(destDir string) error {
	os.MkdirAll(destDir, 0755)
	for _, file := range mnistFiles {
		gzPath := filepath.Join(destDir, file)
		url := mnistMirrorBase + file

		fmt.Println("Downloading:", file)
		if err := downloadFile(url, gzPath); err != nil {
			return err
		}
		if err := extractGzipFile(gzPath); err != nil {
			return err
		}
	}
	return nil
}

func binarize(flatData GoMNIST.RawImage, threshold uint8) []int {
	bits := make([]int, 784)
	for i := 0; i < 784; i++ {
		if flatData[i] > threshold {
			bits[i] = 1
		} else {
			bits[i] = 0
		}
	}
	return bits
}

func loadBinaryMNIST(path string, threshold uint8, sampleLimit int) ([][]int, []int) {
	dir := "data"

	err := EnsureMNISTReady(dir)
	if err != nil {
		log.Fatal("Download or extraction failed:", err)
	}

	// Print all file names to verify what's present
	files, err := os.ReadDir(dir)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Files in", dir, ":")
	for _, f := range files {
		fmt.Println(" -", f.Name())
	}

	fmt.Println("Attempting to load MNIST from uncompressed files...")
	trainSet, _, err := GoMNIST.Load(dir)
	if err != nil {
		log.Fatal("GoMNIST.Load failed:", err)
	}

	fmt.Println("Loaded:", len(trainSet.Images), "training samples")
	X := [][]int{}
	y := []int{}
	n := sampleLimit
	if n > len(trainSet.Images) {
		n = len(trainSet.Images)
	}

	for i := 0; i < n; i++ {
		img := trainSet.Images[i] // GoMNIST.RawImage = [784]uint8
		label := trainSet.Labels[i]
		X = append(X, binarize(img, threshold))
		y = append(y, int(label))
	}

	return X, y
}
