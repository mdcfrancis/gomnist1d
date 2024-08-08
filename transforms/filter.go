package transforms

import (
	"fmt"
	np "github.com/mdcfrancis/gonp"
)

// CreateGaussianKernel creates a 1D Gaussian kernel with a given standard deviation
// Truncation is set to 4.0 by default
func CreateGaussianKernel(sigma float64) np.NpArray {
	truncation := 4.0
	// scipy performs round as int(x + 0.5)
	radius := int(truncation*sigma + 0.5)
	return _createGaussianKernel(sigma, radius)
}

func _createGaussianKernel(sigma float64, radius int) np.NpArray {
	sigma2 := sigma * sigma
	x := np.Arrange(-float64(radius), float64(radius)+1, 1.0)
	x2 := x.PowFloat64(2.0)
	exponent := x2.MulFloat64(-0.5 / sigma2)
	phi_x := np.Exp(exponent)
	phi_x = phi_x.DivFloat64(phi_x.Sum())
	return phi_x
}

func reverse[T any](in []T) []T {
	out := make([]T, len(in))
	for i, v := range in {
		out[len(in)-1-i] = v
	}
	return out
}

// mirror creates a mirrored version of the input array
// abcd becomes dcbaabcddcba
func mirror[T any](in []T) []T {
	out := append(reverse(in), in...)
	out = append(out, reverse(in)...)
	return out
}

// Convolve1D performs a 1D convolution with the given kernel on the input array
// kernel may also be though of as weights to apply to the input array
func Convolve1D(input []float64, kernel np.NpArray) []float64 {
	// create a mirrored input vector to handle boundary conditions
	// this is a common technique in signal processing
	// abcd becomes dcbaabcddbca
	mirrored := mirror(input)
	output := make([]float64, len(input))
	kernelRadius := len(kernel) / 2
	// We do not support edge cases outside of simple reflection edge case
	if kernelRadius > len(input) {
		err := fmt.Errorf("kernel radius must be less or equal to the input size %d %d", kernelRadius, len(input))
		panic(err)
	}
	// for len 4 input and kernel 3
	// the slice starts at len(input) - kernelRadius
	for i, _ := range input {
		start := i + len(input) - kernelRadius
		end := i + len(input) + kernelRadius + 1
		ipSlice := mirrored[start:end]
		output[i] = kernel.Dot(ipSlice)
	}
	return output
}

// GaussianFilter1D applies a 1D Gaussian filter to the input array with the given sigma
func GaussianFilter1D(input np.NpArray, sigma float64) np.NpArray {
	kernel := CreateGaussianKernel(sigma)
	return Convolve1D(input, kernel)
}
