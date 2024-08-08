# Replication of the MNIST1D dataset in 100% GO. 

A 100% golang implementation of the MNIST1D dataset.
https://github.com/greydanus/mnist1d

the test found in data/loader_test.go : TestLoadMNISTImageFileMatches loads the orginal pickle file (converted to JSON) 
and compares with the go generated version. These match including supporting the random seeds. 

## Usage
```go
package main

include (
    "fmt"
    "github.com/mdcfrancis/mnist1d"
)

func main() {
    m := mnist1d.NewDefault()

    ds := m.MakeDataset()
    for i, x := range m.Templates.X {
        fmt.Println("Template", i)
        fmt.Println(mnist1d.ShowAsImage(x))
    }
	
    for i := 0; i < 2; i++ {
        fmt.Println(ds.y[i])
        fmt.Println(mnist1d.ShowAsImage(ds.x[i]))
        fmt.Println(ds.x[i])
    }
}
```

The code attempts to follow closely the code in the original repository so at times looks less 'go like'.
* Imports the package https://github.com/mdcfrancis/gonp to support np like semantics.
* Implements a scipy like Gaussian 1D filter, supporting edge reflection. 

## References
* https://github.com/greydanus/mnist1d : the python reference implementation.

## Data converted from the original repository pickle file 
* https://github.com/greydanus/mnist1d : data/mnist1d.json
