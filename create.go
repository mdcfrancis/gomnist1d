package mnist1d

import (
	"encoding/json"
	"fmt"
	"github.com/mdcfrancis/gomnist1d/transforms"
	np "github.com/mdcfrancis/gonp"
	"github.com/pa-m/randomkit"
	"strings"
)

type Mnist1D struct {
	NumSamples int     `json:"num_samples"`
	TrainSplit float64 `json:"train_split"`
	Seed       uint64  `json:"seed"`
	Url        string  `json:"url"`
	Templates  Templates
	transforms.Rules
}

func (m Mnist1D) String() string {
	return fmt.Sprintf("Mnist1D{NumSamples: %v, TrainSplit: %v, TemplateLen: %v, Seed: %v, Url: %v, Templates: %v, Rules: %v}", m.NumSamples, m.TrainSplit, m.TemplateLen, m.Seed, m.Url, m.Templates, m.Rules)
}

const defaultArgs = `
{
	"num_samples": 5000,
	"train_split": 0.8,
	"template_len": 12,
	"padding": [36,60],
	"scale_coeff": 0.4,
	"max_translation": 48,
	"corr_noise_scale": 0.25,
	"iid_noise_scale": 2e-2,
	"shear_scale": 0.75,
	"shuffle_seq": false,
	"final_seq_length": 40,
	"seed": 42,
	"url": "https://github.com/greydanus/mnist1d/raw/master/mnist1d_data.pkl"
}
`

func NewDefault() Mnist1D {
	m := Mnist1D{}
	_ = json.Unmarshal([]byte(defaultArgs), &m)
	t := m.GetTemplates()
	m.Templates = t
	m.Rnd = randomkit.NewRandomkitSource(m.Seed)
	return m
}

type Templates struct {
	X []np.NpArray
	T np.NpArray
	Y np.NpArray
}

func (t Templates) String() string {
	return fmt.Sprintf("Templates{X: %v, T: %v, Y: %v}", t.X, t.T, t.Y)
}

func (m *Mnist1D) GetTemplates() Templates {
	d0 := np.NpArray{5, 6, 6.5, 6.75, 7, 7, 7, 7, 6.75, 6.5, 6, 5}
	d1 := np.NpArray{5, 3, 3, 3.4, 3.8, 4.2, 4.6, 5, 5.4, 5.8, 5, 5}
	d2 := np.NpArray{5, 6, 6.5, 6.5, 6, 5.25, 4.75, 4, 3.5, 3.5, 4, 5}
	d3 := np.NpArray{5, 6, 6.5, 6.5, 6, 5, 5, 6, 6.5, 6.5, 6, 5}
	d4 := np.NpArray{5, 4.4, 3.8, 3.2, 2.6, 2.6, 5, 5, 5, 5, 5, 5}
	d5 := np.NpArray{5, 3, 3, 3, 3, 5, 6, 6.5, 6.5, 6, 4.5, 5}
	d6 := np.NpArray{5, 4, 3.5, 3.25, 3, 3, 3, 3, 3.25, 3.5, 4, 5}
	d7 := np.NpArray{5, 7, 7, 6.6, 6.2, 5.8, 5.4, 5, 4.6, 4.2, 5, 5}
	d8 := np.NpArray{5, 4, 3.5, 3.5, 4, 5, 5, 4, 3.5, 3.5, 4, 5}
	d9 := np.NpArray{5, 4, 3.5, 3.5, 4, 5, 5, 5, 5, 4.7, 4.3, 5}
	x := np.NpStack{d0, d1, d2, d3, d4, d5, d6, d7, d8, d9}

	x = x.Sub(x.Mean())                // whiten
	x = x.Div(x.StandardDeviation())   // norm ( this is a noop )
	x = x.Sub(x.Slice(0, 1).Column(0)) // signal starts and ends at 0
	t := Templates{
		X: x.DivFloat64(6.0),
		T: np.LinSpace(-5.0, 5.0, len(d0)).DivFloat64(6.0),
		Y: np.NpArray{0, 1, 2, 3, 4, 5, 6, 7, 8, 9},
	}
	return t
}

type DataSet struct {
	X         np.NpStack `json:"x"`
	X_TEST    np.NpStack `json:"x_test"`
	Y         np.NpArray `json:"y"`
	Y_TEST    np.NpArray `json:"y_test"`
	T         np.NpArray `json:"t"`
	Templates Templates  `json:"templates"`
}

func (m Mnist1D) MakeDataset() DataSet {
	// reproducibility
	m.Rnd = randomkit.NewRandomkitSource(m.Seed)
	xs := np.NpStack{}
	ys := np.NpArray{}
	samplesPerClass := m.NumSamples / len(m.Templates.Y)
	var newT np.NpArray
	for labelIx := range m.Templates.Y {
		for exampleIx := 0; exampleIx < samplesPerClass; exampleIx++ {
			x := m.Templates.X[labelIx]
			t := m.Templates.T
			y := m.Templates.Y[labelIx]
			x, newT = m.Rules.Transform(x, t)
			xs = append(xs, x)
			ys = append(ys, y)
		}
	}

	batchShuffle := m.Rnd.Perm(len(ys)) // shuffle batch dimension
	xs = xs.Shuffle(batchShuffle)
	ys = ys.Shuffle(batchShuffle)

	if m.Rules.ShuffleSeq { // maybe shuffle the spatial dimension
		seqShuffle := m.Rnd.Perm(m.Rules.FinalSeqLength)
		xs = xs.Shuffle(seqShuffle)
	}

	newT = newT.DivFloat64(newT.StandardDeviation())

	// Note that the normalization operates across the entire set
	// whereas in template case it is row wise.
	xs = xs.SubFloat64(xs.ScalarMean())              // whiten
	xs = xs.DivFloat64(xs.ScalarStandardDeviation()) // norm

	// train / test split
	splitIx := int(float64(len(ys)) * m.TrainSplit)
	return DataSet{
		X:         xs[:splitIx],
		X_TEST:    xs[splitIx:],
		Y:         ys[:splitIx],
		Y_TEST:    ys[splitIx:],
		T:         newT,
		Templates: m.Templates,
	}
}

func ShowAsImage(m np.NpArray) string {
	// expected length of m is 40
	// The values range from -5 to 5, we will render them as 0 to 10
	width := 40
	ret := make([]string, 0, len(m))
	for _, v := range m {
		// create an array of 10 characters
		row := make([]string, 0, width)
		for j := 0; j < width; j++ {
			idx := int((v + 1) / 2 * float64(width))
			if j == idx {
				row = append(row, "X")
			} else {
				row = append(row, ".")
			}
		}
		ret = append(ret, strings.Join(row, ""))
	}

	return "\n" + strings.Join(ret, "\n")
}
