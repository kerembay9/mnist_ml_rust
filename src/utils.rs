use ndarray::Array2;
use rand::Rng;

pub fn init_variables() -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
    // x: input with size 784 x m
    // h1: hidden layer 1
    // h1 = w1 * x + b

    let w1: Array2<f64> = rand_array2(10, 784);
    let b1: Array2<f64> = rand_array2(10, 1);
    let w2: Array2<f64> = rand_array2(10, 10);
    let b2: Array2<f64> = rand_array2(10, 1);
    (w1,b1,w2,b2)
}

pub fn forward_prop(w1: &Array2<f64>, b1: &Array2<f64>, w2: &Array2<f64>, b2: &Array2<f64>, inp: &Array2<f64>) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
    let z1 = w1.dot(inp) + b1;
    let a1 = relu(z1.clone());
    let z2 = w2.dot(&a1) + b2;
    let a2 =softmax(z2.clone());

    (z1,a1,z2,a2)
}

pub fn rand_array2(rows: usize, cols: usize) -> Array2<f64> {
    let mut rng = rand::thread_rng();
    let mut result = Array2::zeros((rows, cols));

    for i in 0..rows {
        for j in 0..cols {
            result[[i, j]] = rng.gen_range(-0.5..0.5);
        }
    }

    result
}

pub fn relu(input: Array2<f64>) -> Array2<f64> {
    input.map(|&x| f64::max(0.0, x))
}

pub fn softmax(input: Array2<f64>) -> Array2<f64> {
    let exp_sum: f64 = input.map(|&x| f64::exp(x)).sum();
    input.map(|&x| f64::exp(x) / exp_sum)
}