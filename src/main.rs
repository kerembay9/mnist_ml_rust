mod utils;

use utils::*;
use ndarray::Array2;

fn main() {
     // Calling the function and using pattern matching to extract values
     let (w1, b1, w2, b2) = init_variables();
     let inp = Array2::<f64>::zeros((784, 1));
     let (z1,a1,z2,a2) = forward_prop(&w1, &b1, &w2, &b2, &inp);

     println!("z1: {:?}", z1);
     println!("a1: {:?}", a1);
     println!("z2: {:?}", z2);
     println!("a2: {:?}", a2);
}


