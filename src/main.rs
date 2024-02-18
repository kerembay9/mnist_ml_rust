mod utils;

use ndarray_stats::QuantileExt;
use utils::*;
use polars_core::prelude::*;
use polars::prelude::*;
use ndarray::{Array1,Array2,s};

#[cfg(test)]
mod tests;
fn main() {

     let df: DataFrame = read_mnist_test().expect("Error reading DataFrame");
     let ndarray = df.to_ndarray::<Float64Type>(IndexOrder::Fortran).unwrap();
     let x: Array2<f64> = ndarray.clone().slice(s![.., 1..]).t().into_owned();
     let x = &x / *x.max().expect("max error");
     let y: Array1<f64> = ndarray.clone().slice(s![.., 0]).t().into_owned();
     let (w1, b1, w2, b2) = gradient_descent(x,y,500,0.2);
     println!("w1: {:?}", w1);
     println!("b1: {:?}", b1);
     println!("w2: {:?}", w2);
     println!("b2: {:?}", b2);
}


// fn read_mnist_train() -> PolarsResult<DataFrame> {
//      CsvReader::from_path("/Users/kerembayramoglu/Desktop/rust/mnist_ml_rust/dataset/mnist_train.csv")?
//              .has_header(true)
//              .finish()
//  }

 fn read_mnist_test() -> PolarsResult<DataFrame> {
     CsvReader::from_path("/Users/kerembayramoglu/Desktop/rust/mnist_ml_rust/dataset/mnist_test.csv")?
             .has_header(true)
             .finish()
     
 }