mod utils;

use ndarray_stats::QuantileExt;
use utils::*;
use polars_core::prelude::*;
use polars::prelude::*;
use ndarray::{Array1,Array2,s};

#[cfg(test)]
mod tests;
fn main() {
     use std::time::Instant;
     let now = Instant::now();
     // Load the data and preprocess
     
     //Train set
     let df: DataFrame = read_mnist_train().expect("Error reading DataFrame");
     let ndarray = df.to_ndarray::<Float64Type>(IndexOrder::Fortran).unwrap();
     let x_train: Array2<f64> = ndarray.clone().slice(s![.., 1..]).t().into_owned();
     let x_train = &x_train / *x_train.max().expect("max error");
     let y: Array1<f64> = ndarray.clone().slice(s![.., 0]).t().into_owned();

     // test set
     let df: DataFrame = read_mnist_test().expect("Error reading DataFrame");
     let ndarray = df.to_ndarray::<Float64Type>(IndexOrder::Fortran).unwrap();
     let x_test: Array2<f64> = ndarray.clone().slice(s![.., 1..]).t().into_owned();
     let x_test = &x_test / *x_test.max().expect("max error");
     let y_test: Array1<f64> = ndarray.clone().slice(s![.., 0]).t().into_owned();

     // train the model
     let (w1, b1, w2, b2) = gradient_descent(x_train,y,500,0.1);
     
     let predictions = make_predictions(&w1, &b1, &w2, &b2, &x_test);

     let accuracy =  get_accuracy(&predictions, &y_test);

     println!("Test accuracy is {:.2}%", accuracy*100.0);

     let elapsed = now.elapsed();
     println!("Elapsed: {:.2?}", elapsed);
}


fn read_mnist_train() -> PolarsResult<DataFrame> {
     CsvReader::from_path("/Users/kerembayramoglu/Desktop/rust/mnist_ml_rust/dataset/mnist_train.csv")?
             .has_header(true)
             .finish()
 }

 fn read_mnist_test() -> PolarsResult<DataFrame> {
     CsvReader::from_path("/Users/kerembayramoglu/Desktop/rust/mnist_ml_rust/dataset/mnist_test.csv")?
             .has_header(true)
             .finish()
     
 }