use ndarray::{Array, Array1, Array2, Axis, Zip};
use ndarray_stats::QuantileExt;
use polars_core::export::num::Float;
use rand::Rng;


pub fn init_variables() -> (Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>) {

    let depth: usize = 20;

    let w1: Array2<f64> = rand_array2(depth, 784);
    let b1: Array1<f64> = rand_array1(depth);
    let w2: Array2<f64> = rand_array2(10, depth);
    let b2: Array1<f64> = rand_array1(10);
    (w1,b1,w2,b2)
}

pub fn forward_prop(w1: &Array2<f64>, b1: &Array1<f64>, w2: &Array2<f64>, b2: &Array1<f64>, inp: &Array2<f64>) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
    let b1_hat = b1.clone();
    let b2_hat = b2.clone();
    let z1 = w1.dot(inp)  + b1_hat.insert_axis(ndarray::Axis(1));
    let a1 = relu(&z1);
    let z2 = w2.dot(&a1)  + b2_hat.insert_axis(ndarray::Axis(1));
    let a2 =softmax(&(z2));
    
    (z1,a1,a2,z2)
}

pub fn rand_array2(rows: usize, cols: usize) -> Array2<f64> {
    let mut rng = rand::thread_rng();
    let mut result = Array2::zeros((rows, cols));

    for i in 0..rows {
        for j in 0..cols {
            result[[i, j]] = rng.gen_range(-0.5..0.5);
        }
    }
    // println!("result {:?}",result);
    result
}

pub fn rand_array1(size: usize) -> Array1<f64> {
    let mut rng = rand::thread_rng();
    let mut result = Array1::zeros(size);

    for i in 0..size {
        
            result[i] = rng.gen_range(-0.5..0.5);
    }

    result
}

pub fn relu(input: &Array2<f64>) -> Array2<f64> {
    let res = input.map(|&x| f64::max(0.0, x));
    
    assert!(!res.iter().any(|&x| x.is_nan()));
    
    res

}

pub fn deriv_relu(input: &Array2<f64>) -> Array2<f64> {
    let res = input.map(|&x| if x > 0.0 { 1.0 } else { 0.0 });
    
    assert!(!res.iter().any(|&x| x.is_nan()));

    res

}

pub fn softmax(x: &Array2<f64>) -> Array2<f64> {

    let expx = x.mapv(|e| e.exp() );
        &expx / expx.sum_axis(Axis(0))

}

pub fn one_hot(input: &Array1<f64>) -> Array2<f64> {

    let num_classes = 10;

    let mut result = Array::zeros((input.len(), num_classes));

    for (i, &class_index) in input.iter().enumerate() {
        result[[i, class_index as usize]] = 1.0;
    }

    result.reversed_axes()
}

pub fn back_prop(z1: &Array2<f64>, a1: &Array2<f64>, a2: &Array2<f64>, w2: &Array2<f64>, x: &Array2<f64>, y: &Array1<f64>) -> (Array2<f64>, f64, Array2<f64>, f64) {
    let m: f64 = a2.shape()[1] as f64;
    let one_hot_y = one_hot(&y);
    let dz2 = a2 - &one_hot_y;
    let dw2 = (1.0 / m) * dz2.dot(&a1.t());
    let db2 = (1.0 / m) * dz2.sum();
    let dz1 = w2.t().dot(&dz2) * deriv_relu(&z1);
    let dw1 = (1.0 / m) * dz1.dot(&x.t());
    let db1 = (1.0 / m) * dz1.sum();

    (dw1,db1,dw2,db2)
}

pub fn update_params(w1: &mut Array2<f64>,b1: &mut Array1<f64>,w2: &mut Array2<f64>,b2: &mut Array1<f64>,dw1: &Array2<f64>,db1: f64,dw2: &Array2<f64>,db2: f64,alpha: f64) {
    Zip::from(w1)
        .and(dw1)
        .for_each(|w1_elem, dw1_elem| *w1_elem -= alpha * dw1_elem);

    *b1 -= alpha * db1;

    Zip::from(w2)
        .and(dw2)
        .for_each(|w2_elem, dw2_elem| *w2_elem -= alpha * dw2_elem);

    *b2 -= alpha * db2;

}

pub fn get_predictions(a2: &Array2<f64>) -> Array1<f64> {
    
    let result:Vec<f64> = a2.axis_iter(ndarray::Axis(1))
    .map(|row| row.argmax().expect("argmax error") as f64)
    .collect();

    let result_array_1d = Array1::from(result);
    
    result_array_1d
}

pub fn get_accuracy(predictions: &Array1<f64>, y: &Array1<f64>) -> f64 {
    // println!("{:?} {:?}", predictions, y);

    let num_correct = predictions
        .iter()
        .zip(y.iter())
        .filter(|(pred, actual)| pred == actual)
        .count() as f64;

    num_correct / y.len() as f64
}

pub fn gradient_descent(x :  Array2<f64>, y:  Array1<f64>, iterations: u32, alpha: f64) -> (Array2<f64>, Array1<f64>, Array2<f64>, Array1<f64>) {
    let (mut w1,mut b1,mut w2,mut b2) = init_variables();
     for i in 0..iterations {
        let (z1,a1,a2,_z2) = forward_prop(&w1, &b1, &w2, &b2, &x);
        let (dw1,db1,dw2,db2) = back_prop(&z1,&a1,&a2,&w2,&x,&y);        
        update_params(&mut w1, &mut b1, &mut w2, &mut b2, &dw1, db1, &dw2, db2, alpha);

        if i % 50 == 0 {
            println!("Iteration: {}", i);
            // Assuming A2 and Y are already defined
            let predictions = get_predictions(&a2);
            // println!("{:?}",y);
            let accuracy: f64 = get_accuracy(&predictions, &y);

            println!("accuracy: {}", accuracy);
        }
    }
    (w1, b1, w2, b2)
}

pub fn make_predictions (w1: &Array2<f64>, b1: &Array1<f64>, w2: &Array2<f64>, b2: &Array1<f64>, x: &Array2<f64>) -> Array1<f64> {
    let (_,_,_,a2) = forward_prop(&w1, &b1, &w2, &b2, &x);
    let predictions = get_predictions(&a2);
    predictions
}

