use ndarray::array;
use crate::utils::{softmax, one_hot, relu, deriv_relu, get_predictions, get_accuracy, forward_prop, update_params};

#[cfg(test)]
mod tests {
    use ndarray::{arr1, arr2};
    // Import necessary items from the outer module
    use super::*;
    #[test]
    fn test_softmax() {
        // Arrange
        let input_array = array![[1.0, 2.0, 3.0], [2.0, 2.0, 2.0]];

        // Act
        let result = softmax(&input_array).t().into_owned();

        // Assert
        // Check that the sum of each row in the result is close to 1.0
        for row in result.outer_iter() {
            assert!((row.sum() - 1.0).abs() < 1e-6);
        }
    }
    #[test]
    fn test_one_hot() {
        // Arrange
        let input_array = array![0.0, 1.0,  0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]; // Example input

        // Act
        let result = one_hot(&input_array);

        // Assert
        // Check that each row has only one element set to 1.0, and the rest are 0.0
        for row in result.t().outer_iter() {
            let num_ones: usize = row.iter().filter(|&&x| x == 1.0).count();
            assert_eq!(num_ones, 1);
        }
    }
    #[test]
    fn test_relu() {
        // Arrange
        let input_array = array![[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]];

        // Act
        let result = relu(&input_array);

        // Assert
        // Check that each element in the result is greater than or equal to 0.0
        for &x in result.iter() {
            assert!(x >= 0.0);
        }
    }

    #[test]
    fn test_deriv_relu() {
        // Arrange
        let input_array = array![[1.0, -2.0, 3.0], [-4.0, 5.0, -6.0]];

        // Act
        let result = deriv_relu(&input_array);

        // Assert
        // Check that each element in the result is either 0.0 or 1.0
        for &x in result.iter() {
            assert!(x == 0.0 || x == 1.0);
        }
    }
    #[test]
    fn test_get_predictions() {
        // Arrange
        let input_array = array![
            [0.2, 0.3, 0.5],
            [0.7, 0.1, 0.2],
            [0.1, 0.2, 0.7],
        ].t().to_owned();

        // Act
        let result = get_predictions(&input_array);

        // Assert
        // Check that the result matches the expected predictions
        assert_eq!(result, array![2.0, 0.0, 2.0]);
    }
    #[test]
    fn test_get_accuracy() {
        // Arrange
        let predictions = array![0.0, 1.0, 2.0, 1.0, 0.0];
        let y = array![0.0, 1.0, 2.0, 2.0, 0.0];

        // Act
        let accuracy = get_accuracy(&predictions, &y);

        // Assert
        // Check that the accuracy matches the expected value
        assert_eq!(accuracy, 4.0 / 5.0); // Three correct predictions out of five
    }
    #[test]
    fn test_forward_prop() {
        // Arrange
        let w1 = arr2(&[[-10.0], [2.0]]);
        let b1 = arr1(&[5.0 ,6.0]);
        let w2 = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let b2 = arr1(&[5.0 ,6.0]);
        let inp = arr2(&[[3.0, 4.0]]);

        // Act
        let (z1, a1, a2, z2) = forward_prop(&w1, &b1, &w2, &b2, &inp);

        // Assert
        // You can add more specific assertions based on your expectations
        assert_eq!(z1, arr2(&[[-25.0, -35.0], [12.0, 14.0]]));
        assert_eq!(a1, arr2(&[[0.0, 0.0], [12.0, 14.0]]));
        assert_eq!(z2, arr2(&[[29.0, 33.0], [54.0, 62.0]]));
        assert_eq!(a2, softmax(&z2));
    }
    #[test]
    fn test_update_params() {
        let mut w1 = arr2(&[[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]]);
        let mut b1 = arr1(&[1.0, 2.0, 3.0]);
        let mut w2 = arr2(&[[0.5, 1.0, 1.5]]);
        let mut b2 = arr1(&[0.5]);
        let dw1 = arr2(&[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]);
        let db1 = 0.6;
        let dw2 = arr2(&[[0.1, 0.2, 0.3]]);
        let db2 = 0.1;

        let alpha = 1.0;

        update_params(&mut w1, &mut b1, &mut w2, &mut b2, &dw1, db1, &dw2, db2, alpha);

        let expected_w1 = arr2(&[[0.9, 1.8, 2.7],[1.6, 3.5, 5.4],[2.3, 5.2, 8.1]]);
        let expected_b1 = arr1(&[0.4, 1.4, 2.4]);
        let expected_w2 = arr2(&[[0.4, 0.8, 1.2]]);
        let expected_b2 = arr1(&[0.4]);

        assert_eq!(w1, expected_w1);
        assert_eq!(b1, expected_b1);
        assert_eq!(w2, expected_w2);
        assert_eq!(b2, expected_b2);
    }
    
}
