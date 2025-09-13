use crate::regressor::GradientBoostingRegressor;

pub mod regressor;
pub mod stump;

fn main() {
    // Sample data: features (X) and target (y)
    let x_train: Vec<Vec<f64>> = vec![
        vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0],
        vec![6.0], vec![7.0], vec![8.0], vec![9.0], vec![10.0],
    ];
    // A non-linear relationship: y = x*sin(x)
    let y_train: Vec<f64> = x_train
        .iter()
        .map(|row| row[0] * row[0].sin())
        .collect();

    // Create and train the model
    let n_estimators = 100;
    let learning_rate = 0.1;
    let mut model = GradientBoostingRegressor::new(n_estimators, learning_rate);
    
    model.fit(&x_train, &y_train);

    // Make predictions on new data
    let x_test: Vec<Vec<f64>> = vec![vec![2.5], vec![5.5], vec![8.5]];
    let predictions = model.predict(&x_test);

    println!("Model Details: {:#?}", model);
    println!("\nTest Data Predictions:");
    for (i, p) in predictions.iter().enumerate() {
        let actual = x_test[i][0] * x_test[i][0].sin();
        println!("- Input: {:.1}, Prediction: {:.4}, Actual: {:.4}", x_test[i][0], p, actual);
    }
}