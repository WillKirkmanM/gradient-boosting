use crate::regressor::DecisionStump;

impl DecisionStump {
    /// Predicts the output for a single data point.
    pub fn predict(&self, features: &[f64]) -> f64 {
        if features[self.feature_index] <= self.threshold {
            self.left_value
        } else {
            self.right_value
        }
    }

    /// Finds the best split for the data to predict the target values (residuals).
    pub fn fit(x: &[Vec<f64>], y: &[f64]) -> Self {
        let n_features = x[0].len();
        let mut best_stump = DecisionStump {
            feature_index: 0,
            threshold: 0.0,
            left_value: 0.0,
            right_value: 0.0,
        };
        let mut min_error = f64::MAX;

        // Iterate over each feature to find the best split.
        for feature_index in 0..n_features {
            let unique_values: Vec<f64> = x.iter().map(|row| row[feature_index]).collect();
            
            // Iterate over each unique value as a potential threshold.
            for threshold in unique_values {
                let mut left_indices = vec![];
                let mut right_indices = vec![];

                for (i, row) in x.iter().enumerate() {
                    if row[feature_index] <= threshold {
                        left_indices.push(i);
                    } else {
                        right_indices.push(i);
                    }
                }

                if left_indices.is_empty() || right_indices.is_empty() {
                    continue; // Skip splits that don't separate the data.
                }

                // Calculate the average of the residuals for each leaf.
                let left_sum: f64 = left_indices.iter().map(|&i| y[i]).sum();
                let left_avg = left_sum / left_indices.len() as f64;
                
                let right_sum: f64 = right_indices.iter().map(|&i| y[i]).sum();
                let right_avg = right_sum / right_indices.len() as f64;

                // Calculate the squared error for this split.
                let left_error: f64 = left_indices.iter().map(|&i| (y[i] - left_avg).powi(2)).sum();
                let right_error: f64 = right_indices.iter().map(|&i| (y[i] - right_avg).powi(2)).sum();
                let total_error = left_error + right_error;

                // If this split is the best so far, save it.
                if total_error < min_error {
                    min_error = total_error;
                    best_stump = DecisionStump {
                        feature_index,
                        threshold,
                        left_value: left_avg,
                        right_value: right_avg,
                    };
                }
            }
        }
        best_stump
    }
}