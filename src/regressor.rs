// A decision stump (a tree with only one split).
#[derive(Debug)]
pub struct DecisionStump {
    // The index of the feature to split on.
    pub feature_index: usize,
    // The value to split the feature at.
    pub threshold: f64,
    // The prediction value if the feature is less than or equal to the threshold.
    pub left_value: f64,
    // The prediction value if the feature is greater than the threshold.
    pub right_value: f64,
}

// The main Gradient Boosting model.
#[derive(Debug)]
pub struct GradientBoostingRegressor {
    // A collection of weak learners (decision stumps).
    stumps: Vec<DecisionStump>,
    // The learning rate shrinks the contribution of each tree.
    learning_rate: f64,
    // The initial prediction, typically the mean of the target values.
    initial_prediction: f64,
    // Number of estimators to train.
    n_estimators: usize,
}

impl GradientBoostingRegressor {
    /// Create a new GradientBoostingRegressor.
    pub fn new(n_estimators: usize, learning_rate: f64) -> Self {
        Self {
            stumps: Vec::new(),
            learning_rate,
            initial_prediction: 0.0,
            n_estimators,
        }
    }

    /// Fit the model to data x (features) and y (targets).
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) {
        if x.is_empty() || y.is_empty() || x.len() != y.len() {
            return;
        }

        // initial prediction = mean(y)
        let mean = y.iter().sum::<f64>() / y.len() as f64;
        self.initial_prediction = mean;

        // current predictions
        let mut preds: Vec<f64> = vec![self.initial_prediction; y.len()];

        for _ in 0..self.n_estimators {
            // residuals = y - preds
            let residuals: Vec<f64> = y.iter().zip(preds.iter()).map(|(yi, pi)| yi - pi).collect();

            // fit a decision stump to residuals
            let stump = DecisionStump::fit(x, &residuals);
            // update predictions
            for (i, row) in x.iter().enumerate() {
                preds[i] += self.learning_rate * stump.predict(row);
            }
            self.stumps.push(stump);
        }
    }

    /// Predict outputs for given feature rows.
    pub fn predict(&self, x: &[Vec<f64>]) -> Vec<f64> {
        x.iter()
            .map(|row| {
                let mut pred = self.initial_prediction;
                for stump in &self.stumps {
                    pred += self.learning_rate * stump.predict(row);
                }
                pred
            })
            .collect()
    }
}