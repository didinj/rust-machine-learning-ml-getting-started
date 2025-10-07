use linfa::prelude::*;
use linfa_datasets::iris;
use linfa_trees::DecisionTree;
use std::collections::HashSet;

fn main() {
    // Load the Iris dataset
    let dataset = iris();

    // Display basic dataset information
    println!("Number of samples: {}", dataset.nsamples());
    println!("Number of features: {}", dataset.nfeatures());
    let unique_targets: HashSet<_> = dataset.targets().iter().cloned().collect();
    println!("Target names: {:?}", unique_targets);

    // Split the dataset into 80% training and 20% testing
    let (train, test) = dataset.split_with_ratio(0.8);

    println!("Training samples: {}", train.nsamples());
    println!("Testing samples: {}", test.nsamples());

    // Train the Decision Tree model
    let model = DecisionTree::params().fit(&train).unwrap();

    // Predict using the test data
    let predictions = model.predict(&test);

    // Evaluate model accuracy
    let cm = predictions.confusion_matrix(&test).unwrap();
    let accuracy = cm.accuracy();
    println!("Model accuracy: {:.2}%", accuracy * 100.0);
}

// === Unit tests ===
#[cfg(test)]
mod tests {
    use super::*;
    use linfa_datasets::iris;
    use linfa_nn::{ LinearSearch, NearestNeighbour, distance::L2Dist };
    use ndarray::Array1;
    use std::collections::HashMap;

    #[test]
    fn test_knn_accuracy() {
        let dataset = iris();
        let (train, valid) = dataset.split_with_ratio(0.8);

        let train_records = train.records().to_owned();
        let train_targets = train.targets().to_owned();
        let valid_records = valid.records();

        let nn_index = LinearSearch::new()
            .from_batch(&train_records, L2Dist)
            .expect("Failed to build index");

        let mut predictions = Vec::with_capacity(valid_records.nrows());
        for sample in valid_records.outer_iter() {
            let neighbours = nn_index.k_nearest(sample, 3).unwrap();

            let mut votes = HashMap::new();
            for (_pt, idx) in neighbours {
                *votes.entry(train_targets[idx]).or_insert(0) += 1;
            }
            let predicted = votes
                .into_iter()
                .max_by_key(|(_, c)| *c)
                .unwrap().0;
            predictions.push(predicted);
        }

        let pred_ds = Dataset::new(valid_records.to_owned(), Array1::from(predictions));
        let acc = pred_ds.confusion_matrix(&valid).unwrap().accuracy();
        assert!(acc > 0.8, "Model accuracy too low: {}", acc);
    }
}
