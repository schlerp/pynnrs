extern crate nalgebra as na;
extern crate rand;

use std::ops::AddAssign;

use na::{Dyn, OMatrix, U1};
use pyo3::prelude::*;

type OMatrixDxD = OMatrix<f64, Dyn, Dyn>;
type OMatrixDx1 = OMatrix<f64, Dyn, U1>;

#[pyclass]
struct Layer {
    weights: OMatrixDxD,
    biases: OMatrixDx1,
    outputs: OMatrixDx1,
}

#[pyclass]
struct Network {
    layers: Vec<Layer>,
    learning_rate: f64,
}

#[pyclass]
struct EpochResult {
    epoch: usize,
    loss: f64,
}

#[pyclass]
struct TrainingHistory {
    epochs: usize,
    learning_rate: f64,
    epoch_results: Vec<EpochResult>,
}

fn relu(x: &OMatrixDx1) -> OMatrixDx1 {
    x.sup(&OMatrixDx1::zeros(x.shape().0))
}

fn relu_inv(z: &OMatrixDx1) -> OMatrixDx1 {
    z.map(|val: f64| -> f64 {
        if val > 0.0 {
            return 1.0;
        } else {
            return 0.0;
        }
    })
}

fn mean_scale(x: OMatrixDx1) -> OMatrixDx1 {
    (x.add_scalar(-x.mean())).unscale(x.max() - x.min())
}

fn mse(y: &OMatrixDx1, y_hat: &OMatrixDx1) -> f64 {
    0.5 * (y_hat - y).sum().powi(2)
}

impl Layer {
    fn new(input_size: usize, width: usize) -> Layer {
        let layer = Layer {
            weights: OMatrixDxD::new_random(width, input_size),
            biases: OMatrixDx1::zeros(width),
            outputs: OMatrixDx1::zeros(width),
        };
        layer
    }

    fn forward(&mut self, x: OMatrixDx1) -> OMatrixDx1 {
        self.outputs = &self.weights * &x + &self.biases;
        self.outputs.clone()
    }

    fn print_shapes(&mut self) {
        println!("weights: {:?}", self.weights.shape());
        println!("biases:  {:?}", self.biases.shape());
    }
}

#[pymethods]
impl Network {
    #[new]
    fn new(layer_sizes: Vec<usize>) -> Network {
        let mut network = Network {
            layers: Vec::new(),
            learning_rate: 0.01,
        };

        for layer_size_index in 0..(layer_sizes.len() - 1) {
            println!(
                "initializing layer {}, in: {}, out: {}",
                layer_size_index + 1,
                layer_sizes[layer_size_index],
                layer_sizes[layer_size_index + 1]
            );
            network.layers.push(Layer::new(
                layer_sizes[layer_size_index],
                layer_sizes[layer_size_index + 1],
            ))
        }

        return network;
    }

    fn eval(&mut self, x: Vec<f64>) -> Vec<f64> {
        let x_matrix: OMatrixDx1 = OMatrixDx1::from_vec(x);
        let y_hat: OMatrixDx1 = self.forward(&x_matrix);
        y_hat.as_slice().to_vec()
    }

    fn train(&mut self, x: Vec<f64>, y: Vec<f64>, epochs: usize) -> TrainingHistory {
        let x_matrix: OMatrixDx1 = OMatrixDx1::from_vec(x);
        let y_matrix: OMatrixDx1 = OMatrixDx1::from_vec(y);
        self._train(x_matrix, y_matrix, epochs)
    }
}

impl Network {
    fn forward(&mut self, x: &OMatrixDx1) -> OMatrixDx1 {
        let mut z = x.clone();
        for layer_index in 0..self.layers.len() {
            println!("layer: {:?}", layer_index + 1);
            println!("x in:    {:?}", z.shape());
            self.layers[layer_index].print_shapes();
            z = self.layers[layer_index].forward(z);
            z = relu(&z);
        }
        z
    }

    fn backward(&mut self, y: &OMatrixDx1, y_hat: &OMatrixDx1) {
        let error: OMatrixDx1 = y_hat.clone() - y;
        let prev_delta: OMatrixDx1 = error.component_mul(&relu_inv(y_hat));

        let mut deltas: Vec<OMatrixDx1> = vec![prev_delta.clone()];

        for layer_index in (0..(self.layers.len())).rev() {
            println!("prev_delta.T shape: {:?}", prev_delta.transpose().shape());
            println!(
                "weights shape:  {:?}",
                self.layers[layer_index].weights.shape()
            );
            let delta = &self.layers[layer_index].weights * &prev_delta.transpose();
            let delta = delta.component_mul(&relu_inv(&self.layers[layer_index].outputs.clone()));
            deltas.push(delta.clone());
        }

        // reverse our deltas
        deltas.reverse();

        // update our weights
        for layer_index in 0..self.layers.len() {
            let this_outputs = self.layers[layer_index].outputs.clone();
            let weight_update =
                (this_outputs.transpose() * &deltas[layer_index]).scale(-self.learning_rate);
            self.layers[layer_index].weights.add_assign(weight_update)
        }
    }

    fn _train(&mut self, x: OMatrixDx1, y: OMatrixDx1, epochs: usize) -> TrainingHistory {
        let mut epoch_results: Vec<EpochResult> = vec![];
        for i in 0..epochs {
            let y_hat = self.forward(&x);
            self.backward(&y, &y_hat);
            let loss = mse(&y, &y_hat);
            println!("epoch #{:?} completed, MSE: {:?}", i, loss);

            epoch_results.push(EpochResult { epoch: i, loss })
        }
        TrainingHistory {
            epochs,
            learning_rate: self.learning_rate,
            epoch_results,
        }
    }
}

#[pyfunction]
fn eval(network: &mut Network, x: Vec<f64>) -> PyResult<Vec<f64>> {
    Ok(network.eval(x))
}

#[pyfunction]
fn train(
    network: &mut Network,
    x: Vec<f64>,
    y: Vec<f64>,
    epochs: usize,
) -> PyResult<TrainingHistory> {
    Ok(network.train(x, y, epochs))
}

#[pymodule]
fn pynnrs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Layer>()?;
    m.add_class::<Network>()?;
    m.add_function(wrap_pyfunction!(eval, m)?)?;
    m.add_function(wrap_pyfunction!(train, m)?)?;
    Ok(())
}
