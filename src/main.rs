// imports in rust
// imports everything in ndarray which is numpy equivalent

//mod hyper_engine;   // tells Rust to look for src/hyper_engine.rs
// mod hyper_neuron;   // tells Rust to look for src/hyper_neuron.rs


//use ndarray::prelude::*;
// use rand::Rng;
// Declare the modules (the files must be named exactly like this)
mod hyper_engine;
use std::time::Instant;   
mod hyper_neuron;
use rand::Rng; 
use hyper_engine::Value;
use hyper_neuron::{MLP, Module};

// python code in micrograd
/*class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"
*/



   // to generate training data

fn main() {
    // Create the network: 3 inputs → 4 → 4 → 1 output
    let mlp = MLP::new(3, &[4, 4, 1]);

    // ---------- Generate synthetic training data ----------
    let mut rng = rand::thread_rng();
    let n_samples = 100;
    let xs: Vec<[f64; 3]> = (0..n_samples)
        .map(|_| {
            [
                rng.gen_range(-2.0..2.0),
                rng.gen_range(-2.0..2.0),
                rng.gen_range(-2.0..2.0),
            ]
        })
        .collect();
    let ys: Vec<f64> = xs
        .iter()
        .map(|&[x1, x2, x3]| 2.0 * x1 - 3.0 * x2 + 1.5 * x3 + 0.5)
        .collect();

    // ---------- Training hyperparameters ----------
    let epochs = 200;
    let learning_rate = 0.01;

    // ---------- Start profiling ----------
    let start = Instant::now();

    for epoch in 0..epochs {
        // ----- Forward pass: compute mean squared error loss -----
        let total_loss: Value = xs
            .iter()
            .zip(ys.iter())
            .map(|(x, &y_true)| {
                // Convert the f64 inputs into Value leaves
                let x_vals = vec![Value::new(x[0]), Value::new(x[1]), Value::new(x[2])];
                let pred = mlp.forward(&x_vals)[0].clone();   // single output
                let diff = pred - Value::new(y_true);
                diff.clone() * diff   // squared error (Value * Value)
            })
            .fold(Value::new(0.0), |acc, se| acc + se)
            * Value::new(1.0 / n_samples as f64);   // mean

        // ----- Backward pass -----
        mlp.zero_grad();
        total_loss.backward();

        // ----- Gradient descent update -----
        for param in mlp.parameters() {
            let new_val = param.data() - learning_rate * param.grad();
            param.set_data(new_val);
        }

        // ----- Progress every 20 epochs -----
        if epoch % 20 == 0 {
            println!(
                "Epoch {:3}: loss = {:.6}",
                epoch,
                total_loss.data()
            );
        }
    }
    // ---------- End profiling ----------
    let elapsed = start.elapsed();
    println!("Training completed in {:.4} seconds", elapsed.as_secs_f64());
    // ---------- Test on a new input ----------
    let test_x = vec![Value::new(1.5), Value::new(-0.5), Value::new(0.7)];
    let test_out = mlp.forward(&test_x)[0].clone();
    let expected = 2.0 * 1.5 - 3.0 * (-0.5) + 1.5 * 0.7 + 0.5;
    println!(
        "\nAfter training:\n  f(1.5, -0.5, 0.7) = {:.6}  (expected {:.6})",
        test_out.data(),
        expected
    );
}