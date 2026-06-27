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



//mod hyper_engine;   // tells Rust to look for src/hyper_engine.rs


use crate::hyper_engine::Value;           // Your existing autograd engine
use rand::Rng;               // For random weight initialisation
use std::fmt;                // For nice printing (like Python's __repr__)

// ─────────────────────────────────────────────────────────────────
// 1. Module trait – equivalent to the Python `Module` class
//    In Rust, a trait defines shared behaviour (like a Python ABC).
// ─────────────────────────────────────────────────────────────────
pub trait Module {
    /// Return a list of all trainable parameters (weights and biases).
    /// In Rust we return a Vec<Value> (a vector of shared pointers).
    fn parameters(&self) -> Vec<Value>;

    /// Set the gradient of every parameter to zero.
    /// This should be called before each backward pass.
    fn zero_grad(&self) {
        for p in self.parameters() {
            p.set_grad(0.0);        // p.grad is Rc<Cell<f64>>, .set() mutates it
        }
    }
}

// ─────────────────────────────────────────────────────────────────
// 2. Neuron – equivalent to `class Neuron(Module)`
//    A single neuron with weights, a bias, and an optional ReLU.
// ─────────────────────────────────────────────────────────────────
pub struct Neuron {
    w: Vec<Value>,      // list of weight Values (each is an Rc<ValueData>)
    b: Value,           // bias
    nonlin: bool,       // whether to apply ReLU activation
}

impl Neuron {
    /// Create a new Neuron with `nin` inputs.
    /// `nonlin` = true adds a ReLU after the linear combination.
    pub fn new(nin: usize, nonlin: bool) -> Self {
        let mut rng = rand::thread_rng();   // like Python's random module
        let w = (0..nin)
            .map(|_| Value::new(rng.gen_range(-1.0..1.0)))
            .collect();                     // list comprehension equivalent
        let b = Value::new(0.0);
        Neuron { w, b, nonlin }
    }

    /// Forward pass: given a slice of input Values, compute the output Value.
    /// In Python this is `__call__(self, x)`. Here we use a method.
    pub fn forward(&self, x: &[Value]) -> Value {
        // weighted sum: sum(wi * xi for wi,xi in zip(self.w, x))
        // Rust: zip two iterators, map to wi*xi, then sum with fold starting from bias
        let act = self
            .w
            .iter()                     // iterate over weights
            .zip(x.iter())              // pair each weight with an input
            .map(|(wi, xi)| wi.clone() * xi.clone())  // Value * Value (need clones)
            .fold(self.b.clone(), |acc, term| acc + term); // sum += term; start with bias

        if self.nonlin {
            act.relu()                  // apply ReLU (returns a new Value node)
        } else {
            act                        // linear neuron, no activation
        }
    }
}

impl Module for Neuron {
    fn parameters(&self) -> Vec<Value> {
        // Return a new vector containing all weights followed by the bias.
        // Rust: we can use [self.w.clone(), vec![self.b.clone()]].concat()
        let mut params = self.w.clone();
        params.push(self.b.clone());
        params
    }
}

impl fmt::Display for Neuron {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}Neuron({})",
            if self.nonlin { "ReLU" } else { "Linear" },
            self.w.len()
        )
    }
}

// ─────────────────────────────────────────────────────────────────
// 3. Layer – equivalent to `class Layer(Module)`
//    A layer contains a list of Neurons, all with the same number of inputs.
// ─────────────────────────────────────────────────────────────────
pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    /// Create a new Layer with `nin` inputs and `nout` neurons (outputs).
    /// `nonlin` is passed to every Neuron.
    pub fn new(nin: usize, nout: usize, nonlin: bool) -> Self {
        let neurons = (0..nout)
            .map(|_| Neuron::new(nin, nonlin))
            .collect();
        Layer { neurons }
    }

    /// Forward pass: apply each neuron to the input slice.
    /// Returns a vector of outputs (one per neuron).
    /// For consistency with the Python version, if there is only one output
    /// we return it directly wrapped in a Vec (but Python returns the scalar).
    /// We'll keep it as a Vec<Value> for simplicity.
    pub fn forward(&self, x: &[Value]) -> Vec<Value> {
        self.neurons.iter().map(|n| n.forward(x)).collect()
    }
}

impl Module for Layer {
    fn parameters(&self) -> Vec<Value> {
        // Flatten: for each neuron, collect its parameters
        self.neurons
            .iter()
            .flat_map(|n| n.parameters().into_iter())
            .collect()
    }
}

impl fmt::Display for Layer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let neuron_strs: Vec<String> = self.neurons.iter().map(|n| n.to_string()).collect();
        write!(f, "Layer of [{}]", neuron_strs.join(", "))
    }
}

// ─────────────────────────────────────────────────────────────────
// 4. MLP – equivalent to `class MLP(Module)`
//    A multi‑layer perceptron: a list of Layers.
// ─────────────────────────────────────────────────────────────────
pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    /// Create an MLP.
    /// `nin`  : number of input features
    /// `nouts`: sizes of the hidden/output layers (e.g., [4, 4, 1])
    ///
    /// All layers except the last use ReLU (nonlin = true).
    pub fn new(nin: usize, nouts: &[usize]) -> Self {
        // Build the list of sizes: [nin, nouts[0], nouts[1], ...]
        let mut sizes = vec![nin];
        sizes.extend_from_slice(nouts);

        let layers = (0..nouts.len())
            .map(|i| {
                let nonlin = i != nouts.len() - 1; // True for all but last layer
                Layer::new(sizes[i], sizes[i + 1], nonlin)
            })
            .collect();

        MLP { layers }
    }

    /// Forward pass: pass the input slice through every layer sequentially.
    pub fn forward(&self, x: &[Value]) -> Vec<Value> {
        // Start with the input vector.
        // For each layer, apply forward and update the current vector.
        let mut out = x.to_vec();   // clone the input Values
        for layer in &self.layers {
            out = layer.forward(&out);
        }
        out
    }
}

impl Module for MLP {
    fn parameters(&self) -> Vec<Value> {
        self.layers
            .iter()
            .flat_map(|l| l.parameters().into_iter())
            .collect()
    }
}

impl fmt::Display for MLP {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let layer_strs: Vec<String> = self.layers.iter().map(|l| l.to_string()).collect();
        write!(f, "MLP of [{}]", layer_strs.join(", "))
    }
}

