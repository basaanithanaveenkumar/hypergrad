// Python equivalent (micrograd) as comments:
// ==========================================
// class Value:
//     def __init__(self, data, _children=(), _op=''):
//         self.data = data
//         self.grad = 0.0
//         self._backward = lambda: None
//         self._prev = set(_children)
//         self._op = _op
//
//     def __add__(self, other):
//         other = other if isinstance(other, Value) else Value(other)
//         out = Value(self.data + other.data, (self, other), '+')
//         def _backward():
//             self.grad += out.grad
//             other.grad += out.grad
//         out._backward = _backward
//         return out
//
//     def __mul__(self, other):
//         other = other if isinstance(other, Value) else Value(other)
//         out = Value(self.data * other.data, (self, other), '*')
//         def _backward():
//             self.grad += other.data * out.grad
//             other.grad += self.data * out.grad
//         out._backward = _backward
//         return out
//
//     def __pow__(self, other):
//         assert isinstance(other, (int, float)), "only supporting int/float powers"
//         out = Value(self.data**other, (self,), f'**{other}')
//         def _backward():
//             self.grad += other * (self.data ** (other-1)) * out.grad
//         out._backward = _backward
//         return out
//
//     def relu(self):
//         out = Value(self.data if self.data > 0 else 0.0, (self,), 'ReLU')
//         def _backward():
//             self.grad += (out.data > 0) * out.grad
//         out._backward = _backward
//         return out
//
//     def tanh(self):
//         x = self.data
//         t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
//         out = Value(t, (self,), 'tanh')
//         def _backward():
//             self.grad += (1 - t**2) * out.grad
//         out._backward = _backward
//         return out
//
//     def exp(self):
//         out = Value(math.exp(self.data), (self,), 'exp')
//         def _backward():
//             self.grad += out.data * out.grad
//         out._backward = _backward
//         return out
//
//     def backward(self):
//         topo = []
//         visited = set()
//         def build_topo(v):
//             if v not in visited:
//                 visited.add(v)
//                 for child in v._prev:
//                     build_topo(child)
//                 topo.append(v)
//         build_topo(self)
//         self.grad = 1.0
//         for v in reversed(topo):
//             v._backward()

use std::cell::Cell;                  // Interior mutability for gradients
use std::collections::HashSet;        // For topological sort visited set
use std::ops::{Add, Mul, Neg, Sub}; // Operator overloading traits
use std::rc::Rc;                      // Reference counting to share graph nodes

// binary for operation with two operands
// unary for operations with one operand
// code is refactored by deepseek

/* Micrograd explanation Addition backward
Suppose c = a + b
Derivative dc/da = 1, dc/db = 1
So backward becomes a.grad += out.grad, b.grad += out.grad # 1*out.grad
Very simple.
For multiplication c = a*b, dc/da = b, dc/db = a

Why += instead of =? This confuses almost everyone.
Look at a * a, variable a is used twice.
Derivative db/da = 2a. When backprop runs, left path contributes a, right path contributes a. Total a+a = 2a.
So gradients must be added.
*/

// Python equivalent struct: Value holds data, grad, prev (children), backward (closure)
struct ValueData {
    data: Cell<f64>,          
    grad: Rc<Cell<f64>>,
    prev: Vec<Value>,
    backward: Option<Box<dyn Fn()>>,
}

#[derive(Clone)]
pub struct Value(Rc<ValueData>);

impl Value {
    // Python __init__: 
    // def __init__(self, data, _children=(), _op=''):
    //     self.data = data
    //     self.grad = 0.0
    //     self._backward = lambda: None
    //     self._prev = set(_children)
    pub fn new(data: f64) -> Self {
        Value(Rc::new(ValueData {
            data: Cell::new(data),
            grad: Rc::new(Cell::new(0.0)),
            prev: Vec::new(),
            backward: None,
        }))
    }

    pub fn data(&self) -> f64 {
        self.0.data.get()
    }

    pub fn set_data(&self, val: f64) {
        self.0.data.set(val);
    }

    pub fn grad(&self) -> f64 {
        self.0.grad.get()
    }

    pub fn set_grad(&self, g: f64) {
        self.0.grad.set(g);
    }

    // Helper for binary operations: creates a new Value and sets its backward closure.
    // Equivalent to the pattern in Python where each op creates out with _backward.
    fn binary(
        lhs: Value,
        rhs: Value,
        output_value: f64,
        local_grad_fn: impl Fn(f64, f64, f64) -> (f64, f64) + 'static,
    ) -> Value {
        let inputs = vec![lhs.clone(), rhs.clone()];
        let output_grad_cell = Rc::new(Cell::new(0.0));

        let lhs_grad_cell = lhs.0.grad.clone();
        let rhs_grad_cell = rhs.0.grad.clone();
        let lhs_value = lhs.0.data.get();
        let rhs_value = rhs.0.data.get();

        let backward_closure = {
            let output_grad_cell = output_grad_cell.clone();
            move || {
                let upstream = output_grad_cell.get();
                let (gl, gr) = local_grad_fn(upstream, lhs_value, rhs_value);
                lhs_grad_cell.set(lhs_grad_cell.get() + gl);
                rhs_grad_cell.set(rhs_grad_cell.get() + gr);
            }
        };

        Value(Rc::new(ValueData {
            data: Cell::new(output_value),
            grad: output_grad_cell,
            prev: inputs,
            backward: Some(Box::new(backward_closure)),
        }))
    }

    // Helper for unary operations: similar to binary.
    fn unary(
        input: Value,
        output_value: f64,
        local_grad_fn: impl Fn(f64, f64, f64) -> f64 + 'static,
    ) -> Value {
        let inputs = vec![input.clone()];
        let output_grad_cell = Rc::new(Cell::new(0.0));
        let input_grad_cell = input.0.grad.clone();
        let input_value = input.0.data.get();

        let backward_closure = {
            let output_grad_cell = output_grad_cell.clone();
            move || {
                let upstream = output_grad_cell.get();
                let gi = local_grad_fn(upstream, output_value, input_value);
                input_grad_cell.set(input_grad_cell.get() + gi);
            }
        };

        Value(Rc::new(ValueData {
            data: Cell::new(output_value),
            grad: output_grad_cell,
            prev: inputs,
            backward: Some(Box::new(backward_closure)),
        }))
    }

    // Python backward:
    // def backward(self):
    //     topo = []
    //     visited = set()
    //     def build_topo(v):
    //         if v not in visited:
    //             visited.add(v)
    //             for child in v._prev:
    //                 build_topo(child)
    //             topo.append(v)
    //     build_topo(self)
    //     self.grad = 1.0
    //     for v in reversed(topo):
    //         v._backward()
    pub fn backward(&self) {
        self.0.grad.set(1.0);
        let mut topo = Vec::new();
        let mut visited = HashSet::new();

        fn build_topo(node: Value, topo: &mut Vec<Value>, visited: &mut HashSet<usize>) {
            let ptr = Rc::as_ptr(&node.0) as usize;
            if !visited.insert(ptr) {
                return;
            }
            for child in &node.0.prev {
                build_topo(child.clone(), topo, visited);
            }
            topo.push(node.clone());
        }

        build_topo(self.clone(), &mut topo, &mut visited);

        for node in topo.iter().rev() {
            if let Some(ref backward_fn) = node.0.backward {
                backward_fn();
            }
        }
    }

    // Python __pow__: out = Value(self.data**other, (self,), f'**{other}')
    // def _backward(): self.grad += other * (self.data ** (other-1)) * out.grad
    pub fn pow(&self, exponent: f64) -> Value {
        let val = self.0.data.get();
        Value::unary(
            self.clone(),
            val.powf(exponent),
            move |upstream, _out, inp| upstream * exponent * inp.powf(exponent - 1.0),
        )
    }

    // Python exp: out = Value(math.exp(self.data), (self,), 'exp')
    // def _backward(): self.grad += out.data * out.grad
    pub fn exp(self) -> Value {
        let val = self.0.data.get();
        let out = val.exp();
        Value::unary(self, out, move |upstream, output, _| upstream * output)
    }

    // Python tanh: out = Value(t, (self,), 'tanh')
    // def _backward(): self.grad += (1 - t**2) * out.grad
    pub fn tanh(self) -> Value {
        let val = self.0.data.get();
        let out = val.tanh();
        Value::unary(self, out, move |upstream, output, _| upstream * (1.0 - output * output))
    }

    // Python relu: out = Value(self.data if self.data > 0 else 0.0, (self,), 'ReLU')
    // def _backward(): self.grad += (out.data > 0) * out.grad
    pub fn relu(self) -> Value {
        let val = self.0.data.get();
        let out = val.max(0.0);
        Value::unary(self, out, move |upstream, _out, inp| {
            if inp > 0.0 { upstream } else { 0.0 }
        })
    }
}

impl Add for Value {
    type Output = Value;
    // Python __add__:
    // out = Value(self.data + other.data, (self, other), '+')
    // def _backward(): self.grad += out.grad; other.grad += out.grad
    fn add(self, rhs: Value) -> Value {
        let lhs_data = self.0.data.get();
        let rhs_data = rhs.0.data.get();
        Value::binary(self, rhs, lhs_data + rhs_data, move |up, _, _| (up, up))
    }
}

impl Sub for Value {
    type Output = Value;
    // Python __sub__: implemented as self + (-other) or directly.
    // def __sub__(self, other): return self + (-other)
    // Equivalent: local grad (up, -up)
    fn sub(self, rhs: Value) -> Value {
        let lhs_data = self.0.data.get();
        let rhs_data = rhs.0.data.get();
        Value::binary(self, rhs, lhs_data - rhs_data, move |up, _, _| (up, -up))
    }
}

impl Mul for Value {
    type Output = Value;
    // Python __mul__:
    // out = Value(self.data * other.data, (self, other), '*')
    // def _backward(): self.grad += other.data * out.grad; other.grad += self.data * out.grad
    fn mul(self, rhs: Value) -> Value {
        let lhs_data = self.0.data.get();
        let rhs_data = rhs.0.data.get();
        Value::binary(self, rhs, lhs_data * rhs_data, move |up, lv, rv| (up * rv, up * lv))
    }
}

impl Neg for Value {
    type Output = Value;
    // Python __neg__: return self * -1
    // def __neg__(self): return self * -1
    // Or directly: out = Value(-self.data, (self,), 'neg')
    // def _backward(): self.grad += -out.grad
    fn neg(self) -> Value {
        let data = self.0.data.get();
        Value::unary(self, -data, move |up, _, _| -up)
    }
}

// Python supports left multiplication by scalar: 2 * Value(3) -> __rmul__
// Python __rmul__: return self * other (if other is scalar)
impl Mul<Value> for f64 {
    type Output = Value;
    fn mul(self, rhs: Value) -> Value {
        Value::new(self) * rhs
    }
}

// Python right multiplication by scalar: Value(3) * 2 -> __mul__ already handles via Mul<f64>? 
// But in Rust we need separate impl.
impl Mul<f64> for Value {
    type Output = Value;
    fn mul(self, rhs: f64) -> Value {
        self * Value::new(rhs)
    }
}