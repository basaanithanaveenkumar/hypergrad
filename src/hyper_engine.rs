use std::cell::Cell;                  // Interior mutability for gradients
use std::collections::HashSet;        // For topological sort visited set
use std::ops::{Add, Mul, Neg, Sub}; // Operator overloading traits
use std::rc::Rc;                      // Reference counting to share graph nodes


// binary for operation with two operands

// unary for operations with one operand


// code is refactored by deepseek

/* Micrograd explanation Addition backwardSupposec = a + b
Derivative  dc/da = 1

dc/db = 1
So backward becomesa.grad += out.grad

b.grad += out.grad # 1*out.grad

Very simple.for multiplication

c=a*b
dc/da = b

dc/db = a



Why += instead of = ?This confuses almost everyone.Look at
a

 \
 *

 /
aS
uppose b = a * aThe variableais used twice.Derivativedb/da = 2aWhen backprop runs,left path contributesaright path contributesaTotala+a

=
2a

So gradients must be added.
*/


struct ValueData {
    data: Cell<f64>,          
    grad: Rc<Cell<f64>>,
    prev: Vec<Value>,
    backward: Option<Box<dyn Fn()>>,
}

#[derive(Clone)]
pub struct Value(Rc<ValueData>);

impl Value {
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

    pub fn pow(&self, exponent: f64) -> Value {
        let val = self.0.data.get();
        Value::unary(
            self.clone(),
            val.powf(exponent),
            move |upstream, _out, inp| upstream * exponent * inp.powf(exponent - 1.0),
        )
    }

    pub fn exp(self) -> Value {
        let val = self.0.data.get();
        let out = val.exp();
        Value::unary(self, out, move |upstream, output, _| upstream * output)
    }

    pub fn tanh(self) -> Value {
        let val = self.0.data.get();
        let out = val.tanh();
        Value::unary(self, out, move |upstream, output, _| upstream * (1.0 - output * output))
    }

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
    fn add(self, rhs: Value) -> Value {
        let lhs_data = self.0.data.get();
        let rhs_data = rhs.0.data.get();
        Value::binary(self, rhs, lhs_data + rhs_data, move |up, _, _| (up, up))
    }
}

impl Sub for Value {
    type Output = Value;
    fn sub(self, rhs: Value) -> Value {
        let lhs_data = self.0.data.get();
        let rhs_data = rhs.0.data.get();
        Value::binary(self, rhs, lhs_data - rhs_data, move |up, _, _| (up, -up))
    }
}

impl Mul for Value {
    type Output = Value;
    fn mul(self, rhs: Value) -> Value {
        let lhs_data = self.0.data.get();
        let rhs_data = rhs.0.data.get();
        Value::binary(self, rhs, lhs_data * rhs_data, move |up, lv, rv| (up * rv, up * lv))
    }
}

impl Neg for Value {
    type Output = Value;
    fn neg(self) -> Value {
        let data = self.0.data.get();
        Value::unary(self, -data, move |up, _, _| -up)
    }
}

impl Mul<Value> for f64 {
    type Output = Value;
    fn mul(self, rhs: Value) -> Value {
        Value::new(self) * rhs
    }
}

impl Mul<f64> for Value {
    type Output = Value;
    fn mul(self, rhs: f64) -> Value {
        self * Value::new(rhs)
    }
}