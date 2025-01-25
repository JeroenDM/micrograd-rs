#![feature(unboxed_closures)]

use micrograd_rs::*;
use rand::distributions::Uniform;
use rand::Rng;

#[derive(Debug)]
pub struct Neuron {
    weights: Vec<Value>,
    bias: Value,
}

impl Neuron {
    pub fn new(nin: usize) -> Self {
        let range = Uniform::from(-1.0..1.0);
        Self {
            weights: rand::thread_rng()
                .sample_iter(&range)
                .take(nin)
                .map(|x| Value::new(x))
                .collect(),
            bias: Value::new(rand::thread_rng().sample(&range)),
        }
    }

    pub fn forward(&self, x: &[Value]) -> Value {
        let y: Value = x
            .iter()
            .zip(&self.weights)
            .fold(self.bias.clone(), |acc, out| &acc + &(out.0 * out.1));
        return y.relu();
    }
}

struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(nin: usize, nout: usize) -> Self {
        let mut neurons = Vec::new();
        for _ in 0..nout {
            neurons.push(Neuron::new(nin));
        }
        Self { neurons }
    }

    pub fn forward(&self, x: &[Value]) -> Vec<Value> {
        let mut out = Vec::new();
        for n in &self.neurons {
            out.push(n.forward(x))
        }
        out
    }
}

struct MLP {
    layers : Vec<Layer>
}

impl MLP {
    pub fn new(nin : usize, nouts : &[usize]) -> Self {
        let mut layers = Vec::new();
        for nout in nouts {
            layers.push(Layer::new(nin, *nout));
        }
        Self { layers }
    }

    pub fn forward(&self, x: &[Value]) -> Vec<Value> {
        let mut out = Vec::from(x);
        for layer in &self.layers {
            out = layer.forward(&out);
        }
        out
    }
}

fn neuron() {
    let x = vec![Value::new(1.0), Value::new(2.0), Value::new(-2.0)];
    // let x = vec![3.0, 2.0, 1.0];

    // let y: Value = ws
    //     .iter()
    //     .zip(x)
    //     .fold(Value::new(0.0), |acc, out| &acc + &(out.0 * out.1));
    // dbg!(y.data());

    let net = MLP::new(3, &[5, 2]);
    let y = net.forward(&x);
    dbg!(y);
}

fn main() {
    // let a = Value::new(10.0);
    // let b = Value::new(3.0);
    // let c = &a + &-(&(&a * &b));
    // // let c = &mul(&a, &b);
    // // let c = &add(&a, &b);

    // c.set_grad(1.0);
    // c.backwards();
    // dbg!(a.grad());
    // dbg!(b.grad());

    // let a = Value::new(-4.0);
    // let b = Value::new(2.0);
    // let mut c = &a + &b;
    // let d = &(&a * &b) + &(b.pow(3));
    // c = &(&c + &c) + 1.0;

    // dbg!(c);
    // dbg!(d);
    neuron();
}
