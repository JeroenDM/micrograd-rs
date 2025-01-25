use std::{cell::Cell, rc::Rc};

/// Wrapper type to allow storing the nodes in a graph
/// without having to wrap it into an Rc manually everywhere.
#[derive(Debug, Clone)]
pub struct Value(Rc<_Value>);

#[derive(Debug)]
struct _Value {
    data: f64,
    grad: Cell<f64>,
    done: Cell<bool>,
    op: Op,
}

/// Enum describing a directed acyclic computation graph.
#[derive(Debug, Clone)]
enum Op {
    Constant,
    Add(Value, Value),
    Mul(Value, Value),
    PowI(Value, i32),
    Relu(Value),
}

impl Value {
    pub fn new(data: f64) -> Self {
        Self(Rc::new(_Value {
            data,
            grad: Cell::new(0.0),
            done: Cell::new(false),
            op: Op::Constant,
        }))
    }

    pub fn data(&self) -> f64 {
        self.0.data
    }

    pub fn grad(&self) -> f64 {
        self.0.grad.get()
    }

    pub fn set_grad(&self, x: f64) {
        self.0.grad.set(x)
    }

    pub fn get_data(&self) -> f64 {
        self.0.data
    }

    pub fn backwards(&self) {
        if !self.0.done.get() {
            self.0.done.set(true);
            let g_parent = self.0.grad.get();
            match &self.0.op {
                Op::Constant => (),
                Op::Add(x, y) => {
                    x.set_grad(x.grad() + g_parent);
                    y.set_grad(y.grad() + g_parent);
                    x.backwards();
                    y.backwards();
                }
                Op::Mul(x, y) => {
                    x.set_grad(x.grad() + y.0.data * g_parent);
                    y.set_grad(y.grad() + x.0.data * g_parent);
                    x.backwards();
                    y.backwards();
                }
                Op::PowI(x, n) => {
                    x.set_grad(x.grad() + (*n as f64) * x.data().powi(n - 1) * g_parent);
                    x.backwards();
                }
                Op::Relu(x) => {
                    x.set_grad(x.grad() + if x.data() > 0.0 { g_parent } else { 0.0 });
                    x.backwards();
                }
            }
        }
    }

    pub fn reset(&self) {
        if self.0.done.get() {
            self.set_grad(0.0);
            self.0.done.set(false);
            match &self.0.op {
                Op::Constant => (),
                Op::Add(x, y) => {
                    x.reset();
                    y.reset();
                }
                Op::Mul(x, y) => {
                    x.reset();
                    y.reset();
                }
                Op::PowI(x, _) => x.reset(),
                Op::Relu(x) => x.reset(),
            }
        }
    }
}

pub fn add(x: &Value, y: &Value) -> Value {
    Value(Rc::new(_Value {
        data: x.get_data() + y.get_data(),
        grad: Cell::new(0.0),
        done: Cell::new(false),
        op: Op::Add(x.clone(), y.clone()),
    }))
}

pub fn mul(x: &Value, y: &Value) -> Value {
    Value(Rc::new(_Value {
        data: x.get_data() * y.get_data(),
        grad: Cell::new(0.0),
        done: Cell::new(false),
        op: Op::Mul(x.clone(), y.clone()),
    }))
}

pub fn pow(x: &Value, n: i32) -> Value {
    Value(Rc::new(_Value {
        data: x.get_data().powi(n),
        grad: Cell::new(0.0),
        done: Cell::new(false),
        op: Op::PowI(x.clone(), n),
    }))
}

pub fn relu(x: &Value) -> Value {
    Value(Rc::new(_Value {
        data: if x.data() > 0.0 { x.data() } else { 0.0 },
        grad: Cell::new(0.0),
        done: Cell::new(false),
        op: Op::Relu(x.clone()),
    }))
}

pub fn neg(x: &Value) -> Value {
    mul(x, &Value::new(-1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ops_forward() {
        let a = Value::new(10.0);
        let b = Value::new(3.0);
        assert_eq!(neg(&a).data(), -10.0);
        assert_eq!(add(&a, &b).data(), 13.0);
        assert_eq!(mul(&a, &b).data(), 30.0);
        assert_eq!(pow(&a, 3).data(), 1000.0);

        assert_eq!(relu(&Value::new(1.0)).data(), 1.0);
        assert_eq!(relu(&Value::new(-1.0)).data(), 0.0);
    }

    #[test]
    fn add_backwards() {
        let a = Value::new(10.0);
        let b = Value::new(3.0);
        let c = add(&a, &b);
        c.set_grad(2.0);
        c.backwards();
        assert_eq!(a.grad(), 2.0);
        assert_eq!(b.grad(), 2.0);
    }

    #[test]
    fn mul_backwards() {
        let a = Value::new(10.0);
        let b = Value::new(3.0);
        let c = mul(&a, &b);
        c.set_grad(2.0);
        c.backwards();
        assert_eq!(a.grad(), 6.0);
        assert_eq!(b.grad(), 20.0);
    }

    #[test]
    fn pow_backwards() {
        let a = Value::new(10.0);
        let c = pow(&a, 3);
        c.set_grad(2.0);
        c.backwards();
        assert_eq!(a.grad(), 3.0 * 10.0_f64.powi(2) * 2.0);
    }

    #[test]
    fn relu_backwards() {
        let a = Value::new(10.0);
        let c = relu(&a);
        c.set_grad(2.0);
        c.backwards();
        assert_eq!(a.grad(), 2.0);

        let b = Value::new(-10.0);
        let d = relu(&b);
        d.set_grad(2.0);
        d.backwards();
        assert_eq!(b.grad(), 0.0);
    }

    #[test]
    fn add_mul_neg_backwards() {
        let a = Value::new(10.0);
        let b = Value::new(3.0);
        let c = add(&a, &neg(&mul(&a, &b)));
        c.set_grad(3.0);
        c.backwards();
        assert_eq!(a.grad(), -6.0);
        assert_eq!(b.grad(), -30.0);
    }

    #[test]
    fn add_mul_neg_backwards_reset() {
        let a = Value::new(10.0);
        let b = Value::new(3.0);
        let c = add(&a, &neg(&mul(&a, &b)));

        c.set_grad(1.0);
        c.backwards();
        assert_eq!(a.grad(), -2.0);
        assert_eq!(b.grad(), -10.0);

        c.reset();
        c.set_grad(1.0);
        c.backwards();
        assert_eq!(a.grad(), -2.0);
        assert_eq!(b.grad(), -10.0);

        c.reset();
        c.set_grad(2.0);
        c.backwards();
        assert_eq!(a.grad(), -4.0);
        assert_eq!(b.grad(), -20.0);
    }
}
