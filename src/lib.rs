use std::{cell::Cell, ops, rc::Rc};

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

    pub fn pow(&self, n: i32) -> Value {
        Value(Rc::new(_Value {
            data: self.data().powi(n),
            grad: Cell::new(0.0),
            done: Cell::new(false),
            op: Op::PowI(self.clone(), n),
        }))
    }

    pub fn relu(&self) -> Value {
        Value(Rc::new(_Value {
            data: if self.data() > 0.0 { self.data() } else { 0.0 },
            grad: Cell::new(0.0),
            done: Cell::new(false),
            op: Op::Relu(self.clone()),
        }))
    }
}

impl ops::Add<&Value> for &Value {
    type Output = Value;

    fn add(self, rhs: &Value) -> Self::Output {
        Value(Rc::new(_Value {
            data: self.data() + rhs.data(),
            grad: Cell::new(0.0),
            done: Cell::new(false),
            op: Op::Add(self.clone(), rhs.clone()),
        }))
    }
}

impl ops::Mul<&Value> for &Value {
    type Output = Value;

    fn mul(self, rhs: &Value) -> Self::Output {
        Value(Rc::new(_Value {
            data: self.data() * rhs.data(),
            grad: Cell::new(0.0),
            done: Cell::new(false),
            op: Op::Mul(self.clone(), rhs.clone()),
        }))
    }
}

impl ops::Add<f64> for &Value {
    type Output = Value;

    fn add(self, rhs: f64) -> Self::Output {
        self + &Value::new(rhs)
    }
}

impl ops::Add<&Value> for f64 {
    type Output = Value;

    fn add(self, rhs: &Value) -> Self::Output {
        &Value::new(self) + rhs
    }
}

impl ops::Mul<f64> for &Value {
    type Output = Value;

    fn mul(self, rhs: f64) -> Self::Output {
        self + &Value::new(rhs)
    }
}

impl ops::Mul<&Value> for f64 {
    type Output = Value;

    fn mul(self, rhs: &Value) -> Self::Output {
        &Value::new(self) * rhs
    }
}

impl ops::Neg for &Value {
    type Output = Value;

    fn neg(self) -> Self::Output {
        self * &Value::new(-1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ops_forward() {
        let a = Value::new(10.0);
        let b = Value::new(3.0);
        assert_eq!(-(&a).data(), -10.0);
        assert_eq!((&a + &b).data(), 13.0);
        assert_eq!((&a + 5.0).data(), 15.0);
        assert_eq!((&a * &b).data(), 30.0);
        assert_eq!((&a).pow(3).data(), 1000.0);

        assert_eq!(Value::new(1.0).relu().data(), 1.0);
        assert_eq!(Value::new(-1.0).relu().data(), 0.0);
    }

    #[test]
    fn add_backwards() {
        let a = Value::new(10.0);
        let b = Value::new(3.0);
        let c = &a + &b;
        c.set_grad(2.0);
        c.backwards();
        assert_eq!(a.grad(), 2.0);
        assert_eq!(b.grad(), 2.0);
    }

    #[test]
    fn mul_backwards() {
        let a = Value::new(10.0);
        let b = Value::new(3.0);
        let c = &a * &b;
        c.set_grad(2.0);
        c.backwards();
        assert_eq!(a.grad(), 6.0);
        assert_eq!(b.grad(), 20.0);
    }

    #[test]
    fn pow_backwards() {
        let a = Value::new(10.0);
        let c = (&a).pow(3);
        c.set_grad(2.0);
        c.backwards();
        assert_eq!(a.grad(), 3.0 * 10.0_f64.powi(2) * 2.0);
    }

    #[test]
    fn relu_backwards() {
        let a = Value::new(10.0);
        let c = a.relu();
        c.set_grad(2.0);
        c.backwards();
        assert_eq!(a.grad(), 2.0);

        let b = Value::new(-10.0);
        let d = b.relu();
        d.set_grad(2.0);
        d.backwards();
        assert_eq!(b.grad(), 0.0);
    }

    #[test]
    fn add_mul_neg_backwards() {
        let a = Value::new(10.0);
        let b = Value::new(3.0);
        let c = &a + &-(&(&a * &b));
        c.set_grad(3.0);
        c.backwards();
        assert_eq!(a.grad(), -6.0);
        assert_eq!(b.grad(), -30.0);
    }

    #[test]
    fn add_mul_neg_backwards_reset() {
        let a = Value::new(10.0);
        let b = Value::new(3.0);
        let c = &a + &-(&(&a * &b));

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
