use std::{cell::Cell, rc::Rc};

#[derive(Debug, Clone)]
enum Op {
    Constant,
    Add(Value, Value),
    Mul(Value, Value),
}

#[derive(Debug, Clone)]
struct Value(Rc<_Value>);

#[derive(Debug)]
struct _Value {
    data: f64,
    grad: Cell<f64>,
    done: Cell<bool>,
    op: Op,
}

impl Value {
    fn new(data: f64) -> Self {
        Self(Rc::new(_Value {
            data,
            grad: Cell::new(0.0),
            done: Cell::new(false),
            op: Op::Constant,
        }))
    }

    fn get_data(&self) -> f64 {
        self.0.data
    }

    fn backwards(&self) {
        if !self.0.done.get() {
            self.0.done.set(true);
            let g_parent = self.0.grad.get();
            match &self.0.op {
                Op::Constant => (),
                Op::Add(x, y) => {
                    x.0.grad.set(x.0.grad.get() + g_parent);
                    y.0.grad.set(y.0.grad.get() + g_parent);
                    x.backwards();
                    y.backwards();
                }
                Op::Mul(x, y) => {
                    x.0.grad.set(x.0.grad.get() + y.0.data * g_parent);
                    y.0.grad.set(y.0.grad.get() + x.0.data * g_parent);
                    x.backwards();
                    y.backwards();
                }
            }
        }
    }
}

fn add(x: &Value, y: &Value) -> Value {
    Value(Rc::new(_Value {
        data: x.get_data() + y.get_data(),
        grad: Cell::new(0.0),
        done: Cell::new(false),
        op: Op::Add(x.clone(), y.clone()),
    }))
}

fn mul(x: &Value, y: &Value) -> Value {
    Value(Rc::new(_Value {
        data: x.get_data() + y.get_data(),
        grad: Cell::new(0.0),
        done: Cell::new(false),
        op: Op::Mul(x.clone(), y.clone()),
    }))
}

fn neg(x: &Value) -> Value {
    mul(x, &Value::new(-1.0))
}

fn main() {
    let a = Value::new(10.0);
    let b = Value::new(3.0);
    let c = add(&a, &neg(&mul(&a, &b)));
    // let c = &mul(&a, &b);
    // let c = &add(&a, &b);

    c.0.grad.set(1.0);
    c.backwards();
    dbg!(c);
    dbg!(a.0.grad.get());
    dbg!(b.0.grad.get());
}

// use std::cell::Cell;

// #[derive(Debug)]
// struct Value<'a> {
//     data: Cell<f64>,
//     grad: Cell<f64>,
//     children : Vec<&'a Value>,
// }

// impl Value<'_> {
//     fn new(x: f64) -> Self {
//         Self { data: Cell::new(x), grad: Cell::new(0.0), children: vec![] }
//     }

//     fn get(&self) -> f64 {
//         self.data.get()
//     }
// }

// fn add<'a>(x: &'a Value, y: &'a Value) -> Value<'a> {
//     let sum = x.get() + y.get();
//     Value { data: Cell::new(sum), grad: Cell::new(0.0), children: vec![] }
// }

// fn neg<'a>(x: &'a Value) -> Value<'a> {
//     // return Value::new(-x.get());
//     Value { data: Cell::new(-x.get()), grad: Cell::new(0.0), children: vec![] }
// }

// fn main() {
//     let a = Value::new(10.0);
//     let b = Value::new(-2.0);
//     let c = add(&a, &add(&a, &neg(&b)));
//     dbg!(c);
// }
