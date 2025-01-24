use core::fmt;
use std::{cell::Cell, ops, rc::Rc};

struct Value<'a>(Rc<_Value<'a>>);

enum Op {
    None,
    Add,
    Sub,
}

pub struct _Value<'a> {
    data: f32,
    op: Op,
    children: Vec<&'a Value<'a>>,
    grad: Cell<f32>,
}

impl fmt::Display for Value<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.data)
    }
}

impl Value<'_> {
    pub fn new(data: f32) -> Self {
        Self(Rc::new(_Value {
            data,
            op: Op::None,
            children: Vec::new(),
            grad: Cell::new(0.0),
        }))
    }
}

impl<'a> ops::Add<&'a Value<'a>> for &'a Value<'a> {
    type Output = Value<'a>;

    fn add(self, rhs: &'a Value<'a>) -> Self::Output {
        Value(Rc::new(_Value {
            data: self.0.data + rhs.0.data,
            op: Op::Add,
            children: vec![self, &rhs],
            grad: Cell::new(0.0),
        }))
    }
}

impl<'a> ops::Neg for &'a Value<'a> {
    type Output = Value<'a>;

    fn neg(self) -> Self::Output {
        Value(Rc::new(_Value {
            data: -self.0.data,
            op: Op::Add,
            children: vec![self],
            grad: Cell::new(0.0),
        }))
    }
}

impl<'a> ops::Sub<&'a Value<'a>> for &'a Value<'a> {
    type Output = Value<'a>;

    fn sub(self, rhs: &'a Value<'a>) -> Self::Output {
        let temp = -rhs;
        let res = self + &temp;
        return res;
    }
}

fn print_tree(root: &Value, depth: i32) {
    print!("|");
    for _ in 0..depth {
        print!("--");
    }
    println!("[{}]", root);
    let n = root.0.children.len();
    if n > 0 {
        for r in &root.0.children {
            print_tree(r, depth + 1);
        }
    }
}

#[derive(Debug)]
struct Data<'a> {
    i: i32,
}

fn sub<'a>(a: &'a Data<'a>, b: &'a Data<'a>) -> Data<'a> {
    Data { i: a.i - b.i }
}

fn add<'l>(a: &'l Data<'l>, b: &'l Data<'l>) -> Data<'l> {
    let temp = Data::<'l> { i: a.i + b.i };
    sub(&a, &temp)
}

fn main() {
    let a = Value::new(32.0);
    let b = Value::new(3.0);
    // let c = Value::new(-1.22);
    let d = &a + &b;
    print_tree(&d, 0);
    // let e = &d - &c;
    // print_tree(&e, 0);

    let u = Data { i: 3 };
    let v = Data { i: 4 };
    let w = Data { i: 1 };
    let z = add(&add(&u, &v), &w);
    dbg!(z);
}
