use micrograd_rs::*;

fn main() {
    let a = Value::new(10.0);
    let b = Value::new(3.0);
    let c = add(&a, &neg(&mul(&a, &b)));
    // let c = &mul(&a, &b);
    // let c = &add(&a, &b);

    c.set_grad(1.0);
    c.backwards();
    dbg!(a.grad());
    dbg!(b.grad());
}

