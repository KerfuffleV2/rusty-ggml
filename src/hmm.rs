#![allow(unused_variables, unused_imports, dead_code)]

// Random ideas

use crate::{context::*, tensor::*, util::*};

fn testes() {
    let ctx = GgmlContextBuilder::new().mem_size(1024 * 1024).build();
    {
        let t1 = ctx.tensor(GgmlElementType::F32, [1]);
        let t2 = ctx.tensor(GgmlElementType::F32, [2]);
        let t3 = t1.add(t2);
        let t4 = ctx.tensor(GgmlElementType::F32, [1, 3]);
        let t5 = t4.mul_mat(t3);
        let t6 = t4.view([1, 2], [1, 2]);
        let t7 = ctx.tensor(GgmlElementType::F32, [1, 2, 3]);
        let t8 = t7.get_rows(t1);
    }
    {
        let t1 = ctx.tensor(GgmlElementType::F32, [1]);
        let t2 = ctx.tensor(GgmlElementType::F32, [2]);
        let t3 = &t1 + t2;
        let t4 = ctx.tensor(GgmlElementType::F32, [1, 3]);
        let t5 = t4.mul_mat(t3);
        let t6 = t4.view([1, 2], [1, 2]);
        let t7 = ctx.tensor(GgmlElementType::F32, [1, 2, 3]);
        let t8 = t7.get_rows(t1);
    }
}

// unsafe extern "C" fn derp<const X: usize>() {
//     println!("It is: {X}");
// }

// fn merp(x: unsafe extern "C" fn()) {}

// fn zerp() {
//     merp(derp::<1>);
//     merp(derp::<2>);
// }
