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

/// use std::ffi::c_int;
unsafe extern "C" fn binary_map_fn(
    aaaaaaaaaaaaaaaaan: c_int,
    dst: *mut f32,
    src0: *const f32,
    src1: *const f32,
) {
    let dst = ::std::slice::from_raw_parts_mut(dst, n as usize);
    let src = ::std::slice::from_raw_parts(src, n as usize);
    dst.iter_mut()
        .zip(src.iter().copied())
        .for_each(|(dst, src0)| *dst = src + 10);
}
