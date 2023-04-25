#![allow(unused_variables, unused_imports, dead_code)]

// Random ideas

use crate::{context::*, tensor::*};

fn testes() {
    let mut ctx = GgmlContextBuilder::new().mem_size(1024 * 1024).build();
    {
        let t1 = ctx.tensor(GgmlElementType::F32, [1]);
        let t2 = ctx.tensor(GgmlElementType::F32, [2]);
        let t3 = t1.add(t2);
        let t4 = ctx.tensor(GgmlElementType::F32, [1, 3]);
        let t5 = t4.mul_mat_smallrhs(t3);
        let t6 = t4.view([1, 2], [1, 2]);
        let t7 = ctx.tensor(GgmlElementType::F32, [1, 2, 3]);
        let t8 = t7.get_rows(t1);
    }
    {
        let t1 = ctx.tensor(GgmlElementType::F32, [1]);
        let t2 = ctx.tensor(GgmlElementType::F32, [2]);
        let t3 = &t1 + t2;
        let t4 = ctx.tensor(GgmlElementType::F32, [1, 3]);
        let t5 = t4.mul_mat_smallrhs(t3);
        let t6 = t4.view([1, 2], [1, 2]);
        let t7 = ctx.tensor(GgmlElementType::F32, [1, 2, 3]);
        let t8 = t7.get_rows(t1);
    }
}

// pub trait GgmlMulMat<const LDIMS: usize, const RDIMS: usize>
// where
//     Dim<LDIMS>: DimValid,
//     Dim<RDIMS>: DimValid,
// {
//     type Output;
//     fn mul_mat<T: AsRef<Tensor<LDIMS>>>(&self, rhs: T) -> Tensor<RDIMS>;
// }

// impl<const LDIMS: usize, const RDIMS: usize> GgmlMulMat<LDIMS, RDIMS> for Tensor<RDIMS>
// where
//     Dim<LDIMS>: DimValid,
//     Dim<RDIMS>: DimValid,
//     DimPair<LDIMS, RDIMS>: DimGtE,
// {
//     type Output = bool;
//     fn mul_mat<T: AsRef<Tensor<LDIMS>>>(&self, rhs: T) -> Tensor<RDIMS> {
//         todo!()
//     }
// }

// impl<const LDIMS: usize, const RDIMS: usize> GgmlMulMat<LDIMS, RDIMS> for Tensor<RDIMS>
// where
//     Dim<LDIMS>: DimValid,
//     Dim<RDIMS>: DimValid,
//     DimPair<LDIMS, RDIMS>: DimLt,
// {
//     type Output = ();
//     fn mul_mat<T: AsRef<Tensor<LDIMS>>>(&self, rhs: T) -> Tensor<RDIMS> {
//         todo!()
//     }
// }

// unsafe extern "C" fn derp<const X: usize>() {
//     println!("It is: {X}");
// }

// fn merp(x: unsafe extern "C" fn()) {}

// fn zerp() {
//     merp(derp::<1>);
//     merp(derp::<2>);
// }
