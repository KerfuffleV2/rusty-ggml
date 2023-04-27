use std::ops;

use ggml_sys_bleedingedge as gg;

use super::tensor::*;
use crate::dims::*;

impl<const DIMS: usize> GTensor<DIMS> where Dim<DIMS>: DimValid {}

impl<'a, const DIMS: usize, T: AsRef<GTensor<DIMS>>> ops::BitXor<T> for &'a GTensor<DIMS>
where
    Dim<DIMS>: DimValid,
    GTensor<DIMS>: GMulMat<DIMS, DIMS>,
{
    type Output = <GTensor<DIMS> as GMulMat<DIMS, DIMS>>::Output;

    fn bitxor(self, rhs: T) -> Self::Output {
        GTensor::mul_mat(self, rhs.as_ref())
    }
}

impl<const DIMS: usize, T: AsRef<GTensor<DIMS>>> ops::BitXor<T> for GTensor<DIMS>
where
    Dim<DIMS>: DimValid,
    Self: GMulMat<DIMS, DIMS>,
{
    type Output = <Self as GMulMat<DIMS, DIMS>>::Output;

    fn bitxor(self, rhs: T) -> Self::Output {
        &self ^ rhs.as_ref()
    }
}

// FIXME: There should be a better way to do this.
pub trait GMulMat<const LDIMS: usize, const RDIMS: usize>
where
    Dim<LDIMS>: DimValid,
    Dim<RDIMS>: DimValid,
{
    type Output;

    fn mul_mat<T: AsRef<GTensor<RDIMS>>>(&self, rhs: T) -> Self::Output;
}

impl<const DIMS: usize> GMulMat<DIMS, DIMS> for GTensor<DIMS>
where
    Dim<DIMS>: DimValid,
{
    type Output = GTensor<DIMS>;

    fn mul_mat<T: AsRef<GTensor<DIMS>>>(&self, rhs: T) -> Self::Output {
        self.new_binary(rhs, |ctx, ltptr, rtptr| unsafe {
            gg::ggml_mul_mat(ctx, ltptr, rtptr)
        })
    }
}

// This is rather unpleasant.
macro_rules! mk_gmulmatinstances {
    ( $( ($l:literal, $r:literal, $o:literal) ),+ ) => { $(
        impl GMulMat<$l, $r> for GTensor<$l> {
            type Output = GTensor<$o>;

            fn mul_mat<T: AsRef<GTensor<$r>>>(&self, rhs: T) -> Self::Output {
                self.new_binary(rhs, |ctx, ltptr, rtptr| unsafe {
                    gg::ggml_mul_mat(ctx, ltptr, rtptr)
                })
            }
        }

        impl<'a, 'b> ops::BitXor<&'b GTensor<$r>> for &'a GTensor<$l>
        where
            GTensor<$l>: GMulMat<$l, $r>,
        {
            type Output = GTensor<$o>;

            fn bitxor(self, rhs: &'b GTensor<$r>) -> Self::Output {
                GTensor::mul_mat(self, rhs)
            }
        }

        impl<'a> ops::BitXor<GTensor<$r>> for &'a GTensor<$l>
        where
            GTensor<$l>: GMulMat<$l, $r>,
        {
            type Output = GTensor<$o>;

            fn bitxor(self, rhs: GTensor<$r>) -> Self::Output {
                GTensor::mul_mat(self, &rhs)
            }
        }

        impl<'a> ops::BitXor<&'a GTensor<$r>> for GTensor<$l>
        where
            GTensor<$l>: GMulMat<$l, $r>,
        {
            type Output = GTensor<$o>;

            fn bitxor(self, rhs: &'a GTensor<$r>) -> Self::Output {
                GTensor::mul_mat(&self, rhs)
            }
        }

        impl ops::BitXor<GTensor<$r>> for GTensor<$l>
        where
            GTensor<$l>: GMulMat<$l, $r>,
        {
            type Output = GTensor<$o>;

            fn bitxor(self, rhs: GTensor<$r>) -> Self::Output {
                GTensor::mul_mat(&self, &rhs)
            }
        }
    )*};
}

mk_gmulmatinstances!(
    (2, 1, 1),
    (3, 1, 1),
    (1, 2, 1),
    (3, 2, 2),
    (1, 3, 1),
    (2, 3, 2)
);
