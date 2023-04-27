use std::ops;

use ggml_sys_bleedingedge as gg;

use super::tensor::*;
use crate::dims::*;

macro_rules! mk_simple_bops {
  ( $([$opname:ident, $gfname:ident]),* $(,)? ) => { $(
    pub fn $opname<T: AsRef<GTensor<DIMS>>>(&self, rhs: T) -> Self {
        self.new_binary(rhs, |ctx, ltptr, rtptr| unsafe {
            gg::$gfname(ctx, ltptr, rtptr)
        })
    }
  )* }
}

macro_rules! mk_gopinstances {
    ( $( ($trait:ident, $fun:ident) ),+ ) => { $(
        impl<'a, const DIMS: usize, T: AsRef<GTensor<DIMS>>> ops::$trait<T> for &'a GTensor<DIMS>
        where
            Dim<DIMS>: DimValid,
        {
            type Output = GTensor<DIMS>;

            fn $fun(self, rhs: T) -> Self::Output {
                GTensor::$fun(self, rhs)
            }
        }

        impl<const DIMS: usize, T: AsRef<GTensor<DIMS>>> ops::$trait<T> for GTensor<DIMS>
        where
            Dim<DIMS>: DimValid,
        {
            type Output = GTensor<DIMS>;

            fn $fun(self, rhs: T) -> Self::Output {
                GTensor::$fun(&self, rhs)
            }
        }
    )*};
}

impl<const DIMS: usize> GTensor<DIMS>
where
    Dim<DIMS>: DimValid,
{
    mk_simple_bops! {
      [add,ggml_add],
      [sub,ggml_sub],
      [mul,ggml_mul],
      [div,ggml_div],
    }

    pub fn scale<const RDIMS: usize, T: AsRef<GTensor<RDIMS>>>(&self, rhs: T) -> Self
    where
        Dim<RDIMS>: DimValid,
        DimPair<1, RDIMS>: DimEq,
    {
        self.new_binary(rhs, |ctx, ltptr, rtptr| unsafe {
            gg::ggml_scale(ctx, ltptr, rtptr)
        })
    }

    pub fn repeat<const RDIMS: usize, T: AsRef<GTensor<RDIMS>>>(&self, rhs: T) -> GTensor<RDIMS>
    where
        Dim<RDIMS>: DimValid,
    {
        self.new_binary(rhs, |ctx, ltptr, rtptr| unsafe {
            gg::ggml_repeat(ctx, ltptr, rtptr)
        })
    }

    pub fn conv_1d<const RDIMS: usize, const ODIMS: usize, T: AsRef<GTensor<RDIMS>>>(
        &self,
        rhs: T,
        is_2s: bool,
    ) -> Self
    where
        Dim<RDIMS>: DimValid,
        Dim<ODIMS>: DimValid,
        DimPair<DIMS, 2>: DimGtE,
        DimPair<DIMS, 4>: DimLt,
        DimPair<RDIMS, 2>: DimGtE,
        DimPair<ODIMS, 2>: DimEq,
    {
        self.new_binary(rhs, |ctx, ltptr, rtptr| unsafe {
            if is_2s {
                gg::ggml_conv_1d_2s(ctx, ltptr, rtptr)
            } else {
                gg::ggml_conv_1d_1s(ctx, ltptr, rtptr)
            }
        })
    }

    // FIXME: Needs shape fixup?
    pub fn permute(&self, axes: [usize; 4]) -> Self {
        self.new_unary(|ctx, tptr| unsafe {
            gg::ggml_permute(
                ctx,
                tptr,
                axes[0] as i32,
                axes[1] as i32,
                axes[2] as i32,
                axes[3] as i32,
            )
        })
    }

    pub fn reshape_with<const RDIMS: usize, T: AsRef<GTensor<RDIMS>>>(
        &self,
        rhs: T,
    ) -> GTensor<RDIMS>
    where
        Dim<RDIMS>: DimValid,
    {
        self.new_binary(rhs, |ctx, ltptr, rtptr| unsafe {
            gg::ggml_reshape(ctx, ltptr, rtptr)
        })
    }

    pub fn get_rows<const RDIMS: usize, const ODIMS: usize, T: AsRef<GTensor<RDIMS>>>(
        &self,
        rhs: T,
    ) -> GTensor<ODIMS>
    where
        Dim<RDIMS>: DimValid,
        Dim<ODIMS>: DimValid,
        DimPair<DIMS, 2>: DimGtE,
        DimPair<RDIMS, 2>: DimLt,
        DimPair<ODIMS, 2>: DimEq,
    {
        self.new_binary(rhs, |ctx, ltptr, rtptr| unsafe {
            gg::ggml_get_rows(ctx, ltptr, rtptr)
        })
    }
}

mk_gopinstances!((Add, add), (Sub, sub), (Mul, mul), (Div, div));
