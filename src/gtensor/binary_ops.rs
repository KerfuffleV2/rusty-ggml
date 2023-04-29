use std::ops;

use ggml_sys_bleedingedge as gg;

use super::tensor::*;
use crate::dims::*;

macro_rules! mk_simple_bops {
  ( $( $(#[$attr:meta])* [$opname:ident, $gfname:ident]),* $(,)? ) => { $(
    $(#[$attr])*
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
        /// Add tensor `B` to tensor `A`.
        /// Returns a new tensor.
        ///
        /// `a.add(b)` or `a + b`
        ///
        /// **Invariants**
        /// 1. `A` and `B` must have the same shape.
        /// 2. Result will have the shape of `A`.
        ///
        /// **Example** (pseudocode):
        /// ```rust
        /// let a = [2, 2, 2];
        /// let b = [1, 1, 1];
        /// let result = a.add(b);
        /// assert_eq!(result, [3, 3, 3]);
        /// ```
        [add,ggml_add],
        /// Subtract tensor `B` from tensor `A`.
        /// Returns a new tensor.
        ///
        /// `a.sub(b)` or `a - b`
        ///
        /// **Invariants**
        /// 1. `A` and `B` must have the same shape.
        /// 2. Result will have the shape of `A`.
        ///
        /// **Example** (pseudocode):
        /// ```rust
        /// let a = [3, 3, 3];
        /// let b = [1, 1, 1];
        /// let result = a.div(b);
        /// assert_eq!(result, [2, 2, 2]);
        /// ```
        [sub,ggml_sub],

        /// Multiply tensor `A` by tensor `B`.
        /// Returns a new tensor.
        ///
        /// **Note**: This is elementwise multiplication, not matrix multiplication.
        ///
        /// `a.mul(b)` or `a * b`
        ///
        /// **Invariants**
        /// 1. `A` and `B` must have the same shape.
        /// 2. Result will have the shape of `A`.
        ///
        /// **Example** (pseudocode):
        /// ```rust
        /// let a = [3, 3, 3];
        /// let b = [2, 2, 2];
        /// let result = a.mul(b);
        /// assert_eq!(result, [6, 6, 6]);
        /// ```
        [mul,ggml_mul],

        /// Elementwise divide tensor `A` by tensor `B`.
        /// Returns a new tensor.
        ///
        /// `a.div(b)` or `a / b`
        ///
        /// **Invariants**
        /// 1. `A` and `B` must have the same shape.
        /// 2. Result will have the shape of `A`.
        ///
        /// **Example** (pseudocode):
        /// ```rust
        /// let a = [6, 6, 6];
        /// let b = [2, 2, 2];
        /// let result = a.div(b);
        /// assert_eq!(result, [3, 3, 3]);
        /// ```
        [div,ggml_div],
    }

    /// Scale tensor `A` by tensor `B`.
    /// This is basically just scalar multiplication.
    /// Returns a new tensor.
    ///
    /// `a.scale(b)`
    ///
    /// **Invariants**
    /// 1. Tensor `B` must have shape `[1]`. (AKA 1d tensor with a single item.)
    /// 2. Result will have the shape of `A`.
    ///
    /// **Example** (pseudocode):
    /// ```rust
    /// let a = [3, 3, 3];
    /// let b = [2];
    /// let result = a.scale(b);
    /// assert_eq!(result, [6, 6, 6]);
    /// ```
    pub fn scale<const RDIMS: usize, T: AsRef<GTensor<RDIMS>>>(&self, rhs: T) -> Self
    where
        Dim<RDIMS>: DimValid,
        DimPair<1, RDIMS>: DimEq,
    {
        self.new_binary(rhs, |ctx, ltptr, rtptr| unsafe {
            gg::ggml_scale(ctx, ltptr, rtptr)
        })
    }

    // FIXME: Verify this is correct.
    /// Repeat tensor `A` based on the shape of tensor `B`.
    /// The content of `B` is not used.
    /// Returns a new tensor.
    ///
    /// `a.repeat(b)`
    ///
    /// **Invariants**
    /// 1. Both `A` and `B` must be 1d or 2d tensors.
    /// 2. Neither `A` or `B` can be transposed or permuted.
    /// 3. The shape of `B` must be divisible by the shape of `A`. In other words,
    ///    `b_rows % a_rows` and `b_cols % a_cols` must both be `0`.
    /// 4. Result will have the shape of `B`.
    ///
    /// **Example** (pseudocode):
    /// ```rust
    /// let a =
    ///     [ [2, 3],
    ///       [4, 5] ];
    /// let b =
    ///     [ [1, 1, 1, 1],
    ///       [1, 1, 1, 1],
    ///       [1, 1, 1, 1],
    ///       [1, 1, 1, 1] ];
    /// let expected =
    ///     [ [2, 3, 2, 3],
    ///       [4, 5, 4, 5],
    ///       [2, 3, 2, 3],
    ///       [4, 5, 4, 5] ];
    /// let result = a.repeat(b);
    /// assert_eq!(result, expected);
    /// ```
    pub fn repeat<const RDIMS: usize, T: AsRef<GTensor<RDIMS>>>(&self, rhs: T) -> GTensor<RDIMS>
    where
        Dim<RDIMS>: DimValid,
        DimPair<DIMS, 3>: DimLt,
        DimPair<RDIMS, 3>: DimLt,
    {
        self.new_binary(rhs, |ctx, ltptr, rtptr| unsafe {
            gg::ggml_repeat(ctx, ltptr, rtptr)
        })
    }

    /// # !!!! FIXME !!!!
    /// # !!!! FIXME !!!!
    /// # !!!! FIXME !!!!
    pub fn conv_1d<
        const STRIDE: usize,
        const RDIMS: usize,
        const ODIMS: usize,
        T: AsRef<GTensor<RDIMS>>,
    >(
        &self,
        rhs: T,
    ) -> Self
    where
        Dim<RDIMS>: DimValid,
        Dim<ODIMS>: DimValid,
        DimPair<DIMS, 2>: DimGtE,
        DimPair<DIMS, 4>: DimLt,
        DimPair<RDIMS, 2>: DimGtE,
        DimPair<ODIMS, 2>: DimEq,
        DimPair<STRIDE, 1>: DimGtE,
        DimPair<STRIDE, 3>: DimLt,
    {
        self.new_binary(rhs, |ctx, ltptr, rtptr| unsafe {
            if STRIDE == 1 {
                gg::ggml_conv_1d_1s(ctx, ltptr, rtptr)
            } else {
                gg::ggml_conv_1d_2s(ctx, ltptr, rtptr)
            }
        })
    }

    // FIXME: Verify this is correct.
    /// Create a view `A` of based on the specified order of dimensions.
    /// Returns a new tensor.
    ///
    /// **Note**: Dimensions start from `0`, so `0` is the 1st dimension, `3` is the 4th.
    ///
    /// `a.permute([4, 2, 1, 1])`
    ///
    /// **Invariants**
    /// 1. The axes must be unique. `[0, 0, 1, 2]` would be invalid, for example.
    /// 2. The axes must be a valid dimension. In other words, a number between `0` and `3` inclusive.
    ///
    /// **Example** (pseudocode):
    /// ```rust
    /// let a =
    ///     [ [1, 1, 1],
    ///       [2, 2, 3],
    ///       [3, 3, 3] ];
    /// let expected =
    ///     [ [1, 2, 3],
    ///       [1, 2, 3],
    ///       [1, 2, 3] ];
    /// let result = a.permute([1, 0, 2, 3]);
    /// assert_eq!(result, expected);
    /// ```
    pub fn permute(&self, axes: [usize; 4]) -> Self {
        self.new_unary(|ctx, tptr| unsafe {
            gg::ggml_permute(
                ctx,
                tptr,
                axes[1] as i32,
                axes[0] as i32,
                axes[2] as i32,
                axes[3] as i32,
            )
        })
    }

    /// # !!!! FIXME !!!!
    /// # !!!! FIXME !!!!
    /// # !!!! FIXME !!!!
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

    /// # !!!! FIXME !!!!
    /// # !!!! FIXME !!!!
    /// # !!!! FIXME !!!!
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
