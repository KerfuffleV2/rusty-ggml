use std::ops;

use ggml_sys_bleedingedge as gg;

use super::tensor::*;
use crate::{dims::*, util::GType, validation::*};

macro_rules! mk_simple_bops {
  ( $( $(#[$attr:meta])* [$opname:ident, $gfname:ident]),* $(,)? ) => { $(
    $(#[$attr])*
    pub fn $opname<T: AsRef<GTensor<DIMS>>>(&self, rhs: T) -> Self {
        self.new_binary(rhs, |ctx, ictx, ltptr, rtptr| {
            let mr = GMemoryRequest::estimate_tensor_request_ictx(
                ctx, ictx, self.md.typ, self.md.shape
            ).fit_or_die()?;
            Ok((mr, unsafe { gg::$gfname(ictx.gptr(), ltptr, rtptr) }))
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
        /// ```ignore
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
        /// ```ignore
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
        /// ```ignore
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
        /// ```ignore
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
    /// ```ignore
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
        self.new_binary(rhs, |ctx, ictx, ltptr, rtptr| {
            let mr =
                GMemoryRequest::estimate_tensor_request_ictx(ctx, ictx, self.md.typ, self.md.shape)
                    .fit_or_die()?;
            Ok((mr, unsafe { gg::ggml_scale(ictx.gptr(), ltptr, rtptr) }))
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
    /// ```ignore
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
        let rmd = rhs.as_ref().md.clone();
        self.new_binary(rhs, |ctx, ictx, ltptr, rtptr| {
            let mr =
                GMemoryRequest::estimate_tensor_request_ictx(ctx, ictx, self.md.typ, rmd.shape)
                    .fit_or_die()?;
            Ok((mr, unsafe { gg::ggml_repeat(ictx.gptr(), ltptr, rtptr) }))
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
        let rmd = rhs.as_ref().md.clone();
        self.new_binary(rhs, |ctx, ictx, ltptr, rtptr| {
            // FIXME: Double check this calculation.
            let shp = match ODIMS {
                2 => vec![self.md.ggml_ne[2] as usize, rmd.ggml_ne[1] as usize],
                3 => vec![self.md.ggml_ne[2] as usize, rmd.ggml_ne[1] as usize / 2],
                _ => Err(GTensorError::InvalidOperation)?,
            };
            let mr = GMemoryRequest::estimate_tensor_request_ictx(ctx, ictx, GType::F32, shp)
                .fit_or_die()?;
            Ok((mr, unsafe {
                if STRIDE == 1 {
                    gg::ggml_conv_1d_s1_ph(ictx.gptr(), ltptr, rtptr)
                } else {
                    gg::ggml_conv_1d_s2_ph(ictx.gptr(), ltptr, rtptr)
                }
            }))
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
    /// ```ignore
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
        self.new_unary(|ctx, ictx, tptr| {
            let mr = GMemoryRequest::estimate_tensor_request_ictx(ctx, ictx, self.md.typ, [])
                .fit_or_die()?;
            unsafe {
                Ok((
                    mr,
                    gg::ggml_permute(
                        ictx.gptr(),
                        tptr,
                        axes[1] as i32,
                        axes[0] as i32,
                        axes[2] as i32,
                        axes[3] as i32,
                    ),
                ))
            }
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
        let rmd = rhs.as_ref().md.clone();
        self.new_binary(rhs, |ctx, ictx, ltptr, rtptr| {
            let mr =
                GMemoryRequest::estimate_tensor_request_ictx(ctx, ictx, self.md.typ, rmd.shape)
                    .fit_or_die()?;
            Ok((mr, unsafe { gg::ggml_reshape(ictx.gptr(), ltptr, rtptr) }))
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
        let rmd = rhs.as_ref().md.clone();
        self.new_binary(rhs, |ctx, ictx, ltptr, rtptr| {
            let mr = GMemoryRequest::estimate_tensor_request_ictx(
                ctx,
                ictx,
                GType::F32,
                [self.md.shape[1], rmd.shape[0]],
            )
            .fit_or_die()?;
            Ok((mr, unsafe { gg::ggml_get_rows(ictx.gptr(), ltptr, rtptr) }))
        })
    }
}

mk_gopinstances!((Add, add), (Sub, sub), (Mul, mul), (Div, div));

#[cfg(test)]
mod tests {
    use crate::{context::*, gtensor::GMulMat, util::GType};
    use anyhow::Result;

    macro_rules! test_binop_simple {
        (
            $fname:ident ; $meth:ident(
             $input_a:expr ;
             $shape_a:expr ,
             $input_b:expr ;
             $shape_b:expr =>
             $expect:expr
        ) ) => {
            #[test]
            pub fn $fname() -> Result<()> {
                let expect = $expect;
                let mut output = expect.clone();
                let ctx = GContextBuilder::new().mem_size(1024 * 1024).build()?;
                // let mut ctx = GContextBuilder::new().mem_size(719).build()?;
                // let bid = ctx.register_scratch_buffer(ScratchBuffer::new(195))?;
                // ctx.set_scratch_buffer(Some(bid))?;
                let mut g = GGraph::new(1);
                let mut ta = ctx.tensor(GType::F32, $shape_a)?;
                ta.populate_f32($input_a);
                let mut tb = ctx.tensor(GType::F32, $shape_b)?;
                tb.populate_f32($input_b);
                let t2 = ta.$meth(tb);
                g.build_forward_expand(&t2)?;
                ctx.compute(&mut g)?;
                t2.copy_to_slice_f32(&mut output)?;
                assert_eq!(output, expect);
                Ok(())
            }
        };
    }

    test_binop_simple!(test_add ; add(
        [2.0, 2.0, 2.0]; [3],
        [1.0, 1.0, 1.0]; [3] => [3.0, 3.0, 3.0]
    ));

    test_binop_simple!(test_sub ; sub(
        [3.0, 3.0, 3.0]; [3],
        [1.0, 1.0, 1.0]; [3] => [2.0, 2.0, 2.0]
    ));

    test_binop_simple!(test_mul ; mul(
        [3.0, 3.0, 3.0]; [3],
        [2.0, 2.0, 2.0]; [3] => [6.0, 6.0, 6.0]
    ));

    test_binop_simple!(test_div ; div(
        [6.0, 6.0, 6.0]; [3],
        [2.0, 2.0, 2.0]; [3] => [3.0, 3.0, 3.0]
    ));

    test_binop_simple!(test_mul_mat ; mul_mat(
        [1.0, 1.0, 2.0, 2.0]; [2, 2],
        [2.0, 2.0]; [2]
        => [4.0, 8.0]
    ));

    test_binop_simple!(test_scale ; scale(
        [3.0, 3.0, 3.0]; [3],
        [2.0]; [1] => [6.0, 6.0, 6.0]
    ));

    test_binop_simple!(test_repeat ; repeat(
        [2.0, 3.0, 4.0, 5.0]; [2, 2],
        [
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
        ]; [4, 4] =>
        [
            2.0, 3.0, 2.0, 3.0,
            4.0, 5.0, 4.0, 5.0,
            2.0, 3.0, 2.0, 3.0,
            4.0, 5.0, 4.0, 5.0,
        ]
    ));
}
