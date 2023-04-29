use ggml_sys_bleedingedge as gg;

use super::tensor::*;
use crate::dims::*;

macro_rules! mk_simple_uops {
  ( $($(#[$attr:meta])* [$opname:ident, $gfname:ident]),* $(,)? ) => { $(
    $(#[$attr])*
    pub fn $opname(&self) -> Self {
        self.new_unary(|ctx, tptr| unsafe { gg::$gfname(ctx, tptr) })
    }
  )* }
}

impl<const DIMS: usize> GTensor<DIMS>
where
    Dim<DIMS>: DimValid,
{
    mk_simple_uops! {


        /// Elementwise square of tensor `A`.
        /// This is the same as a map where each element is
        /// multiplied by itself.
        /// Returns a new tensor.
        ///
        /// `a.sqr()`
        ///
        /// **Invariants**
        /// 1. Result will have the shape of `A`.
        ///
        /// **Example** (pseudocode):
        /// ```ignore
        /// let a = [2, 2, 2];
        /// let result = a.sqr();
        /// assert_eq!(result, [4, 4, 4]);
        /// ```
        [sqr, ggml_sqr],

        /// Elementwise square root of tensor `A`.
        /// Returns a new tensor.
        ///
        /// `a.sqrt()`
        ///
        /// **Invariants**
        /// 1. Result will have the shape of `A`.
        ///
        /// **Example** (pseudocode):
        /// ```ignore
        /// let a = [9, 9, 9];
        /// let result = a.sqrt();
        /// assert_eq!(result, [3, 3, 3]);
        /// ```
        [sqrt, ggml_sqrt],

        // [mean, ggml_mean],

        /// Elementwise `abs` of tensor `A`.
        /// Returns a new tensor.
        ///
        /// `a.abs()`
        ///
        /// **Invariants**
        /// 1. Result will have the shape of `A`.
        ///
        /// **Example** (pseudocode):
        /// ```ignore
        /// let a = [-1, -2, -3];
        /// let result = a.abs();
        /// assert_eq!(result, [1, 2, 3]);
        /// ```
        [abs, ggml_abs],

        /// Elementwise sign operation on tensor `A`.
        /// Returns a new tensor.
        ///
        /// `a.sgn()`
        ///
        /// **Invariants**
        /// 1. If an element is 0 then the result will be 0.
        /// 2. If an element is over 0 then the result will be 1.
        /// 3. If an element is less than 0 then the result will be -1.
        /// 4. Result will have the shape of `A`.
        ///
        /// **Example** (pseudocode):
        /// ```ignore
        /// let a = [-5, 0, 6];
        /// let result = a.sgn();
        /// assert_eq!(result, [-1, 0, 1]);
        /// ```
        [sgn, ggml_sgn],

        /// Elementwise negation operation on tensor `A`.
        /// In other words, it just flips the sign.
        /// Returns a new tensor.
        ///
        /// `a.neg()`
        ///
        /// **Invariants**
        /// 1. Result will have the shape of `A`.
        ///
        /// **Example** (pseudocode):
        /// ```ignore
        /// let a = [1, -1, -6, 7];
        /// let result = a.sgn();
        /// assert_eq!(result, [-1, 1, 6, -7]);
        /// ```
        [neg, ggml_neg],

        /// Perform LayerNorm operation on tensor `A`.
        /// Returns a new tensor.
        ///
        /// `a.norm()`
        ///
        /// See [this helpful explanation](https://github.com/bzhangGo/rmsnorm/blob/2e726f1a3f106bb719056422f4f9b6aca03d3ce6/README.md)
        /// for more information and comparison with the [GTensor::rms_norm] function.
        [rms_norm, ggml_rms_norm],

        /// Perform RMSNorm operation on tensor `A`.
        /// Returns a new tensor.
        ///
        /// `a.rms_norm()`
        ///
        /// See [this helpful explanation](https://github.com/bzhangGo/rmsnorm/blob/2e726f1a3f106bb719056422f4f9b6aca03d3ce6/README.md)
        /// for more information and comparison with the [GTensor::norm] function.
        [norm, ggml_norm],

        /// Elementwise step operation on tensor `A`.
        /// Returns a new tensor.
        ///
        /// `a.step()`
        ///
        /// See <https://en.wikipedia.org/wiki/Activation_function>
        ///
        /// **Invariants**
        /// 1. If an element is over 0 then the result will be 1.
        /// 2. If an element is less or equal to 0 then the result will be 0.
        /// 3. Result will have the shape of `A`.
        ///
        /// **Example** (pseudocode):
        /// ```ignore
        /// let a = [1, -1, -6, 7];
        /// let result = a.step();
        /// assert_eq!(result, [1, 0, 0, 1]);
        /// ```
        [step, ggml_step],

        /// Perform ReLU operation on tensor `A`.
        /// Returns a new tensor.
        ///
        /// `a.relu()`
        ///
        /// See <https://en.wikipedia.org/wiki/Activation_function>
        ///
        /// **Invariants**
        /// 1. If an element is over 0 then the element passes through unchanged.
        /// 2. If an element is less or equal to 0 then the result will be 0.
        /// 3. Result will have the shape of `A`.
        ///
        /// **Example** (pseudocode):
        /// ```ignore
        /// let a = [1, -1, -6, 7];
        /// let result = a.relu();
        /// assert_eq!(result, [1, 0, 0, 7]);
        [relu, ggml_relu],

        /// Perform GELU (AKA "Gaussian Error Linear Unit")
        /// operation on tensor `A`.
        /// Returns a new tensor.
        ///
        /// `a.gelu()`
        ///
        /// See <https://en.wikipedia.org/wiki/Activation_function>
        [gelu, ggml_gelu],

        /// Perform SiLU (AKA "Sigmoid Linear Unit")
        /// operation on tensor `A`.
        /// Returns a new tensor.
        ///
        /// `a.silu()`
        ///
        /// See <https://en.wikipedia.org/wiki/Activation_function>
        [silu, ggml_silu],

        /// # !!!! FIXME !!!!
        /// # !!!! FIXME !!!!
        /// # !!!! FIXME !!!!
        [cont, ggml_cont],

        /// Create a view `A` with the first and second dimensions flipped.
        /// Returns a new tensor.
        ///
        /// **Note**: This is the same as
        /// `a.permute([1, 0, 2, 3])`.
        ///
        /// `a.transpose()`
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
        /// let result = a.transpose();
        /// assert_eq!(result, expected);
        /// ```
        [transpose, ggml_transpose],

        /// Apply the `softmax` (AKA `softargmax` or "normalized
        /// exponential function") to `A`.
        /// Returns a new tensor.
        ///
        /// `a.soft_max()`
        ///
        /// This one is a bit too complicated to explain here. See
        /// <https://en.wikipedia.org/wiki/Softmax_function>
        ///
        /// **Invariants**
        /// 1. Result will have the shape and type of `A`.
        [soft_max, ggml_soft_max],
    }

    /// Elementwise `mean` of tensor `A`.
    /// Returns a new tensor.
    ///
    /// `a.mean()`
    ///
    /// **Invariants**
    /// 1. Result will be a 1 dimensional tensor with one item.
    /// 2. The result tensor will have type [GType::F32](crate::util::GType::F32).
    ///
    /// **Example** (pseudocode):
    /// ```ignore
    /// let a = [1, 2, 3, 4];
    /// let result = a.mean();
    /// assert_eq!(result, [2.5]);
    /// ```
    pub fn mean<const ODIMS: usize>(&self) -> GTensor<ODIMS>
    where
        Dim<ODIMS>: DimValid,
        DimPair<ODIMS, 2>: DimLt,
    {
        self.new_unary(|ctx, tptr| unsafe { gg::ggml_mean(ctx, tptr) })
    }

    /// Elementwise `sum` of tensor `A`.
    /// Returns a new tensor.
    ///
    /// `a.sum()`
    ///
    /// **Invariants**
    /// 1. Result will be a 1 dimensional tensor with one item.
    /// 2. The result tensor will the same type as `A`.
    ///
    /// **Example** (pseudocode):
    /// ```ignore
    /// let a = [1, 2, 3, 4];
    /// let result = a.sum();
    /// assert_eq!(result, [10]);
    /// ```
    pub fn sum<const ODIMS: usize>(&self) -> GTensor<ODIMS>
    where
        Dim<ODIMS>: DimValid,
        DimPair<ODIMS, 2>: DimLt,
    {
        self.new_unary(|ctx, tptr| unsafe { gg::ggml_sum(ctx, tptr) })
    }

    /// # !!!! FIXME !!!!
    /// # !!!! FIXME !!!!
    /// # !!!! FIXME !!!!
    pub fn reshape<const ODIMS: usize>(&self, ne: [usize; ODIMS]) -> GTensor<ODIMS>
    where
        Dim<ODIMS>: DimValid,
        DimPair<ODIMS, 2>: DimGtE,
        DimPair<ODIMS, 4>: DimLt,
    {
        self.new_unary(|ctx, tptr| unsafe {
            match ODIMS {
                2 => gg::ggml_reshape_2d(ctx, tptr, ne[1] as i64, ne[0] as i64),
                3 => gg::ggml_reshape_3d(ctx, tptr, ne[1] as i64, ne[0] as i64, ne[2] as i64),
                _ => panic!("Bad reshape dimension!"),
            }
        })
    }

    /// # !!!! FIXME !!!!
    /// # !!!! FIXME !!!!
    /// # !!!! FIXME !!!!
    pub fn view<const ODIMS: usize>(
        &self,
        ne: [i64; ODIMS],
        offset: [usize; ODIMS],
    ) -> GTensor<ODIMS>
    where
        Dim<ODIMS>: DimValid,
        DimPair<ODIMS, 3>: DimLt,
    {
        self.new_unary(|ctx, tptr| unsafe {
            let elsize = gg::ggml_element_size(tptr);
            match ODIMS {
                1 => gg::ggml_view_1d(ctx, tptr, ne[0], elsize * offset[0]),
                2 => gg::ggml_view_2d(
                    ctx,
                    tptr,
                    ne[1],
                    ne[0],
                    elsize * offset[1],
                    elsize * offset[0],
                ),
                3 => gg::ggml_view_3d(
                    ctx,
                    tptr,
                    ne[1],
                    ne[0],
                    ne[2],
                    elsize * offset[1],
                    elsize * offset[0],
                    elsize * offset[2],
                ),
                _ => panic!("Bad view dimension!"),
            }
        })
    }

    /// # !!!! FIXME !!!!
    /// # !!!! FIXME !!!!
    /// # !!!! FIXME !!!!
    pub fn diag_mask_inf(self, val: usize) -> Self {
        self.new_unary(|ctx, tptr| unsafe { gg::ggml_diag_mask_inf(ctx, tptr, val as i32) })
    }

    /// # !!!! FIXME !!!!
    /// # !!!! FIXME !!!!
    /// # !!!! FIXME !!!!
    pub fn rope(self, n_past: usize, n_dims: usize, mode: usize) -> Self {
        self.new_unary(|ctx, tptr| unsafe {
            gg::ggml_rope(ctx, tptr, n_past as i32, n_dims as i32, mode as i32)
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{context::*, util::GType};

    macro_rules! test_uop_simple {
        (
            $fname:ident ; $meth:ident(
             $input:expr ;
             $shape:expr =>
             $expect:expr
        ) ) => {
            #[test]
            pub fn $fname() {
                let expect = $expect;
                let mut output = expect.clone();
                let ctx = GgmlContextBuilder::new().mem_size(1024 * 1024).build();
                let mut g = GgmlGraph::new(1);
                let mut t = ctx.tensor(GType::F32, $shape);
                t.populate_f32($input);
                let t2 = t.$meth();
                g.build_forward_expand(&t2);
                ctx.compute(&mut g);
                t2.copy_to_slice_f32(&mut output);
                assert_eq!(output, expect);
            }
        };
    }

    test_uop_simple!(test_sqr ; sqr(
        [2.0, 2.0, 2.0]; [3] => [4.0, 4.0, 4.0]
    ));

    test_uop_simple!(test_sqrt ; sqrt(
        [9.0, 9.0, 9.0]; [3] => [3.0, 3.0, 3.0]
    ));

    test_uop_simple!(test_abs ; abs(
        [-1.0, -2.0, -3.0]; [3] => [1.0, 2.0, 3.0]
    ));

    test_uop_simple!(test_sgn ; sgn(
        [-5.0, 0.0, 6.0]; [3] => [-1.0, 0.0, 1.0]
    ));

    test_uop_simple!(test_neg ; neg(
        [1.0, -1.0, 6.0, -7.0]; [4] => [-1.0, 1.0, -6.0, 7.0]
    ));

    test_uop_simple!(test_step ; step(
        [1.0, -1.0, -6.0, 7.0]; [4] => [1.0, 0.0, 0.0, 1.0]
    ));

    test_uop_simple!(test_relu ; relu(
        [1.0, -1.0, -6.0, 7.0]; [4] => [1.0, 0.0, 0.0, 7.0]
    ));

    test_uop_simple!(test_mean ; mean(
        [1.0, 2.0, 3.0, 4.0]; [4] => [2.5]
    ));

    test_uop_simple!(test_sum ; sum(
        [1.0, 2.0, 3.0, 4.0]; [4] => [10.0]
    ));

    // #[test]
    // pub fn test_sqr() {
    //     let ctx = GgmlContextBuilder::new().mem_size(1024 * 1024).build();
    //     let mut g = GgmlGraph::new(1);
    //     let mut t = ctx.tensor(GType::F32, [3]);
    //     t.populate_f32([2.0, 2.0, 2.0]);
    //     let t2 = t.sqr();
    //     g.build_forward_expand(&t2);
    //     ctx.compute(&mut g);
    //     let mut output = [0.0; 3];
    //     t2.copy_to_slice_f32(&mut output);
    //     assert_eq!(output, [4.0, 4.0, 4.0]);
    // }
}
