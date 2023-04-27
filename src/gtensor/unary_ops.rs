use ggml_sys_bleedingedge as gg;

use super::tensor::*;
use crate::dims::*;

macro_rules! mk_simple_uops {
  ( $([$opname:ident, $gfname:ident]),* $(,)? ) => { $(
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
        [rms_norm, ggml_rms_norm],
        [norm, ggml_norm],
        [silu, ggml_silu],
        [sqr, ggml_sqr],
        [sqrt, ggml_sqrt],
        [mean, ggml_mean],
        [abs, ggml_abs],
        [sgn, ggml_sgn],
        [neg, ggml_neg],
        [step, ggml_step],
        [relu, ggml_relu],
        [gelu, ggml_gelu],
        [cont, ggml_cont],
        [transpose, ggml_transpose],
        [soft_max, ggml_soft_max],
    }

    pub fn sum<const ODIMS: usize>(&self) -> GTensor<ODIMS>
    where
        Dim<ODIMS>: DimValid,
        DimPair<ODIMS, 2>: DimLt,
    {
        self.new_unary(|ctx, tptr| unsafe { gg::ggml_sum(ctx, tptr) })
    }

    pub fn diag_mask_inf(self, val: usize) -> Self {
        self.new_unary(|ctx, tptr| unsafe { gg::ggml_diag_mask_inf(ctx, tptr, val as i32) })
    }

    pub fn rope(self, n_past: usize, n_dims: usize, mode: usize) -> Self {
        self.new_unary(|ctx, tptr| unsafe {
            gg::ggml_rope(ctx, tptr, n_past as i32, n_dims as i32, mode as i32)
        })
    }

    // FIXME: Needs shape fixup?
    pub fn reshape<const ODIMS: usize>(&self, ne: [usize; ODIMS]) -> GTensor<ODIMS>
    where
        Dim<ODIMS>: DimValid,
        DimPair<ODIMS, 2>: DimGtE,
        DimPair<ODIMS, 4>: DimLt,
    {
        self.new_unary(|ctx, tptr| unsafe {
            match ODIMS {
                2 => gg::ggml_reshape_2d(ctx, tptr, ne[0] as i64, ne[1] as i64),
                3 => gg::ggml_reshape_3d(ctx, tptr, ne[0] as i64, ne[1] as i64, ne[2] as i64),
                _ => panic!("Bad reshape dimension!"),
            }
        })
    }

    // FIXME: Needs shape fixup?
    pub fn view<const ODIMS: usize>(
        &self,
        ne: [i64; ODIMS],
        offset: [usize; ODIMS],
    ) -> GTensor<ODIMS>
    where
        Dim<ODIMS>: DimValid,
        DimPair<ODIMS, 4>: DimLt,
    {
        self.new_unary(|ctx, tptr| unsafe {
            let elsize = gg::ggml_element_size(tptr);
            match ODIMS {
                1 => gg::ggml_view_1d(ctx, tptr, ne[0], elsize * offset[0]),
                2 => gg::ggml_view_2d(
                    ctx,
                    tptr,
                    ne[0],
                    ne[1],
                    elsize * offset[0],
                    elsize * offset[1],
                ),
                3 => gg::ggml_view_3d(
                    ctx,
                    tptr,
                    ne[0],
                    ne[1],
                    ne[2],
                    elsize * offset[0],
                    elsize * offset[1],
                    elsize * offset[2],
                ),
                _ => panic!("Bad view dimension!"),
            }
        })
    }
}
