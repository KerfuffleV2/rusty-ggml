use ggml_sys_bleedingedge as gg;

use super::{matmul::GMulMat, tensor::*};
use crate::dims::*;

impl<const DIMS: usize> GTensor<DIMS>
where
    Dim<DIMS>: DimValid,
{
    /// # !!!! FIXME !!!!
    /// # !!!! FIXME !!!!
    /// # !!!! FIXME !!!!
    pub fn flash_attn<
        const KDIMS: usize,
        const VDIMS: usize,
        const ODIMS: usize,
        KT: AsRef<GTensor<KDIMS>>,
        VT: AsRef<GTensor<VDIMS>>,
    >(
        &self,
        kt: KT,
        vt: VT,
        masked: bool,
    ) -> GTensor<ODIMS>
    where
        Dim<KDIMS>: DimValid,
        Dim<VDIMS>: DimValid,
        Dim<ODIMS>: DimValid,
        Self: GMulMat<KDIMS, DIMS>,
        DimPair<ODIMS, 4>: DimEq,
    {
        let kref = kt.as_ref();
        let vref = vt.as_ref();

        let ctx = self.vaqctx_bin(kref);
        assert_eq!(
            self.ctx.ptrval, vref.ctx.ptrval,
            "Cannot perform operation between tensors from different contexts!"
        );
        let (ctxp, qtptr, ktptr, vtptr) = (
            ctx.as_ptr(),
            self.tptr.as_ptr(),
            kref.tptr.as_ptr(),
            vref.tptr.as_ptr(),
        );
        unsafe {
            GTensor::<ODIMS>::new_from_ptr(
                &self.ctx,
                gg::ggml_flash_attn(ctxp, qtptr, ktptr, vtptr, masked),
            )
        }
    }

    /// # !!!! FIXME !!!!
    /// # !!!! FIXME !!!!
    /// # !!!! FIXME !!!!
    pub fn flash_ff<
        const B0DIMS: usize,
        const B1DIMS: usize,
        const C0DIMS: usize,
        const C1DIMS: usize,
        const ODIMS: usize,
        B0T: AsRef<GTensor<B0DIMS>>,
        B1T: AsRef<GTensor<B1DIMS>>,
        C0T: AsRef<GTensor<C0DIMS>>,
        C1T: AsRef<GTensor<C1DIMS>>,
    >(
        // Just an operation that takes 5 tensors. No big deal.
        &self,
        b0t: B0T,
        b1t: B1T,
        c0t: C0T,
        c1t: C1T,
    ) -> GTensor<ODIMS>
    where
        Dim<B0DIMS>: DimValid,
        Dim<B1DIMS>: DimValid,
        Dim<C0DIMS>: DimValid,
        Dim<C1DIMS>: DimValid,
        Dim<ODIMS>: DimValid,
        Self: GMulMat<B0DIMS, DIMS>,
        DimPair<ODIMS, 4>: DimEq,
    {
        let (b0ref, b1ref, c0ref, c1ref) = (b0t.as_ref(), b1t.as_ref(), c0t.as_ref(), c1t.as_ref());

        let ctx = self.vaqctx_bin(b0ref);
        assert!(
            [b1ref.ctx.ptrval, c0ref.ctx.ptrval, c1ref.ctx.ptrval]
                .into_iter()
                .all(|v| v == self.ctx.ptrval),
            "Cannot perform operation between tensors from different contexts!"
        );

        let (ctxp, atptr, b0tptr, b1tptr, c0tptr, c1tptr) = (
            ctx.as_ptr(),
            self.tptr.as_ptr(),
            b0ref.tptr.as_ptr(),
            b1ref.tptr.as_ptr(),
            c0ref.tptr.as_ptr(),
            c1ref.tptr.as_ptr(),
        );
        unsafe {
            GTensor::<ODIMS>::new_from_ptr(
                &self.ctx,
                gg::ggml_flash_ff(ctxp, atptr, b0tptr, b1tptr, c0tptr, c1tptr),
            )
        }
    }
}
