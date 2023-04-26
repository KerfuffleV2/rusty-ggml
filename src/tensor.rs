use std::{ops, ptr::NonNull, sync::MutexGuard};

use gg::ggml_flash_attn;
use num_traits::FromPrimitive;

use ggml_sys_bleedingedge as gg;

use crate::{
    context::{GContext, IContext},
    dims::*,
    util::*,
};

#[derive(Debug, Clone, PartialEq)]
pub struct GTensorMetadata<const DIMS: usize> {
    pub typ: GType,
    pub op: gg::ggml_op,
    pub shape: [usize; DIMS],
    pub len_bytes: usize,
    pub len_elements: usize,
    pub element_size: usize,
}

impl<const DIMS: usize> GTensorMetadata<DIMS>
where
    Dim<DIMS>: DimValid,
{
    /// # Safety
    /// Must be called with context mutex held.
    pub(crate) fn from_ptr(tp: NonNull<gg::ggml_tensor>) -> Self {
        let (tr, tp) = (unsafe { tp.as_ref() }, tp.as_ptr());
        let (op, typ, shape) = {
            assert_eq!(DIMS, tr.n_dims as usize, "Unexpected number of dimensions!");
            let mut shp = [0; DIMS];
            shp.iter_mut()
                .zip(tr.ne[0..DIMS].iter())
                .for_each(|(d, s)| *d = *s as usize);
            (tr.op, tr.type_, shp)
        };
        let typ = GType::from_u32(typ).expect("Bad type!");
        unsafe {
            Self {
                typ,
                op,
                shape,
                len_bytes: gg::ggml_nbytes(tp),
                len_elements: gg::ggml_nelements(tp) as usize,
                element_size: typ.element_size(),
            }
        }
    }
}

#[derive(Clone)]
// TODO: Don't panic when something goes wrong, instead
// set state in tensor and context to indicate we're dead and
// just allow other operations (except actually creating/runnning
// the graph).
pub struct GTensor<const DIMS: usize> {
    pub(crate) ctx: GContext,
    pub(crate) md: GTensorMetadata<DIMS>,
    pub(crate) tptr: NonNull<gg::ggml_tensor>,
}

impl<const DIMS: usize> PartialEq for GTensor<DIMS> {
    fn eq(&self, other: &Self) -> bool {
        self.tptr == other.tptr
    }
}

impl<const DIMS: usize> AsRef<GTensor<DIMS>> for GTensor<DIMS>
where
    Dim<DIMS>: DimValid,
{
    fn as_ref(&self) -> &GTensor<DIMS> {
        self
    }
}

impl<const DIMS: usize> GTensor<DIMS>
where
    Dim<DIMS>: DimValid,
{
    pub const DIMS: usize = DIMS;

    pub fn dims(&self) -> usize {
        DIMS
    }

    /// In bytes
    pub fn len(&self) -> usize {
        self.md.len_bytes
    }

    pub fn is_empty(&self) -> bool {
        self.md.len_bytes == 0
    }

    pub fn elements(&self) -> usize {
        self.md.len_elements
    }

    /// In bytes
    pub fn element_size(&self) -> usize {
        self.md.element_size
    }

    pub fn shape(&self) -> [usize; DIMS] {
        self.md.shape
    }

    pub fn ggml_op(&self) -> gg::ggml_op {
        self.md.op
    }

    pub fn element_type(&self) -> GType {
        self.md.typ
    }

    pub fn metadata(&self) -> GTensorMetadata<DIMS> {
        self.md.clone()
    }

    pub fn get_ne(&self) -> [i64; 4] {
        let _ctx = self.ctx.ictx.lock().expect("Failed to get context mutex");
        unsafe { self.tptr.as_ref().ne }
    }

    /// # Safety
    /// Yeah right.
    pub unsafe fn with_data_mut<F, O>(&mut self, fun: F) -> O
    where
        F: FnOnce(&mut [u8]) -> O,
    {
        let _ctx = self.ctx.ictx.lock().expect("Failed to get context mutex");
        fun(std::slice::from_raw_parts_mut(
            self.tptr.as_ref().data as *mut u8,
            self.md.len_bytes,
        ))
    }

    /// # Safety
    /// Yeah right.
    pub unsafe fn with_data<F, O>(&self, fun: F) -> O
    where
        F: FnOnce(&[u8]) -> O,
    {
        let _ctx = self.ctx.ictx.lock().expect("Failed to get context mutex");
        fun(std::slice::from_raw_parts(
            self.tptr.as_ref().data as *const u8,
            self.md.len_bytes,
        ))
    }

    //
    // Internal functions
    //

    /// # Safety
    /// Must be called with context mutex held.
    pub(crate) unsafe fn new_from_ptr(ctx: &GContext, p: *mut gg::ggml_tensor) -> Self {
        let tptr = NonNull::new(p).expect("Got null pointer");
        Self {
            ctx: ctx.clone(),
            md: GTensorMetadata::from_ptr(tptr),
            tptr,
        }
    }

    fn with_tensor<T, F>(&self, fun: F) -> T
    where
        F: FnOnce(*mut gg::ggml_context, *mut gg::ggml_tensor) -> T,
    {
        let ctx = self.ctx.ictx.lock().expect("Failed to get context mutex");
        fun(ctx.as_ptr(), self.tptr.as_ptr())
    }

    fn new_unary<const ODIMS: usize, F>(&self, fun: F) -> GTensor<ODIMS>
    where
        Dim<ODIMS>: DimValid,
        F: FnOnce(*mut gg::ggml_context, *mut gg::ggml_tensor) -> *mut gg::ggml_tensor,
    {
        let ctx = self.ctx.ictx.lock().expect("Failed to get context mutex");
        let (ctxp, tptr) = (ctx.as_ptr(), self.tptr.as_ptr());
        unsafe { GTensor::<ODIMS>::new_from_ptr(&self.ctx, fun(ctxp, tptr)) }
    }

    // RHS dims enforced elsewhere if necessary.
    fn new_binary<const RDIMS: usize, const ODIMS: usize, F, T>(
        &self,
        rhs: T,
        fun: F,
    ) -> GTensor<ODIMS>
    where
        Dim<RDIMS>: DimValid,
        Dim<ODIMS>: DimValid,
        F: FnOnce(
            *mut gg::ggml_context,
            *mut gg::ggml_tensor,
            *mut gg::ggml_tensor,
        ) -> *mut gg::ggml_tensor,
        T: AsRef<GTensor<RDIMS>>,
    {
        let rhs = rhs.as_ref();
        let ctx = self.vaqctx_bin(rhs);
        let (ctxp, ltptr, rtptr) = (ctx.as_ptr(), self.tptr.as_ptr(), rhs.tptr.as_ptr());
        unsafe { GTensor::<ODIMS>::new_from_ptr(&self.ctx, fun(ctxp, ltptr, rtptr)) }
    }

    fn vaqctx_bin<const X: usize>(&self, other: &GTensor<X>) -> MutexGuard<IContext> {
        assert_eq!(
            self.ctx.ptrval, other.ctx.ptrval,
            "Cannot perform operation between tensors from different contexts!"
        );
        self.ctx.ictx.lock().expect("Failed to get context mutex")
    }

    //
    // Binary ops
    //

    pub fn add<T: AsRef<GTensor<DIMS>>>(&self, rhs: T) -> GTensor<DIMS> {
        self.new_binary(rhs, |ctx, ltptr, rtptr| unsafe {
            gg::ggml_add(ctx, ltptr, rtptr)
        })
    }

    pub fn sub<T: AsRef<GTensor<DIMS>>>(&self, rhs: T) -> Self {
        self.new_binary(rhs, |ctx, ltptr, rtptr| unsafe {
            gg::ggml_sub(ctx, ltptr, rtptr)
        })
    }

    pub fn mul<T: AsRef<GTensor<DIMS>>>(&self, rhs: T) -> Self {
        self.new_binary(rhs, |ctx, ltptr, rtptr| unsafe {
            gg::ggml_mul(ctx, ltptr, rtptr)
        })
    }

    pub fn div<T: AsRef<GTensor<DIMS>>>(&self, rhs: T) -> Self {
        self.new_binary(rhs, |ctx, ltptr, rtptr| unsafe {
            gg::ggml_div(ctx, ltptr, rtptr)
        })
    }

    pub fn map_binary<T: AsRef<GTensor<DIMS>>>(
        &self,
        rhs: T,
        fun: unsafe extern "C" fn(
            arg1: ::std::os::raw::c_int,
            arg2: *mut f32,
            arg3: *const f32,
            arg4: *const f32,
        ),
    ) -> Self {
        self.new_binary(rhs, |ctx, ltptr, rtptr| unsafe {
            gg::ggml_map_binary_f32(ctx, ltptr, rtptr, Some(fun))
        })
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

    pub fn copy_from<T: AsRef<GTensor<DIMS>>>(&mut self, rhs: T) {
        let nt = self.new_binary(rhs, |ctx, ltptr, rtptr| unsafe {
            gg::ggml_cpy(ctx, rtptr, ltptr)
        });

        *self = nt;
    }

    pub fn fill_zero(&mut self) {
        self.with_tensor(|_ctx, tptr| unsafe {
            let _ = gg::ggml_set_zero(tptr);
        })
    }

    pub fn fill_i32(&mut self, val: i32) {
        self.with_tensor(|_ctx, tptr| unsafe {
            let _ = gg::ggml_set_i32(tptr, val);
        })
    }

    pub fn fill_f32(&mut self, val: f32) {
        self.with_tensor(|_ctx, tptr| unsafe {
            let _ = gg::ggml_set_f32(tptr, val);
        })
    }

    pub fn get_f32_1d(&self, index: usize) -> f32 {
        self.with_tensor(|_ctx, tptr| unsafe { gg::ggml_get_f32_1d(tptr, index as i32) })
    }

    pub fn get_i32_1d(&self, index: usize) -> i32 {
        self.with_tensor(|_ctx, tptr| unsafe { gg::ggml_get_i32_1d(tptr, index as i32) })
    }

    pub fn set_f32_1d(&mut self, index: usize, val: f32) {
        self.with_tensor(|_ctx, tptr| unsafe { gg::ggml_set_f32_1d(tptr, index as i32, val) })
    }

    pub fn set_i32_1d(&mut self, index: usize, val: i32) {
        self.with_tensor(|_ctx, tptr| unsafe { gg::ggml_set_i32_1d(tptr, index as i32, val) })
    }

    /// # Safety
    /// Fills a tensor with raw data. It's your responsibility to make sure the format is correct.
    pub unsafe fn populate_raw<S: AsRef<[u8]>>(&mut self, data: S) {
        let data = data.as_ref();
        assert_eq!(
            self.len(),
            data.len(),
            "Bad incoming length for populate_raw"
        );
        self.with_tensor(|_ctx, tptr| {
            let tref = tptr.as_ref().unwrap();
            (tref.data as *mut u8).copy_from_nonoverlapping(data.as_ptr(), data.len())
        })
    }

    // FIXME: More generic versions of these functions.
    pub fn populate_f32<S: AsRef<[f32]>>(&mut self, data: S) {
        let data = data.as_ref();
        assert_eq!(self.md.typ, GType::F32);
        assert_eq!(
            self.elements(),
            data.len(),
            "Bad incoming length for populate_f32"
        );
        unsafe {
            self.with_tensor(|_ctx, tptr| {
                let tref = tptr.as_ref().unwrap();
                (tref.data as *mut f32).copy_from_nonoverlapping(data.as_ptr(), data.len())
            })
        }
    }

    // FIXME: More generic versions of these functions.
    pub fn copy_to_slice_f32<S: AsMut<[f32]>>(&self, mut dest: S) {
        let dest = dest.as_mut();
        let elements = self.elements();
        assert_eq!(self.md.typ, GType::F32);
        assert_eq!(
            elements,
            dest.len(),
            "Destination slice doesnt't have enough space in copy_to_slice_f32"
        );
        unsafe {
            self.with_tensor(|_ctx, tptr| {
                let ts =
                    std::slice::from_raw_parts(tptr.as_ref().unwrap().data as *const f32, elements);
                dest.copy_from_slice(ts);
            })
        }
    }

    //
    // Unary ops
    //

    pub fn rms_norm(&self) -> Self {
        self.new_unary(|ctx, tptr| unsafe { gg::ggml_rms_norm(ctx, tptr) })
    }

    pub fn norm(&self) -> Self {
        self.new_unary(|ctx, tptr| unsafe { gg::ggml_norm(ctx, tptr) })
    }

    pub fn silu(&self) -> Self {
        self.new_unary(|ctx, tptr| unsafe { gg::ggml_silu(ctx, tptr) })
    }

    pub fn sqr(&self) -> Self {
        self.new_unary(|ctx, tptr| unsafe { gg::ggml_sqr(ctx, tptr) })
    }

    pub fn sqrt(&self) -> Self {
        self.new_unary(|ctx, tptr| unsafe { gg::ggml_sqrt(ctx, tptr) })
    }

    pub fn sum<const ODIMS: usize>(&self) -> GTensor<ODIMS>
    where
        Dim<ODIMS>: DimValid,
        DimPair<ODIMS, 2>: DimLt,
    {
        self.new_unary(|ctx, tptr| unsafe { gg::ggml_sum(ctx, tptr) })
    }

    pub fn mean(&self) -> Self {
        self.new_unary(|ctx, tptr| unsafe { gg::ggml_mean(ctx, tptr) })
    }

    pub fn abs(&self) -> Self {
        self.new_unary(|ctx, tptr| unsafe { gg::ggml_abs(ctx, tptr) })
    }

    pub fn sgn(&self) -> Self {
        self.new_unary(|ctx, tptr| unsafe { gg::ggml_sgn(ctx, tptr) })
    }

    pub fn neg(&self) -> Self {
        self.new_unary(|ctx, tptr| unsafe { gg::ggml_neg(ctx, tptr) })
    }

    pub fn step(&self) -> Self {
        self.new_unary(|ctx, tptr| unsafe { gg::ggml_step(ctx, tptr) })
    }

    pub fn relu(&self) -> Self {
        self.new_unary(|ctx, tptr| unsafe { gg::ggml_relu(ctx, tptr) })
    }

    pub fn gelu(&self) -> Self {
        self.new_unary(|ctx, tptr| unsafe { gg::ggml_gelu(ctx, tptr) })
    }

    pub fn cont(&self) -> Self {
        self.new_unary(|ctx, tptr| unsafe { gg::ggml_cont(ctx, tptr) })
    }

    pub fn transpose(&self) -> Self {
        self.new_unary(|ctx, tptr| unsafe { gg::ggml_transpose(ctx, tptr) })
    }

    pub fn diag_mask_inf(self, val: usize) -> Self {
        self.new_unary(|ctx, tptr| unsafe { gg::ggml_diag_mask_inf(ctx, tptr, val as i32) })
    }

    pub fn soft_max(self) -> Self {
        self.new_unary(|ctx, tptr| unsafe { gg::ggml_soft_max(ctx, tptr) })
    }

    pub fn rope(self, n_past: usize, n_dims: usize, mode: usize) -> Self {
        self.new_unary(|ctx, tptr| unsafe {
            gg::ggml_rope(ctx, tptr, n_past as i32, n_dims as i32, mode as i32)
        })
    }

    pub fn map_unary(
        &self,
        fun: unsafe extern "C" fn(arg1: ::std::os::raw::c_int, arg2: *mut f32, arg3: *const f32),
    ) -> Self {
        self.new_unary(|ctx, tptr| unsafe { gg::ggml_map_unary_f32(ctx, tptr, Some(fun)) })
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

    //
    // Other ops
    //

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
                ggml_flash_attn(ctxp, qtptr, ktptr, vtptr, masked),
            )
        }
    }

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

mk_gopinstances!((Add, add), (Sub, sub), (Mul, mul), (Div, div));

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
