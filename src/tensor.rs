use std::{ops, ptr::NonNull, sync::MutexGuard};

use num_traits::FromPrimitive;

use ggml_sys_bleedingedge as gg;

use crate::{
    context::{GgmlContext, GgmlIContext},
    dims::*,
};

#[repr(u32)]
#[derive(
    Debug,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    num_derive::FromPrimitive,
    num_derive::ToPrimitive,
)]
pub enum GgmlElementType {
    F32 = gg::ggml_type_GGML_TYPE_F32,
    F16 = gg::ggml_type_GGML_TYPE_F16,
    Q4_0 = gg::ggml_type_GGML_TYPE_Q4_0,
    Q4_1 = gg::ggml_type_GGML_TYPE_Q4_1,
    Q4_2 = gg::ggml_type_GGML_TYPE_Q4_2,
    Q4_3 = gg::ggml_type_GGML_TYPE_Q4_3,
    Q8_0 = gg::ggml_type_GGML_TYPE_Q8_0,
    I8 = gg::ggml_type_GGML_TYPE_I8,
    I16 = gg::ggml_type_GGML_TYPE_I16,
    I32 = gg::ggml_type_GGML_TYPE_I32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GgmlTensorMetadata<const DIMS: usize> {
    pub typ: GgmlElementType,
    pub op: gg::ggml_op,
    pub shape: [usize; DIMS],
    pub len_bytes: usize,
    pub len_elements: usize,
    pub element_size: usize,
}

impl<const DIMS: usize> GgmlTensorMetadata<DIMS>
where
    Dim<DIMS>: DimValid,
{
    /// # Safety
    /// Must be called with context mutex held.
    pub(crate) fn from_ptr(nnp: NonNull<gg::ggml_tensor>) -> Self {
        let (op, typ, shape) = {
            let tr = unsafe { nnp.as_ref() };
            assert_eq!(DIMS, tr.n_dims as usize, "Unexpected number of dimensions!");
            let mut shp = [0; DIMS];
            shp.iter_mut()
                .zip(tr.ne[0..DIMS].iter())
                .for_each(|(d, s)| *d = *s as usize);
            (tr.op, tr.type_, shp)
        };
        unsafe {
            Self {
                typ: GgmlElementType::from_u32(typ).expect("Bad type!"),
                op,
                shape,
                len_bytes: gg::ggml_nbytes(nnp.as_ptr()),
                len_elements: gg::ggml_nelements(nnp.as_ptr()) as usize,
                element_size: gg::ggml_element_size(nnp.as_ptr()),
            }
        }
    }
}

#[derive(Clone)]
// TODO: Don't panic when something goes wrong, instead
// set state in tensor and context to indicate we're dead and
// just allow other operations (except actually creating/runnning
// the graph).
pub struct GgmlTensor<const DIMS: usize> {
    pub(crate) ctx: GgmlContext,
    pub(crate) md: GgmlTensorMetadata<DIMS>,
    pub(crate) tptr: NonNull<gg::ggml_tensor>,
}

impl<const DIMS: usize> PartialEq for GgmlTensor<DIMS> {
    fn eq(&self, other: &Self) -> bool {
        self.tptr == other.tptr
    }
}

impl<const DIMS: usize> AsRef<GgmlTensor<DIMS>> for GgmlTensor<DIMS>
where
    Dim<DIMS>: DimValid,
{
    fn as_ref(&self) -> &GgmlTensor<DIMS> {
        self
    }
}

impl<const DIMS: usize> GgmlTensor<DIMS>
where
    Dim<DIMS>: DimValid,
{
    /// In bytes
    pub fn len(&self) -> usize {
        let _ctx = self.ctx.ictx.lock().expect("Failed to get context mutex");
        unsafe { gg::ggml_nbytes(self.tptr.as_ptr()) }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn elements(&self) -> usize {
        let _ctx = self.ctx.ictx.lock().expect("Failed to get context mutex");
        unsafe { gg::ggml_nelements(self.tptr.as_ptr()) as usize }
    }

    /// In bytes
    pub fn element_size(&self) -> usize {
        let _ctx = self.ctx.ictx.lock().expect("Failed to get context mutex");
        unsafe { gg::ggml_element_size(self.tptr.as_ptr()) }
    }

    pub fn metadata(&self) -> GgmlTensorMetadata<DIMS> {
        let _ctx = self.ctx.ictx.lock().expect("Failed to get context mutex");
        self.md.clone()
    }

    /// # Safety
    /// Yeah right.
    pub unsafe fn with_data<F>(&mut self, fun: F)
    where
        F: FnOnce(&mut [u8]),
    {
        let len = self.len();
        let _ctx = self.ctx.ictx.lock().expect("Failed to get context mutex");
        fun(std::slice::from_raw_parts_mut(
            self.tptr.as_ref().data as *mut u8,
            len,
        ))
    }

    //
    // Internal functions
    //

    /// # Safety
    /// Must be called with context mutex held.
    pub(crate) unsafe fn new_from_ptr(ctx: &GgmlContext, p: *mut gg::ggml_tensor) -> Self {
        let tptr = NonNull::new(p).expect("Got null pointer");
        Self {
            ctx: ctx.clone(),
            md: GgmlTensorMetadata::from_ptr(tptr),
            tptr,
        }
    }

    fn new_unary<const ODIMS: usize, F>(&self, fun: F) -> GgmlTensor<ODIMS>
    where
        Dim<ODIMS>: DimValid,
        F: FnOnce(*mut gg::ggml_context, *mut gg::ggml_tensor) -> *mut gg::ggml_tensor,
    {
        let ctx = self.ctx.ictx.lock().expect("Failed to get context mutex");
        let (ctxp, tptr) = (ctx.as_ptr(), self.tptr.as_ptr());
        unsafe { GgmlTensor::<ODIMS>::new_from_ptr(&self.ctx, fun(ctxp, tptr)) }
    }

    // RHS dims enforced elsewhere if necessary.
    fn new_binary<const RDIMS: usize, const ODIMS: usize, F, T>(
        &self,
        rhs: T,
        fun: F,
    ) -> GgmlTensor<ODIMS>
    where
        Dim<RDIMS>: DimValid,
        Dim<ODIMS>: DimValid,
        F: FnOnce(
            *mut gg::ggml_context,
            *mut gg::ggml_tensor,
            *mut gg::ggml_tensor,
        ) -> *mut gg::ggml_tensor,
        T: AsRef<GgmlTensor<RDIMS>>,
    {
        let rhs = rhs.as_ref();
        let ctx = self.vaqctx_bin(rhs);
        let (ctxp, ltptr, rtptr) = (ctx.as_ptr(), self.tptr.as_ptr(), rhs.tptr.as_ptr());
        unsafe { GgmlTensor::<ODIMS>::new_from_ptr(&self.ctx, fun(ctxp, ltptr, rtptr)) }
    }

    fn vaqctx_bin<const X: usize>(&self, other: &GgmlTensor<X>) -> MutexGuard<GgmlIContext> {
        assert_eq!(
            self.ctx.ptrval, other.ctx.ptrval,
            "Cannot perform operation between tensors from different contexts!"
        );
        self.ctx.ictx.lock().expect("Failed to get context mutex")
    }

    //
    // Binary ops
    //

    pub fn add<T: AsRef<GgmlTensor<DIMS>>>(&self, rhs: T) -> GgmlTensor<DIMS> {
        self.new_binary(rhs, |ctx, ltptr, rtptr| unsafe {
            gg::ggml_add(ctx, ltptr, rtptr)
        })
    }

    pub fn sub<T: AsRef<GgmlTensor<DIMS>>>(&self, rhs: T) -> Self {
        self.new_binary(rhs, |ctx, ltptr, rtptr| unsafe {
            gg::ggml_sub(ctx, ltptr, rtptr)
        })
    }

    pub fn mul<T: AsRef<GgmlTensor<DIMS>>>(&self, rhs: T) -> Self {
        self.new_binary(rhs, |ctx, ltptr, rtptr| unsafe {
            gg::ggml_mul(ctx, ltptr, rtptr)
        })
    }

    pub fn div<T: AsRef<GgmlTensor<DIMS>>>(&self, rhs: T) -> Self {
        self.new_binary(rhs, |ctx, ltptr, rtptr| unsafe {
            gg::ggml_div(ctx, ltptr, rtptr)
        })
    }

    pub fn map_binary<T: AsRef<GgmlTensor<DIMS>>>(
        &self,
        rhs: T,
        fun: gg::ggml_binary_op_f32_t,
    ) -> Self {
        self.new_binary(rhs, |ctx, ltptr, rtptr| unsafe {
            gg::ggml_map_binary_f32(ctx, ltptr, rtptr, fun)
        })
    }

    pub fn scale<const RDIMS: usize, T: AsRef<GgmlTensor<RDIMS>>>(&self, rhs: T) -> Self
    where
        Dim<RDIMS>: DimValid,
        DimPair<1, RDIMS>: DimEq,
    {
        self.new_binary(rhs, |ctx, ltptr, rtptr| unsafe {
            gg::ggml_scale(ctx, ltptr, rtptr)
        })
    }

    pub fn repeat<const RDIMS: usize, T: AsRef<GgmlTensor<RDIMS>>>(
        &self,
        rhs: T,
    ) -> GgmlTensor<RDIMS>
    where
        Dim<RDIMS>: DimValid,
    {
        self.new_binary(rhs, |ctx, ltptr, rtptr| unsafe {
            gg::ggml_repeat(ctx, ltptr, rtptr)
        })
    }

    // FIXME: Try to find a way to express MIN<DIMS,RDIMS> with const generics.
    pub fn mul_mat_smallrhs<const RDIMS: usize, T: AsRef<GgmlTensor<RDIMS>>>(
        &self,
        rhs: T,
    ) -> GgmlTensor<RDIMS>
    where
        Dim<RDIMS>: DimValid,
        DimPair<RDIMS, DIMS>: DimLt,
    {
        self.new_binary(rhs, |ctx, ltptr, rtptr| unsafe {
            gg::ggml_mul_mat(ctx, ltptr, rtptr)
        })
    }

    pub fn mul_mat<const RDIMS: usize, T: AsRef<GgmlTensor<RDIMS>>>(
        &self,
        rhs: T,
    ) -> GgmlTensor<DIMS>
    where
        Dim<RDIMS>: DimValid,
        DimPair<RDIMS, DIMS>: DimGtE,
    {
        self.new_binary(rhs, |ctx, ltptr, rtptr| unsafe {
            gg::ggml_mul_mat(ctx, ltptr, rtptr)
        })
    }

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

    pub fn reshape_with<const RDIMS: usize, T: AsRef<GgmlTensor<RDIMS>>>(
        &self,
        rhs: T,
    ) -> GgmlTensor<RDIMS>
    where
        Dim<RDIMS>: DimValid,
    {
        self.new_binary(rhs, |ctx, ltptr, rtptr| unsafe {
            gg::ggml_reshape(ctx, ltptr, rtptr)
        })
    }

    pub fn get_rows<const RDIMS: usize, const ODIMS: usize, T: AsRef<GgmlTensor<RDIMS>>>(
        &self,
        rhs: T,
    ) -> GgmlTensor<ODIMS>
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

    pub fn copy_from<T: AsRef<GgmlTensor<DIMS>>>(self, rhs: T) -> Self {
        self.new_binary(rhs, |ctx, ltptr, rtptr| unsafe {
            gg::ggml_cpy(ctx, ltptr, rtptr)
        })
    }

    //
    // Unary ops
    //

    pub fn rms_norm(&self) -> Self {
        self.new_unary(|ctx, tptr| unsafe { gg::ggml_rms_norm(ctx, tptr) })
    }

    pub fn silu(&self) -> Self {
        self.new_unary(|ctx, tptr| unsafe { gg::ggml_silu(ctx, tptr) })
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

    pub fn map_unary(&self, fun: gg::ggml_unary_op_f32_t) -> Self {
        self.new_unary(|ctx, tptr| unsafe { gg::ggml_map_unary_f32(ctx, tptr, fun) })
    }

    pub fn reshape<const ODIMS: usize>(&self, ne: [usize; ODIMS]) -> GgmlTensor<ODIMS>
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

    pub fn view<const ODIMS: usize>(
        &self,
        ne: [i64; ODIMS],
        offset: [usize; ODIMS],
    ) -> GgmlTensor<ODIMS>
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

impl<'a, const DIMS: usize, T: AsRef<GgmlTensor<DIMS>>> ops::Add<T> for &'a GgmlTensor<DIMS>
where
    Dim<DIMS>: DimValid,
{
    type Output = GgmlTensor<DIMS>;

    fn add(self, rhs: T) -> Self::Output {
        GgmlTensor::add(self, rhs)
    }
}

impl<const DIMS: usize, T: AsRef<GgmlTensor<DIMS>>> ops::Add<T> for GgmlTensor<DIMS>
where
    Dim<DIMS>: DimValid,
{
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        &self + rhs
    }
}

impl<'a, const DIMS: usize, T: AsRef<GgmlTensor<DIMS>>> ops::Sub<T> for &'a GgmlTensor<DIMS>
where
    Dim<DIMS>: DimValid,
{
    type Output = GgmlTensor<DIMS>;

    fn sub(self, rhs: T) -> Self::Output {
        GgmlTensor::sub(self, rhs)
    }
}

impl<const DIMS: usize, T: AsRef<GgmlTensor<DIMS>>> ops::Sub<T> for GgmlTensor<DIMS>
where
    Dim<DIMS>: DimValid,
{
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        &self - rhs
    }
}

impl<'a, const DIMS: usize, T: AsRef<GgmlTensor<DIMS>>> ops::Div<T> for &'a GgmlTensor<DIMS>
where
    Dim<DIMS>: DimValid,
{
    type Output = GgmlTensor<DIMS>;

    fn div(self, rhs: T) -> Self::Output {
        GgmlTensor::div(self, rhs)
    }
}

impl<const DIMS: usize, T: AsRef<GgmlTensor<DIMS>>> ops::Div<T> for GgmlTensor<DIMS>
where
    Dim<DIMS>: DimValid,
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        &self / rhs
    }
}

impl<'a, const DIMS: usize, T: AsRef<GgmlTensor<DIMS>>> ops::Mul<T> for &'a GgmlTensor<DIMS>
where
    Dim<DIMS>: DimValid,
{
    type Output = GgmlTensor<DIMS>;

    fn mul(self, rhs: T) -> Self::Output {
        GgmlTensor::mul(self, rhs)
    }
}

impl<const DIMS: usize, T: AsRef<GgmlTensor<DIMS>>> ops::Mul<T> for GgmlTensor<DIMS>
where
    Dim<DIMS>: DimValid,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        &self * rhs
    }
}
