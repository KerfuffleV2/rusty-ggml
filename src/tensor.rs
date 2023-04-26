use std::{ops, ptr::NonNull, sync::MutexGuard};

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

impl<'a, const DIMS: usize, T: AsRef<GTensor<DIMS>>> ops::Add<T> for &'a GTensor<DIMS>
where
    Dim<DIMS>: DimValid,
{
    type Output = GTensor<DIMS>;

    fn add(self, rhs: T) -> Self::Output {
        GTensor::add(self, rhs)
    }
}

impl<const DIMS: usize, T: AsRef<GTensor<DIMS>>> ops::Add<T> for GTensor<DIMS>
where
    Dim<DIMS>: DimValid,
{
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        &self + rhs
    }
}

impl<'a, const DIMS: usize, T: AsRef<GTensor<DIMS>>> ops::Sub<T> for &'a GTensor<DIMS>
where
    Dim<DIMS>: DimValid,
{
    type Output = GTensor<DIMS>;

    fn sub(self, rhs: T) -> Self::Output {
        GTensor::sub(self, rhs)
    }
}

impl<const DIMS: usize, T: AsRef<GTensor<DIMS>>> ops::Sub<T> for GTensor<DIMS>
where
    Dim<DIMS>: DimValid,
{
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        &self - rhs
    }
}

impl<'a, const DIMS: usize, T: AsRef<GTensor<DIMS>>> ops::Div<T> for &'a GTensor<DIMS>
where
    Dim<DIMS>: DimValid,
{
    type Output = GTensor<DIMS>;

    fn div(self, rhs: T) -> Self::Output {
        GTensor::div(self, rhs)
    }
}

impl<const DIMS: usize, T: AsRef<GTensor<DIMS>>> ops::Div<T> for GTensor<DIMS>
where
    Dim<DIMS>: DimValid,
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        &self / rhs
    }
}

impl<'a, const DIMS: usize, T: AsRef<GTensor<DIMS>>> ops::Mul<T> for &'a GTensor<DIMS>
where
    Dim<DIMS>: DimValid,
{
    type Output = GTensor<DIMS>;

    fn mul(self, rhs: T) -> Self::Output {
        GTensor::mul(self, rhs)
    }
}

impl<const DIMS: usize, T: AsRef<GTensor<DIMS>>> ops::Mul<T> for GTensor<DIMS>
where
    Dim<DIMS>: DimValid,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        &self * rhs
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

impl GMulMat<1, 1> for GTensor<1> {
    type Output = GTensor<1>;

    fn mul_mat<T: AsRef<GTensor<1>>>(&self, rhs: T) -> Self::Output {
        self.new_binary(rhs, |ctx, ltptr, rtptr| unsafe {
            gg::ggml_mul_mat(ctx, ltptr, rtptr)
        })
    }
}
impl GMulMat<2, 1> for GTensor<2> {
    type Output = GTensor<1>;

    fn mul_mat<T: AsRef<GTensor<1>>>(&self, rhs: T) -> Self::Output {
        self.new_binary(rhs, |ctx, ltptr, rtptr| unsafe {
            gg::ggml_mul_mat(ctx, ltptr, rtptr)
        })
    }
}
impl GMulMat<3, 1> for GTensor<3> {
    type Output = GTensor<1>;

    fn mul_mat<T: AsRef<GTensor<1>>>(&self, rhs: T) -> Self::Output {
        self.new_binary(rhs, |ctx, ltptr, rtptr| unsafe {
            gg::ggml_mul_mat(ctx, ltptr, rtptr)
        })
    }
}

//

impl GMulMat<1, 2> for GTensor<1> {
    type Output = GTensor<1>;

    fn mul_mat<T: AsRef<GTensor<2>>>(&self, rhs: T) -> Self::Output {
        self.new_binary(rhs, |ctx, ltptr, rtptr| unsafe {
            gg::ggml_mul_mat(ctx, ltptr, rtptr)
        })
    }
}
impl GMulMat<2, 2> for GTensor<2> {
    type Output = GTensor<2>;

    fn mul_mat<T: AsRef<GTensor<2>>>(&self, rhs: T) -> Self::Output {
        self.new_binary(rhs, |ctx, ltptr, rtptr| unsafe {
            gg::ggml_mul_mat(ctx, ltptr, rtptr)
        })
    }
}
impl GMulMat<3, 2> for GTensor<3> {
    type Output = GTensor<2>;

    fn mul_mat<T: AsRef<GTensor<2>>>(&self, rhs: T) -> Self::Output {
        self.new_binary(rhs, |ctx, ltptr, rtptr| unsafe {
            gg::ggml_mul_mat(ctx, ltptr, rtptr)
        })
    }
}

//

impl GMulMat<1, 3> for GTensor<1> {
    type Output = GTensor<1>;

    fn mul_mat<T: AsRef<GTensor<3>>>(&self, rhs: T) -> Self::Output {
        self.new_binary(rhs, |ctx, ltptr, rtptr| unsafe {
            gg::ggml_mul_mat(ctx, ltptr, rtptr)
        })
    }
}
impl GMulMat<2, 3> for GTensor<2> {
    type Output = GTensor<2>;

    fn mul_mat<T: AsRef<GTensor<3>>>(&self, rhs: T) -> Self::Output {
        self.new_binary(rhs, |ctx, ltptr, rtptr| unsafe {
            gg::ggml_mul_mat(ctx, ltptr, rtptr)
        })
    }
}
impl GMulMat<3, 3> for GTensor<3> {
    type Output = GTensor<3>;

    fn mul_mat<T: AsRef<GTensor<3>>>(&self, rhs: T) -> Self::Output {
        self.new_binary(rhs, |ctx, ltptr, rtptr| unsafe {
            gg::ggml_mul_mat(ctx, ltptr, rtptr)
        })
    }
}
