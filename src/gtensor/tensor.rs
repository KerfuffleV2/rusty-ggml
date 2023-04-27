use std::{ptr::NonNull, sync::MutexGuard};

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

    pub(crate) fn with_tensor<T, F>(&self, fun: F) -> T
    where
        F: FnOnce(*mut gg::ggml_context, *mut gg::ggml_tensor) -> T,
    {
        let ctx = self.ctx.ictx.lock().expect("Failed to get context mutex");
        fun(ctx.as_ptr(), self.tptr.as_ptr())
    }

    pub(crate) fn new_unary<const ODIMS: usize, F>(&self, fun: F) -> GTensor<ODIMS>
    where
        Dim<ODIMS>: DimValid,
        F: FnOnce(*mut gg::ggml_context, *mut gg::ggml_tensor) -> *mut gg::ggml_tensor,
    {
        let ctx = self.ctx.ictx.lock().expect("Failed to get context mutex");
        let (ctxp, tptr) = (ctx.as_ptr(), self.tptr.as_ptr());
        unsafe { GTensor::<ODIMS>::new_from_ptr(&self.ctx, fun(ctxp, tptr)) }
    }

    // RHS dims enforced elsewhere if necessary.
    pub(crate) fn new_binary<const RDIMS: usize, const ODIMS: usize, F, T>(
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

    pub(crate) fn vaqctx_bin<const X: usize>(&self, other: &GTensor<X>) -> MutexGuard<IContext> {
        assert_eq!(
            self.ctx.ptrval, other.ctx.ptrval,
            "Cannot perform operation between tensors from different contexts!"
        );
        self.ctx.ictx.lock().expect("Failed to get context mutex")
    }

    //
    // Binary ops
    //

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

    //
    // Other ops
    //
}
