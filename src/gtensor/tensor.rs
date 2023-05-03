use std::{
    ptr::NonNull,
    sync::{atomic::Ordering::SeqCst, Arc},
};

use anyhow::{ensure, Result};
use num_traits::FromPrimitive;
use thiserror::Error;

use ggml_sys_bleedingedge as gg;

use crate::{
    context::{GContext, GContextError, IContext},
    dims::*,
    util::*,
    validation::*,
};

#[derive(Debug, Error, Clone)]
pub enum GTensorError {
    #[error("Type mismatch")]
    TypeMismatch,
    #[error("Bad data length in populate - got {got}, expected {expected}")]
    BadPopulate { got: usize, expected: usize },
    #[error("Invalid tensor operation: invariants violated")]
    InvalidOperation,
    #[error("GGML tensor operation returned NULL")]
    NullPointer,
    #[error("General error: {0}")]
    General(Arc<anyhow::Error>),
}

#[derive(Debug, Clone, PartialEq)]
/// Metadata associated with a [GTensor].
pub struct GTensorMetadata<const DIMS: usize> {
    /// The type of tensor.
    pub typ: GType,

    /// The associated GGML operation if available.
    pub op: gg::ggml_op,

    /// The shape of the tensor.
    pub shape: [usize; DIMS],

    /// The length in bytes.
    pub len_bytes: usize,

    /// The number of elements.
    pub len_elements: usize,

    /// The size of an individual element.
    ///
    /// **Note**: This may not be accurate for quantized types.
    pub element_size: usize,

    pub ggml_ne: [u32; gg::GGML_MAX_DIMS as usize],
    pub ggml_nb: [u32; gg::GGML_MAX_DIMS as usize],
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
                ggml_ne: tr
                    .ne
                    .iter()
                    .map(|v| *v as u32)
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap(),
                ggml_nb: tr
                    .nb
                    .iter()
                    .map(|v| *v as u32)
                    .collect::<Vec<_>>()
                    .try_into()
                    .unwrap(),
            }
        }
    }

    pub(crate) fn new_empty() -> Self {
        Self {
            typ: GType::F32,
            op: 0,
            shape: [0; DIMS],
            len_bytes: 0,
            len_elements: 0,
            element_size: 0,
            ggml_ne: [0; 4],
            ggml_nb: [0; 4],
        }
    }

    // Note: These functions are deliberately written not to
    // use DIMS so they can still work when the const generic
    // gets type-erased.

    pub fn is_scalar(&self) -> bool {
        self.ggml_ne.iter().all(|v| *v == 1)
    }

    pub fn is_vector(&self) -> bool {
        self.ggml_ne.iter().skip(1).all(|v| *v == 1)
    }

    pub fn is_matrix(&self) -> bool {
        self.ggml_ne[2] == 1 && self.ggml_ne[3] == 1
    }

    pub fn is_quantized(&self) -> bool {
        self.typ.is_quantized()
    }

    pub fn can_mul_mat_with<const RDIMS: usize>(&self, other: &GTensorMetadata<RDIMS>) -> bool {
        self.ggml_ne
            .iter()
            .zip(other.ggml_ne.iter())
            .enumerate()
            .all(|(idx, (lels, rels))| idx == 1 || lels == rels)
    }

    pub fn can_repeat_with<const RDIMS: usize>(&self, other: &GTensorMetadata<RDIMS>) -> bool
    where
        Dim<RDIMS>: DimValid,
    {
        DIMS < 3
            && RDIMS < 3
            && !self.is_transposed()
            && !other.is_transposed()
            && self
                .ggml_ne
                .iter()
                .zip(other.ggml_ne.iter())
                .all(|(lels, rels)| lels > &0 && rels % lels == 0)
    }

    pub fn is_permuted(&self) -> bool {
        self.ggml_nb[0] > self.ggml_nb[1]
            || self.ggml_nb[1] > self.ggml_nb[2]
            || self.ggml_nb[2] > self.ggml_nb[3]
    }

    pub fn is_transposed(&self) -> bool {
        self.ggml_nb[0] > self.ggml_nb[1]
    }

    pub fn is_contiguous(&self) -> bool {
        let elsize = self.typ.element_size() as u32;
        let bsize = self.typ.block_size() as u32;
        let (ne, nb) = (&self.ggml_ne, &self.ggml_nb);
        nb[0] == elsize
            && nb[1] == (nb[0] * ne[0]) / bsize
            && nb[2] == nb[1] * ne[1]
            && nb[3] == nb[2] * ne[2]
    }

    pub fn is_padded_1d(&self) -> bool {
        let elsize = self.typ.element_size() as u32;
        let (ne, nb) = (&self.ggml_ne, &self.ggml_nb);
        nb[0] == elsize && nb[2] == nb[1] * ne[1] && nb[3] == nb[2] * ne[2]
    }

    pub fn is_same_shape(&self, other: &Self) -> bool {
        self.ggml_ne
            .iter()
            .zip(other.ggml_ne.iter())
            .all(|(lels, rels)| lels == rels)
    }
}

#[derive(Clone)]
// TODO: Don't panic when something goes wrong, instead
// set state in tensor and context to indicate we're dead and
// just allow other operations (except actually creating/runnning
// the graph).
/// A GGML tensor. It uses a const generic for the dimensions.
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

//
// Internal methods
//
impl<const DIMS: usize> GTensor<DIMS>
where
    Dim<DIMS>: DimValid,
{
    /// # Safety
    /// Must be called with context mutex held.
    pub(crate) unsafe fn new_from_ptr(
        ctx: &GContext,
        ictx: &mut IContext,
        (mr, p): (GMemoryRequest, *mut gg::ggml_tensor),
    ) -> Result<Self> {
        let tptr = NonNull::new(p).ok_or(GTensorError::NullPointer)?;
        ictx.update_used_memory(&mr)?;
        Ok(Self {
            ctx: ctx.clone(),
            md: GTensorMetadata::from_ptr(tptr),
            tptr,
        })
    }

    pub(crate) fn make_dead_clone<const ODIMS: usize>(&self) -> GTensor<ODIMS>
    where
        Dim<ODIMS>: DimValid,
    {
        GTensor {
            ctx: self.ctx.clone(),
            tptr: self.tptr,
            md: GTensorMetadata::new_empty(),
        }
    }

    pub(crate) fn with_tensor<OUT, F>(&self, fun: F) -> Result<OUT>
    where
        F: FnOnce(&GContext, &mut IContext, *mut gg::ggml_tensor) -> Result<OUT>,
    {
        self.ctx
            .with_icontext(|ctx, mut ictx| fun(ctx, &mut ictx, self.tptr.as_ptr()))
    }

    pub(crate) fn with_tensor_infallible<OUT, F>(&self, fun: F) -> Result<OUT>
    where
        F: FnOnce(&GContext, &mut IContext, *mut gg::ggml_tensor) -> OUT,
    {
        self.ctx
            .with_icontext_infallible(|mut ictx| fun(&self.ctx, &mut ictx, self.tptr.as_ptr()))
    }

    pub(crate) fn with_tensor_delay_failure<OUT, DF, F>(&self, dfun: DF, fun: F) -> OUT
    where
        DF: Fn() -> OUT,
        F: FnOnce(&GContext, &mut IContext, *mut gg::ggml_tensor) -> Result<OUT>,
    {
        self.ctx
            .delay_failure_with_icontext(dfun, |ictx| fun(&self.ctx, ictx, self.tptr.as_ptr()))
    }

    pub(crate) fn with_tensor_unit_delay_failure<F>(&self, fun: F)
    where
        F: FnOnce(&GContext, &IContext, *mut gg::ggml_tensor) -> Result<()>,
    {
        self.ctx
            .delay_failure_with_icontext(|| (), |ictx| fun(&self.ctx, ictx, self.tptr.as_ptr()))
    }

    pub(crate) fn new_unary<const ODIMS: usize, F>(&self, fun: F) -> GTensor<ODIMS>
    where
        Dim<ODIMS>: DimValid,
        F: FnOnce(
            &GContext,
            &mut IContext,
            *mut gg::ggml_tensor,
        ) -> Result<(GMemoryRequest, *mut gg::ggml_tensor)>,
    {
        self.with_tensor_delay_failure(
            || self.make_dead_clone(),
            |ctx, ictx, tptr| {
                let fresult = fun(ctx, ictx, tptr)?;
                unsafe { GTensor::<ODIMS>::new_from_ptr(ctx, ictx, fresult) }
            },
        )
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
            &GContext,
            &mut IContext,
            *mut gg::ggml_tensor,
            *mut gg::ggml_tensor,
        ) -> Result<(GMemoryRequest, *mut gg::ggml_tensor)>,
        T: AsRef<GTensor<RDIMS>>,
    {
        let rhs = rhs.as_ref();
        if self.ctx.dead.load(SeqCst) || rhs.ctx.dead.load(SeqCst) {
            self.ctx.dead.store(true, SeqCst);
            rhs.ctx.dead.store(true, SeqCst);
            return self.make_dead_clone();
        }
        assert_eq!(
            self.ctx.ptrval, rhs.ctx.ptrval,
            "Cannot perform operation between tensors from different contexts!"
        );

        self.ctx.delay_failure_with_icontext(
            || self.make_dead_clone(),
            |mut ictx| {
                let ictx = &mut ictx;
                let (ltptr, rtptr) = (self.tptr.as_ptr(), rhs.tptr.as_ptr());
                let fresult = fun(&self.ctx, ictx, ltptr, rtptr)?;
                unsafe { GTensor::<ODIMS>::new_from_ptr(&self.ctx, ictx, fresult) }
            },
        )
    }
}

//
// Utility methods
//
impl<const DIMS: usize> GTensor<DIMS>
where
    Dim<DIMS>: DimValid,
{
    /// Alternate method to access a tensor's
    /// dimensions without needing an actual [GTensor] value
    /// to work with.
    pub const DIMS: usize = DIMS;

    /// Return the number of dimensions for this tensor.
    pub fn dims(&self) -> usize {
        DIMS
    }

    /// Returns the tensor data length in bytes.
    pub fn len(&self) -> usize {
        self.md.len_bytes
    }

    /// `true` if the tensor is empty.
    pub fn is_empty(&self) -> bool {
        self.md.len_bytes == 0
    }

    /// Returns the number of elements in this tensor.
    pub fn elements(&self) -> usize {
        self.md.len_elements
    }

    /// Returns the number of bytes each element uses.
    ///
    /// **Note**: May not be accurate for quantized types.
    pub fn element_size(&self) -> usize {
        self.md.element_size
    }

    /// Return the shape of this tensor as an array.
    ///
    /// **Note**: The length of the shape array will
    /// be equal to the tensor's dimensions.
    pub fn shape(&self) -> [usize; DIMS] {
        self.md.shape
    }

    /// Returns the GGML operation associated with this
    /// tensor if available.
    pub fn ggml_op(&self) -> gg::ggml_op {
        self.md.op
    }

    /// Returns the element type.
    pub fn element_type(&self) -> GType {
        self.md.typ
    }

    /// Returns a copy of the metadata associated with this tensor.
    pub fn metadata(&self) -> GTensorMetadata<DIMS> {
        self.md.clone()
    }

    /// Returns GGML's conception of this tensor's shape.
    ///
    /// **Note**: This is a low level function. Be aware that GGML
    /// shapes have the first two dimensions swapped.
    pub fn get_ne(&self) -> [u32; 4] {
        self.md.ggml_ne
    }

    /// Returns GGML's conception of this tensor's strides in bytes.
    ///
    /// **Note**: This is a low level function. Be aware that GGML
    /// shapes have the first two dimensions swapped. This also
    /// applies to the order of strides.
    ///
    /// **Note 2**: Also be aware that the strides are based on
    /// bytes, and _not_ the number of elements.
    pub fn get_nb(&self) -> [u32; 4] {
        self.md.ggml_nb
    }

    /// Immediately fills the tensor's data with zeros.
    pub fn fill_zero(&mut self) {
        self.with_tensor_unit_delay_failure(|ctx, _ictx, tptr| {
            if !ctx.no_alloc {
                unsafe {
                    gg::ggml_set_zero(tptr);
                }
            }
            Ok(())
        })
    }

    /// Immediately fills the tensor's data with the specified `i32`
    /// value.
    ///
    /// **Invariants**
    /// 1. The tensor's type must not be quantized.
    pub fn fill_i32(&mut self, val: i32) {
        self.with_tensor_unit_delay_failure(|ctx, _ictx, tptr| {
            if self.md.typ.is_quantized() {
                Err(GTensorError::TypeMismatch)?
            }

            if !ctx.no_alloc {
                unsafe {
                    gg::ggml_set_i32(tptr, val);
                }
            }
            Ok(())
        })
    }

    /// Immediately fills the tensor's data with the specified `f32`
    /// value.
    ///
    /// **Invariants**
    /// 1. The tensor's type must not be quantized.
    pub fn fill_f32(&mut self, val: f32) {
        self.with_tensor_unit_delay_failure(|ctx, _ictx, tptr| {
            if self.md.typ.is_quantized() {
                Err(GTensorError::TypeMismatch)?
            }

            if !ctx.no_alloc {
                unsafe {
                    gg::ggml_set_f32(tptr, val);
                }
            }
            Ok(())
        })
    }

    /// Immediately returns the value of an element at the
    /// specified index as a `f32`.
    ///
    /// **Invariants**
    /// 1. The tensor's type must not be quantized.
    /// 2. The index must be valid.
    pub fn get_f32_1d(&self, index: usize) -> Result<f32> {
        self.with_tensor(|ctx, _ictx, tptr| {
            if index >= self.md.len_elements {
                Err(GTensorError::InvalidOperation)?
            }
            if self.md.typ.is_quantized() {
                Err(GTensorError::TypeMismatch)?
            }
            ensure!(!ctx.no_alloc, GContextError::NoAlloc);
            Ok(unsafe { gg::ggml_get_f32_1d(tptr, index as i32) })
        })
    }

    /// Immediately returns the value of an element at the
    /// specified index as an `i32`.
    ///
    /// **Invariants**
    /// 1. The tensor's type must not be quantized.
    /// 2. The index must be valid.
    pub fn get_i32_1d(&self, index: usize) -> Result<i32> {
        self.with_tensor(|ctx, _ictx, tptr| {
            if index >= self.md.len_elements {
                Err(GTensorError::InvalidOperation)?
            }
            if self.md.typ.is_quantized() {
                Err(GTensorError::TypeMismatch)?
            }
            ensure!(!ctx.no_alloc, GContextError::NoAlloc);
            Ok(unsafe { gg::ggml_get_i32_1d(tptr, index as i32) })
        })
    }

    /// Immediately set the value of an element at the
    /// specified index to the specified `f32` value.
    ///
    /// **Invariants**
    /// 1. The tensor's type must not be quantized.
    /// 2. The index must be valid.
    pub fn set_f32_1d(&mut self, index: usize, val: f32) {
        self.with_tensor_unit_delay_failure(|_ctx, _ictx, tptr| {
            if index >= self.md.len_elements {
                Err(GTensorError::InvalidOperation)?
            }
            if self.md.typ.is_quantized() {
                Err(GTensorError::TypeMismatch)?
            }
            if self.ctx.no_alloc {
                return Ok(());
            }
            unsafe { gg::ggml_set_f32_1d(tptr, index as i32, val) };
            Ok(())
        })
    }

    /// Immediately set the value of an element at the
    /// specified index to the specified `i32` value.
    ///
    /// **Invariants**
    /// 1. The tensor's type must not be quantized.
    /// 2. The index must be valid.
    pub fn set_i32_1d(&mut self, index: usize, val: i32) {
        self.with_tensor_unit_delay_failure(|_ctx, _ictx, tptr| {
            if index >= self.md.len_elements {
                Err(GTensorError::InvalidOperation)?
            }
            if self.md.typ.is_quantized() {
                Err(GTensorError::TypeMismatch)?
            }
            if self.ctx.no_alloc {
                return Ok(());
            }
            unsafe {
                gg::ggml_set_i32_1d(tptr, index as i32, val);
            }
            Ok(())
        })
    }
}

//
// Unsafe public methods
//
impl<const DIMS: usize> GTensor<DIMS>
where
    Dim<DIMS>: DimValid,
{
    /// Low level function that allows mutably accessing a tensor's
    /// data as a slice of `u8`.
    ///
    /// # Safety
    /// Since this is working with the raw bytes, you need to be careful
    /// not to reinterpret as the wrong type or set the data to something
    /// that would contain an invalid value for the type.
    pub unsafe fn with_data_mut<F, O>(&mut self, fun: F) -> Result<O>
    where
        F: FnOnce(&mut [u8]) -> O,
    {
        ensure!(!self.ctx.no_alloc, GContextError::NoAlloc);
        self.with_tensor_infallible(|_ctx, _ictx, tptr| {
            fun(std::slice::from_raw_parts_mut(
                tptr.as_ref().unwrap().data as *mut u8,
                self.md.len_bytes,
            ))
        })
    }

    /// Low level function that allows accessing a tensor's
    /// data as a slice of `u8`.
    ///
    /// # Safety
    /// Since this is working with the raw bytes, you need to be careful
    /// not to reinterpret as the wrong type.
    pub unsafe fn with_data<F, O>(&self, fun: F) -> Result<O>
    where
        F: FnOnce(&[u8]) -> O,
    {
        ensure!(!self.ctx.no_alloc, GContextError::NoAlloc);
        self.with_tensor_infallible(|_ctx, _ictx, tptr| {
            fun(std::slice::from_raw_parts_mut(
                tptr.as_ref().unwrap().data as *mut u8,
                self.md.len_bytes,
            ))
        })
    }

    /// # Safety
    /// Fills a tensor with raw data. It's your responsibility to make sure the format is correct.
    pub unsafe fn populate_raw<S: AsRef<[u8]>>(&mut self, data: S) {
        let data = data.as_ref();
        self.with_tensor_unit_delay_failure(|ctx, _ictx, tptr| {
            if self.len() != data.len() {
                Err(GTensorError::BadPopulate {
                    got: data.len(),
                    expected: self.len(),
                })?
            }
            if ctx.no_alloc {
                return Ok(());
            }
            let tref = tptr.as_ref().unwrap();
            (tref.data as *mut u8).copy_from_nonoverlapping(data.as_ptr(), data.len());
            Ok(())
        })
    }
}

//
// Misc methods
//
impl<const DIMS: usize> GTensor<DIMS>
where
    Dim<DIMS>: DimValid,
{
    /// Copies data from the specified tensor into this tensor when the graph runs.
    ///
    /// **Note**: This immediately overwrites `self` with the copy.
    pub fn copy_from<T: AsRef<GTensor<DIMS>>>(&mut self, rhs: T) {
        let nt = self.new_binary(rhs, |ctx, ictx, ltptr, rtptr| {
            let md = GMemoryRequest::estimate_tensor_request_ictx(ctx, ictx, GType::F32, [])
                .fit_or_die()?;
            Ok((md, unsafe { gg::ggml_cpy(ictx.gptr(), rtptr, ltptr) }))
        });

        *self = nt;
    }

    // FIXME: More generic versions of these functions.
    /// Immediately copy the specified `f32` values into this tensor.
    ///
    /// **Invariants**
    /// 1. The tensor must be of type [GType::F32].
    /// 2. The length of the incoming data must match the size of the
    ///     tensor.
    pub fn populate_f32<S: AsRef<[f32]>>(&mut self, data: S) {
        let data = data.as_ref();
        self.with_tensor_unit_delay_failure(|ctx, _ictx, tptr| {
            if self.md.typ != GType::F32 {
                Err(GTensorError::TypeMismatch)?
            }
            if self.elements() != data.len() {
                Err(GTensorError::BadPopulate {
                    got: data.len(),
                    expected: self.elements(),
                })?
            }
            if ctx.no_alloc {
                return Ok(());
            }
            unsafe {
                let tref = tptr.as_ref().unwrap();
                (tref.data as *mut f32).copy_from_nonoverlapping(data.as_ptr(), data.len());
            }
            Ok(())
        })
    }

    // FIXME: More generic versions of these functions.
    /// Immediately copy the data from this tensor to the specified destination.
    ///
    /// **Invariants**
    /// 1. The tensor must be of type [GType::F32].
    /// 2. The length of the destination must match the size of the
    ///     tensor.
    /// 3. The destination must be elements of `f32`.
    pub fn copy_to_slice_f32<S: AsMut<[f32]>>(&self, mut dest: S) -> Result<()> {
        let dest = dest.as_mut();
        let elements = self.elements();

        self.with_tensor(|ctx, _ictx, tptr| {
            if self.md.typ != GType::F32 {
                Err(GTensorError::TypeMismatch)?
            }
            if elements != dest.len() {
                Err(GTensorError::BadPopulate {
                    got: dest.len(),
                    expected: elements,
                })?
            }
            ensure!(!ctx.no_alloc, GContextError::NoAlloc);
            let ts = unsafe {
                std::slice::from_raw_parts(tptr.as_ref().unwrap().data as *const f32, elements)
            };
            dest.copy_from_slice(ts);
            Ok(())
        })
    }
}
