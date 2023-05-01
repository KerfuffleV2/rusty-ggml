use ggml_sys_bleedingedge as gg;

use super::tensor::*;
use crate::{dims::*, util::GType};

#[macro_export]
/// Creates an anonymous function for use with [GTensor::map_unary].
///
/// **Example**:
///
/// ```ignore
/// tensor.map_unary(map_unop!(|src| src + 10))
/// ```
macro_rules! map_unop (
  ( |$srcid:ident| $body:expr  ) => {
    {
      unsafe extern "C" fn __map_uop_fn(n: ::std::ffi::c_int, dst: *mut f32, src: *const f32) {
          let n = n as usize;
          let dst = ::std::slice::from_raw_parts_mut(dst, n);
          let src = ::std::slice::from_raw_parts(src, n);

          dst.iter_mut()
              .zip(src.iter().copied())
              .for_each(|( __map_uop_dstel, $srcid )| {
                *__map_uop_dstel = $body ;
              });
      }
      __map_uop_fn
    }
  }
);

#[macro_export]
/// Creates an anonymous function for use with [GTensor::map_binary].
///
/// **Example**:
///
/// ```ignore
/// a.map_binary(b, map_binop!(|el_a, el_b| a + b))
/// ```
macro_rules! map_binop (
  ( |$src0id:ident, $src1id:ident| $body:expr  ) => {
    {
      unsafe extern "C" fn __map_bop_fn(
        n: ::std::os::raw::c_int,
        dst: *mut f32,
        src0: *const f32,
        src1: *const f32,
      ) {
          let n = n as usize;
          let dst = ::std::slice::from_raw_parts_mut(dst, n);
          let src0 = ::std::slice::from_raw_parts(src0, n);
          let src1 = ::std::slice::from_raw_parts(src1, n);

          dst.iter_mut()
              .zip(src0.iter().copied())
              .zip(src1.iter().copied())
              .for_each(|((__map_bop_dstel, $src0id), $src1id)| {
                *__map_bop_dstel = $body ;
              });
      }
      __map_bop_fn
    }
  }
);

impl<const DIMS: usize> GTensor<DIMS>
where
    Dim<DIMS>: DimValid,
{
    /// Elementwise unary map operation on tensor `A`.
    /// Returns a new tensor.
    ///
    /// **Note**: This function is rather unfriendly to use directly.
    /// See the [map_unop!] macro which will help you create the required
    /// `unsafe extern "C"` function.
    ///
    /// `a.map_unary(unary_fun_ptr)`
    ///
    /// **Invariants**
    /// 1. The tensor must be of type [GType::F32](crate::util::GType::F32).
    /// 2. The result will be the same shape and type as `A`.
    ///
    /// **Example** (pseudocode):
    /// ```ignore
    /// use std::ffi::c_int;
    /// unsafe extern "C" fn unary_map_fn(n: c_int, dst: *mut f32, src: *const f32) {
    ///     let dst = ::std::slice::from_raw_parts_mut(dst, n as usize);
    ///     let src = ::std::slice::from_raw_parts(src, n as usize);
    ///     dst.iter_mut()
    ///         .zip(src.iter().copied())
    ///         .for_each(|(dst, src0)| *dst = src + 10);
    /// }
    ///
    /// let a = [1, 2, 3, 4];
    /// let result = a.map_unary(unary_map_fn);
    /// assert_eq!(result, [11, 12, 13, 14]);
    /// ```
    pub fn map_unary(
        &self,
        fun: unsafe extern "C" fn(arg1: ::std::os::raw::c_int, arg2: *mut f32, arg3: *const f32),
    ) -> Self {
        self.new_unary(|ctx, tptr| {
            if self.md.typ != GType::F32 {
                Err(GTensorError::TypeMismatch)?;
            }
            unsafe { Ok(gg::ggml_map_unary_f32(ctx, tptr, Some(fun))) }
        })
    }

    /// Elementwise binary map operation on tensors `A` and `B`.
    /// Returns a new tensor.
    ///
    /// **Note**: This function is rather unfriendly to use directly.
    /// See the [map_binop!] macro which will help you create the required
    /// `unsafe extern "C"` function.
    ///
    /// `a.map_binary(b, binary_fun_ptr)`
    ///
    /// **Invariants**
    /// 1. The tensor must be of type [GType::F32](crate::util::GType::F32).
    /// 2. `A` and `B` must be the same shape.
    /// 3. The result will be the same shape and type as `A`.
    ///
    /// **Example** (pseudocode):
    /// ```ignore
    /// use std::ffi::c_int;
    /// unsafe extern "C" fn binary_map_fn(
    ///     n: c_int,
    ///     dst: *mut f32,
    ///     src0: *const f32
    ///     src1: *const f32
    /// ) {
    ///     let dst = ::std::slice::from_raw_parts_mut(dst, n as usize);
    ///     let src0 = ::std::slice::from_raw_parts(src, n as usize);
    ///     let src1 = ::std::slice::from_raw_parts(src, n as usize);
    ///     dst.iter_mut()
    ///         .zip(src0.iter().copied())
    ///         .zip(src1.iter().copied())
    ///         .for_each(|((dst, src0), src1)| *dst = src0 + src1);
    /// }
    ///
    /// let a = [1, 2, 3, 4];
    /// let a = [10, 10, 10, 10];
    /// let result = a.map_binary(b, binary_map_fn);
    /// assert_eq!(result, [11, 12, 13, 14]);
    /// ```
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
        let rtyp = rhs.as_ref().md.typ;
        self.new_binary(rhs, |ctx, ltptr, rtptr| {
            //
            if self.md.typ != GType::F32 || rtyp != GType::F32 {
                Err(GTensorError::TypeMismatch)?;
            }
            unsafe { Ok(gg::ggml_map_binary_f32(ctx, ltptr, rtptr, Some(fun))) }
        })
    }
}
