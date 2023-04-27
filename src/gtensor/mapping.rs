use ggml_sys_bleedingedge as gg;

use super::tensor::*;
use crate::dims::*;

#[macro_export]
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
    pub fn map_unary(
        &self,
        fun: unsafe extern "C" fn(arg1: ::std::os::raw::c_int, arg2: *mut f32, arg3: *const f32),
    ) -> Self {
        self.new_unary(|ctx, tptr| unsafe { gg::ggml_map_unary_f32(ctx, tptr, Some(fun)) })
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
}
