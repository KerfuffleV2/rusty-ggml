use std::ffi::c_void;

use anyhow::{ensure, Result};
use num_traits::ToPrimitive;
use thiserror::Error;

use crate::{ggml_sys, util::GType};

#[derive(Debug, Error, PartialEq)]
pub enum GQuantizeError {
    #[error("Cannot quantize type {0:?}")]
    UnquantizableType(GType),
    #[error("Unknown quantization error: {0}")]
    Other(String),
    //
}

#[derive(Debug, Clone, Default)]
pub struct GQuantizer {
    pub hist: [i64; Self::QUANTIZE_HISTOGRAM_SIZE],
    pub buffer: Vec<u8>,
}

impl GQuantizer {
    pub const QUANTIZE_HISTOGRAM_SIZE: usize = 16;

    pub fn histogram(&self) -> [i64; Self::QUANTIZE_HISTOGRAM_SIZE] {
        self.hist
    }

    pub fn quantize(&mut self, typ: GType, input: &[f32]) -> Result<&[u8]> {
        ensure!(typ.is_quantized(), GQuantizeError::UnquantizableType(typ));
        self.buffer.clear();
        self.buffer.reserve((input.len() * 4) + typ.block_size());
        let resultlen = unsafe {
            ggml_sys::ggml_quantize_chunk(
                typ.to_u32().unwrap(),
                input.as_ptr(),
                self.buffer.as_mut_ptr() as *mut c_void,
                0,
                input.len() as i32,
                self.hist.as_mut_ptr(),
            )
        };
        ensure!(
            resultlen <= self.buffer.capacity(),
            GQuantizeError::Other(format!(
                "Unexpect result length {resultlen} > buffer capacity {}",
                self.buffer.capacity()
            ))
        );
        unsafe {
            self.buffer.set_len(resultlen);
        }
        Ok(&self.buffer[0..resultlen])
    }
}
