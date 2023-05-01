use std::ffi::c_void;

use anyhow::{ensure, Result};
use num_traits::ToPrimitive;
use thiserror::Error;

use crate::{ggml_sys, util::GType};

#[derive(Debug, Error, Clone, PartialEq)]
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

    /// Fetches the current histogram.
    pub fn histogram(&self) -> [i64; Self::QUANTIZE_HISTOGRAM_SIZE] {
        self.hist
    }

    /// Reset the histogram.
    pub fn reset_histogram(&mut self) {
        self.hist.iter_mut().for_each(|i| *i = 0);
    }

    /// Quantizes the input `[f32]` slice with the specified type and returns a reference to this
    /// [GQuantizer]'s internal buffer.
    ///
    /// **WARNING**: The return value is only valid while the [GQuantizer] it came from is alive
    /// (lifetimes should ensure this) _and_ until the next time you call the `quantize` method.
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
