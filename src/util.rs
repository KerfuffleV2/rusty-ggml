use num_traits::ToPrimitive;
use thiserror::Error;

use ggml_sys_bleedingedge as gg;

#[derive(Debug, Error, Clone, PartialEq)]
pub enum GError {
    #[error("Quantization error: {0}")]
    Quantization(crate::quantize::GQuantizeError),
    #[error("Context error: {0}")]
    Context(crate::context::GContextError),
    #[error("Tensor error: {0}")]
    Tensor(crate::gtensor::GTensorError),
}

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
/// GGML element type. Items starting with `Q` generally will be quantized.
pub enum GType {
    F32 = gg::ggml_type_GGML_TYPE_F32,
    F16 = gg::ggml_type_GGML_TYPE_F16,
    Q4_0 = gg::ggml_type_GGML_TYPE_Q4_0,
    Q4_1 = gg::ggml_type_GGML_TYPE_Q4_1,
    Q4_2 = gg::ggml_type_GGML_TYPE_Q4_2,
    Q5_0 = gg::ggml_type_GGML_TYPE_Q5_0,
    Q5_1 = gg::ggml_type_GGML_TYPE_Q5_1,
    Q8_0 = gg::ggml_type_GGML_TYPE_Q8_0,
    I8 = gg::ggml_type_GGML_TYPE_I8,
    I16 = gg::ggml_type_GGML_TYPE_I16,
    I32 = gg::ggml_type_GGML_TYPE_I32,
}

impl GType {
    /// Is this type quantized?
    pub fn is_quantized(&self) -> bool {
        self.to_u32()
            .map_or(false, |val| unsafe { gg::ggml_is_quantized(val) })
    }

    /// Returns the element size for a type.
    ///
    /// **Note**: This may not be accurate for quantized types.
    ///
    /// A result of `0` indicates an error.
    pub fn element_size(&self) -> usize {
        self.to_u32()
            .map_or(0, |val| unsafe { gg::ggml_type_size(val) })
    }

    /// Returns the element size for a type as a float. This
    /// can provide a more accurate result for quantized types.
    ///
    /// A result of `0.0` indicates an error.
    pub fn element_sizef(&self) -> f32 {
        self.to_u32()
            .map_or(0.0, |val| unsafe { gg::ggml_type_sizef(val) })
    }

    /// Returns the block size for this type. Generally always `1` for
    /// non-quantized types.
    pub fn block_size(&self) -> usize {
        self.to_u32()
            .map_or(0, |val| unsafe { gg::ggml_blck_size(val) } as usize)
    }
}
