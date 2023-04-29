use num_traits::ToPrimitive;

use ggml_sys_bleedingedge as gg;

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
    pub fn is_quantized(&self) -> bool {
        self.to_u32()
            .map_or(false, |val| unsafe { gg::ggml_is_quantized(val) })
    }

    pub fn element_size(&self) -> usize {
        self.to_u32()
            .map_or(0, |val| unsafe { gg::ggml_type_size(val) })
    }

    pub fn element_sizef(&self) -> f32 {
        self.to_u32()
            .map_or(0.0, |val| unsafe { gg::ggml_type_sizef(val) })
    }

    pub fn block_size(&self) -> usize {
        self.to_u32()
            .map_or(0, |val| unsafe { gg::ggml_blck_size(val) } as usize)
    }
}
