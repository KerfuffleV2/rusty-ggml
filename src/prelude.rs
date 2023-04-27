pub use ggml_sys_bleedingedge as ggml_sys;

pub use crate::{context::*, dims::*, gtensor::*, map_binop, map_unop, quantize::*, util::*};

pub type GTensor1 = GTensor<1>;
pub type GTensor2 = GTensor<2>;
pub type GTensor3 = GTensor<3>;
