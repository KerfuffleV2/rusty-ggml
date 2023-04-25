use std::{
    ffi::c_void,
    ops,
    ptr::NonNull,
    sync::{Arc, Mutex},
};

use ggml_sys_bleedingedge as gg;

use crate::{
    dims::*,
    tensor::{GgmlElementType, GgmlTensor},
};

#[repr(transparent)]
pub(crate) struct GgmlIContext(pub(crate) NonNull<gg::ggml_context>);

impl Drop for GgmlIContext {
    fn drop(&mut self) {
        unsafe { gg::ggml_free(self.0.as_ptr()) }
    }
}

impl ops::Deref for GgmlIContext {
    type Target = NonNull<gg::ggml_context>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl ops::DerefMut for GgmlIContext {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Clone)]
pub struct GgmlContext {
    pub(crate) ptrval: usize,
    pub(crate) ictx: Arc<Mutex<GgmlIContext>>,
}

#[repr(transparent)]
pub struct ScratchBuffer(Box<[u8]>);

impl ScratchBuffer {
    pub fn new(size: usize) -> Self {
        let mut data: Vec<u8> = Vec::with_capacity(size);
        #[allow(clippy::uninit_vec)]
        unsafe {
            data.set_len(size);
        }
        Self(data.into_boxed_slice())
    }
}

#[derive(Default)]
pub struct GgmlContextBuilder {
    mem_size: usize,
    no_alloc: bool,
}

impl GgmlContextBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn mem_size(mut self, mem_size: usize) -> Self {
        self.mem_size = mem_size;
        self
    }

    pub fn no_alloc(mut self, no_alloc: bool) -> Self {
        self.no_alloc = no_alloc;
        self
    }

    pub fn build(self) -> GgmlContext {
        let ptr = unsafe {
            gg::ggml_init(gg::ggml_init_params {
                mem_size: self.mem_size,
                mem_buffer: std::ptr::null_mut(),
                no_alloc: self.no_alloc,
            })
        };
        assert_ne!(ptr, std::ptr::null_mut(), "GGML init failed");
        GgmlContext {
            ptrval: ptr as usize,
            ictx: Arc::new(Mutex::new(GgmlIContext(NonNull::new(ptr).unwrap()))),
        }
    }
}

impl GgmlContext {
    pub fn tensor<const DIMS: usize>(
        &self,
        typ: GgmlElementType,
        shape: [usize; DIMS],
    ) -> GgmlTensor<DIMS>
    where
        Dim<DIMS>: DimValid,
        DimPair<DIMS, 4>: DimLt,
    {
        let ctx = self.ictx.lock().expect("Failed to get context mutex");
        let p = unsafe {
            match DIMS {
                1 => gg::ggml_new_tensor_1d(ctx.as_ptr(), typ as u32, shape[0] as i64),
                2 => gg::ggml_new_tensor_2d(
                    ctx.as_ptr(),
                    typ as u32,
                    shape[1] as i64,
                    shape[0] as i64,
                ),
                3 => gg::ggml_new_tensor_3d(
                    ctx.as_ptr(),
                    typ as u32,
                    shape[1] as i64,
                    shape[0] as i64,
                    shape[2] as i64,
                ),
                _ => unreachable!(),
            }
        };
        unsafe { GgmlTensor::new_from_ptr(self, p) }
    }

    pub fn use_scratch_buffer<'a>(&'a self, maybebuf: Option<&'a mut ScratchBuffer>) {
        let ctx = self.ictx.lock().expect("Failed to get context mutex");
        let (size, data) = maybebuf.map_or_else(
            || (0, std::ptr::null_mut()),
            |buf| (buf.0.len(), buf.0.as_ptr() as *mut c_void),
        );
        unsafe {
            gg::ggml_set_scratch(
                ctx.as_ptr(),
                gg::ggml_scratch {
                    offs: 0,
                    size,
                    data,
                },
            );
        }
    }

    pub fn compute(&self, graph: &mut GgmlGraph) {
        let ctx = self.ictx.lock().expect("Failed to get context mutex");
        unsafe { gg::ggml_graph_compute(ctx.0.as_ptr(), &mut graph.0) }
    }

    pub fn used_mem(&self) -> usize {
        let ctx = self.ictx.lock().expect("Failed to get context mutex");
        unsafe { gg::ggml_used_mem(ctx.0.as_ptr()) }
    }
}

#[repr(transparent)]
pub struct GgmlGraph(gg::ggml_cgraph);

impl GgmlGraph {
    pub fn new(n_threads: usize) -> Self {
        let mut graph = unsafe { std::mem::zeroed::<gg::ggml_cgraph>() };
        graph.n_threads = n_threads as i32;
        Self(graph)
    }

    pub fn build_forward_expand<const DIMS: usize, T: AsRef<GgmlTensor<DIMS>>>(&mut self, tensor: T)
    where
        Dim<DIMS>: DimValid,
    {
        unsafe { gg::ggml_build_forward_expand(&mut self.0, tensor.as_ref().tptr.as_ptr()) }
    }
}
