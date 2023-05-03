use std::{
    ffi::c_void,
    ops,
    ptr::NonNull,
    sync::{
        atomic::{self, AtomicBool},
        Arc, Mutex, MutexGuard,
    },
};

use anyhow::{anyhow, bail, ensure, Result};
use thiserror::Error;

use ggml_sys_bleedingedge as gg;

use crate::{dims::*, gtensor::GTensor, util::GType, validation::*};

#[derive(Debug, Error, Clone)]
pub enum GContextError {
    #[error("Attempt to set invalid scratch buffer id {0}")]
    InvalidScratchBufferId(usize),

    #[error("Failed to lock context mutex")]
    MutexFailure,

    #[error("Context is deceased: {0}")]
    DeadContext(Arc<anyhow::Error>),

    #[error("Unknown error (likely mutex acquisition failure)")]
    Unknown,

    // FIXME: Add the other fields to the message.
    #[error("Not enough memory for request {0:?}")]
    InsufficientMemory(GMemoryRequest),

    // FIXME: Allow including more detail about what went wrong.
    #[error("Could not create tensor")]
    TensorCreationFailed,

    #[error("Attempt to access data in or compute with a no_alloc context")]
    NoAlloc,

    #[error("General error: {0}")]
    General(Arc<anyhow::Error>),
}

pub(crate) struct IContext {
    // Pointer to the GGML context.
    pub(crate) gctx: NonNull<gg::ggml_context>,

    pub(crate) context_memory: usize,
    // Amount of context memory currently used.
    pub(crate) context_used: usize,

    // List of scratch buffers. Only dropped when the `IContext` is
    // finally freed.
    pub(crate) scratch_buffers: Vec<ScratchBuffer>,

    // The current scratch buffer if set.
    pub(crate) current_scratch_buffer: Option<usize>,

    // Populated if an error occurred during some previous
    // operation.
    pub(crate) failed: Option<Arc<anyhow::Error>>,
}

impl Drop for IContext {
    // Since `IContext` lives inside an `Arc` this will only happen
    // when the very last instance of the `Arc` is dropped.
    fn drop(&mut self) {
        unsafe { gg::ggml_free(self.gctx.as_ptr()) }
    }
}

impl ops::Deref for IContext {
    type Target = NonNull<gg::ggml_context>;

    fn deref(&self) -> &Self::Target {
        &self.gctx
    }
}

impl ops::DerefMut for IContext {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.gctx
    }
}

impl IContext {
    pub(crate) unsafe fn gptr(&self) -> *mut gg::ggml_context {
        self.gctx.as_ptr()
    }

    pub(crate) fn update_used_memory(&mut self, mr: &GMemoryRequest) -> Result<()> {
        let mut mr = *mr;
        ensure!(
            mr.required_scratch == 0 || self.current_scratch_buffer == mr.current_scratch_buffer,
            "Scratch buffer mismatch in IContext::use_memory. Current: {:?}, expected: {:?}",
            self.current_scratch_buffer,
            mr.current_scratch_buffer,
        );
        match mr.reqtype {
            GMemoryRequestType::Tensor { .. } => (),
            wut => bail!("Request type {wut:?} currently not implemented in IContext::use_memory"),
        }
        let new_ctx_used = self.context_used + mr.required_ctx;
        if new_ctx_used > self.context_memory {
            mr.fits = false;
            bail!(GContextError::InsufficientMemory(mr));
        }
        if let Some(bufid) = &mr.current_scratch_buffer {
            let buf = &mut self.scratch_buffers[*bufid];
            let new_scratch_used = buf.used + mr.required_scratch;
            if new_scratch_used > buf.buf.len() {
                println!(
                    "MEM(scratch): {new_scratch_used} > {} -- {mr:?}",
                    buf.buf.len()
                );
                mr.fits = false;
                bail!(GContextError::InsufficientMemory(mr));
            }
            buf.used = new_scratch_used;
        }
        self.context_used = new_ctx_used;
        Ok(())
    }
}

#[derive(Clone)]
pub struct GContext {
    // This is just used to validate that operations for objects containing a
    // a context (i.e. tensors) have the same context. It is never actually
    // used as a pointer or updated after the context is created.
    pub(crate) ptrval: usize,

    // Amount of context memory allocated (in bytes)
    #[allow(dead_code)]
    pub(crate) context_size: usize,

    #[allow(dead_code)]
    pub(crate) no_alloc: bool,

    // This atomic is used to mark the context as dead. Ideally we could
    // mark it in the `ictx` field, but one failure condition is failing to
    // acquire the mutex: in that case all we can do is mark the context as
    // dead using this field.
    pub(crate) dead: Arc<AtomicBool>,

    // The real context structure which contains a pointer to the actual
    // GGML context.
    pub(crate) ictx: Arc<Mutex<IContext>>,
}

/// GGML scratch buffer structure used for temporary data storage.
pub struct ScratchBuffer {
    pub(crate) buf: Box<[u8]>,
    pub(crate) used: usize,
}

impl ScratchBuffer {
    /// Create a new scratch buffer with the specified size (in bytes).
    pub fn new(size: usize) -> Self {
        let mut data: Vec<u8> = Vec::with_capacity(size);
        #[allow(clippy::uninit_vec)]
        unsafe {
            data.set_len(size);
        }
        Self {
            buf: data.into_boxed_slice(),
            used: 0,
        }
    }
}

#[derive(Default)]
/// GGML context builder structure used to build a
/// [GContext].
pub struct GContextBuilder {
    mem_size: usize,
    no_alloc: bool,
}

// FIXME: We probably should use the typestate pattern in here to make sure
// people don't do something silly like build an alloc context with 0 memory.
impl GContextBuilder {
    /// Create a new [GContextBuilder].
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the GGML context size.
    pub fn mem_size(mut self, mem_size: usize) -> Self {
        self.mem_size = mem_size;
        self
    }

    /// When a context is in no_alloc mode, apparently this
    /// means that data for tensors is not allocated at all
    /// so all requests will succeed (althoug the context
    /// needs to have at least enough memory for the GGML
    /// objects).
    ///
    /// Naturally, you can't actually execute the graph when
    /// this is turned on, so the main use is to probe how much
    /// memory building a graph will require and then use that
    /// value to create another context with the expected memory
    /// size and then populate that with the actual tensor data.
    pub fn no_alloc(mut self, no_alloc: bool) -> Self {
        self.no_alloc = no_alloc;
        self
    }

    /// Build a GGML context ([GContext]) based on the
    /// builder's configuration.
    pub fn build(self) -> Result<GContext> {
        let ptr = unsafe {
            gg::ggml_init(gg::ggml_init_params {
                mem_size: self.mem_size,
                mem_buffer: std::ptr::null_mut(),
                no_alloc: self.no_alloc,
            })
        };
        ensure!(!ptr.is_null(), "GGML init failed");
        Ok(GContext {
            context_size: self.mem_size,
            no_alloc: self.no_alloc,
            ptrval: ptr as usize,
            ictx: Arc::new(Mutex::new(IContext {
                gctx: NonNull::new(ptr).unwrap(),
                context_used: 0,
                context_memory: self.mem_size,
                scratch_buffers: vec![],
                current_scratch_buffer: None,
                failed: None,
            })),
            dead: Arc::new(AtomicBool::new(false)),
        })
    }
}

impl GContext {
    pub(crate) fn with_icontext<OUT, F>(&self, fun: F) -> Result<OUT>
    where
        F: FnOnce(&GContext, MutexGuard<IContext>) -> Result<OUT>,
    {
        let failed = self.dead.load(atomic::Ordering::SeqCst);
        let ictx = self
            .ictx
            .lock()
            .map_err(|_e| anyhow!(GContextError::MutexFailure))?;
        if let Some(e) = ictx.failed.clone() {
            bail!(GContextError::DeadContext(e));
        }
        if failed {
            bail!(GContextError::Unknown)
        } else {
            fun(self, ictx)
        }
    }

    // FIXME: This logic seems kind of weird. Same problem in `Tensor::with_tensor_infallible`.
    pub(crate) fn with_icontext_infallible<OUT, F>(&self, fun: F) -> Result<OUT>
    where
        F: FnOnce(MutexGuard<IContext>) -> OUT,
    {
        let failed = self.dead.load(atomic::Ordering::SeqCst);
        let mut ctx = self.ictx.lock().map_err(|_e| {
            self.dead.store(true, atomic::Ordering::SeqCst);
            GContextError::MutexFailure
        })?;
        if let Some(e) = ctx.failed.clone() {
            bail!(GContextError::DeadContext(e));
        }
        // This might look weird but the idea is that we might have failed previously
        // due to being unable to acquire the mutex. Since we didn't have the mutex,
        // naturally it was impossible to set the `failed` field inside the `IContext`
        // structure.
        // There probably still is a race condition here but it should be very unlikely.
        if failed {
            let e = GContextError::Unknown;
            ctx.failed = Some(Arc::new(anyhow::Error::new(e.clone())));
            Err(e)?;
        }
        Ok(fun(ctx))
    }

    pub(crate) fn delay_failure_with_icontext<OUT, DF, F>(&self, dfun: DF, fun: F) -> OUT
    where
        DF: Fn() -> OUT,
        F: FnOnce(&mut IContext) -> Result<OUT>,
    {
        self.with_icontext_infallible(|mut ictx| {
            fun(&mut ictx).unwrap_or_else(|e| {
                // We have the context mutex but the handler function returned
                // an error condition. So store the error in the context and mark it as dead.
                self.dead.store(true, atomic::Ordering::SeqCst);
                ictx.failed = Some(Arc::new(e));
                dfun()
            })
        })
        .unwrap_or_else(|_e| {
            // We couldn't get the context mutex, but we can still mark the context as dead.
            self.dead.store(true, atomic::Ordering::SeqCst);
            dfun()
        })
    }

    pub fn estimate_tensor_size<const DIMS: usize>(
        &self,
        typ: GType,
        shape: [usize; DIMS],
    ) -> Result<GMemoryRequest> {
        self.with_icontext(|ctx, ictx| {
            Ok(GMemoryRequest::estimate_tensor_request_ictx(
                ctx, &ictx, typ, shape,
            ))
        })
    }

    /// Create a new tensor with the specified [type](GType) and shape.
    ///
    /// This uses const generics to determine the new tensor's dimensions. The tensor dimensions
    /// will be equal to the number of items in the `shape` array.
    pub fn tensor<const DIMS: usize>(
        &self,
        typ: GType,
        shape: [usize; DIMS],
    ) -> Result<GTensor<DIMS>>
    where
        Dim<DIMS>: DimValid,
        DimPair<DIMS, 4>: DimLt,
    {
        self.with_icontext(|ctx, mut ictx| {
            let mr = GMemoryRequest::estimate_tensor_request_ictx(self, &ictx, typ, shape);
            mr.fit_or_die()?;

            unsafe {
                let p = match DIMS {
                    1 => gg::ggml_new_tensor_1d(ictx.gptr(), typ as u32, shape[0] as i64),
                    2 => gg::ggml_new_tensor_2d(
                        ictx.gptr(),
                        typ as u32,
                        shape[1] as i64,
                        shape[0] as i64,
                    ),
                    3 => gg::ggml_new_tensor_3d(
                        ictx.gptr(),
                        typ as u32,
                        shape[1] as i64,
                        shape[0] as i64,
                        shape[2] as i64,
                    ),
                    _ => unreachable!(),
                };

                if p.is_null() {
                    Err(GContextError::TensorCreationFailed)?;
                }
                GTensor::new_from_ptr(ctx, &mut ictx, (mr, p))
            }
        })
    }

    /// Register a scratch buffer. The return value is the scratch buffer id
    /// which can be used with [Self::set_scratch_buffer].
    pub fn register_scratch_buffer(&mut self, buf: ScratchBuffer) -> Result<usize> {
        self.with_icontext_infallible(|mut ictx| {
            let bufid = ictx.scratch_buffers.len();
            ictx.scratch_buffers.push(buf);
            bufid
        })
    }

    /// Set or clear the current scratch buffer. A valid id as returned by
    /// [Self::register_scratch_buffer] must be supplied.
    ///
    /// **Note**: Scratch buffers cannot be removed directly and are only freed
    /// when the [GContext] structure is dropped.
    pub fn set_scratch_buffer(&self, maybebufid: Option<usize>) -> Result<()> {
        self.with_icontext(|_ctx, mut ictx| {
            let (size, data) = if let Some(bufid) = maybebufid {
                if bufid >= ictx.scratch_buffers.len() {
                    Err(GContextError::InvalidScratchBufferId(bufid))?;
                }
                ictx.current_scratch_buffer = maybebufid;
                let buf = &mut ictx.scratch_buffers[bufid].buf;
                (buf.len(), buf.as_mut_ptr() as *mut c_void)
            } else {
                (0, std::ptr::null_mut())
            };
            unsafe {
                gg::ggml_set_scratch(
                    ictx.gptr(),
                    gg::ggml_scratch {
                        offs: 0,
                        size,
                        data,
                    },
                );
            }
            Ok(())
        })
    }

    /// Runs the supplied graph using this context.
    pub fn compute(&self, graph: &mut GGraph) -> Result<()> {
        ensure!(!self.no_alloc, GContextError::NoAlloc);
        self.with_icontext_infallible(|ictx| unsafe {
            gg::ggml_graph_compute(ictx.gptr(), &mut graph.0)
        })
    }

    /// Returns the amount of memory GGML is currently using.
    pub fn used_mem(&self) -> Result<usize> {
        self.with_icontext_infallible(|ictx| unsafe { gg::ggml_used_mem(ictx.gptr()) })
    }
}

#[repr(transparent)]
pub struct GGraph(gg::ggml_cgraph);

impl GGraph {
    /// Create a new computation graph with the specified number of threads.
    pub fn new(n_threads: usize) -> Self {
        let mut graph = unsafe { std::mem::zeroed::<gg::ggml_cgraph>() };
        graph.n_threads = n_threads as i32;
        Self(graph)
    }

    /// Register a tensor to be processed when the graph is computed.
    pub fn build_forward_expand<const DIMS: usize, T: AsRef<GTensor<DIMS>>>(
        &mut self,
        tensor: T,
    ) -> Result<()>
    where
        Dim<DIMS>: DimValid,
    {
        // FIXME: Should we bail out here if no_alloc?
        tensor
            .as_ref()
            .with_tensor_infallible(|_ctx, _ictx, tptr| unsafe {
                gg::ggml_build_forward_expand(&mut self.0, tptr)
            })
    }
}
