use crate::gl::{GLuint, glGenBuffers, glBufferData, glBufferSubData, glDeleteBuffers, GLenum, GL_ARRAY_BUFFER, GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW, GL_DYNAMIC_DRAW, GL_STREAM_DRAW};
use crate::{BufferType, IndexType, Context, Usage};
use std::mem;

#[derive(Clone, Copy, Debug)]
pub struct Buffer {
    pub(crate) gl_buf: GLuint,
    pub(crate) buffer_type: BufferType,
    pub(crate) size: usize,
    pub(crate) index_type: Option<IndexType>,
}

impl Buffer {
    /// Create an immutable buffer resource object.
    /// ```ignore
    /// #[repr(C)]
    /// struct Vertex {
    ///     pos: Vec2,
    ///     uv: Vec2,
    /// }
    /// let vertices: [Vertex; 4] = [
    ///     Vertex { pos : Vec2 { x: -0.5, y: -0.5 }, uv: Vec2 { x: 0., y: 0. } },
    ///     Vertex { pos : Vec2 { x:  0.5, y: -0.5 }, uv: Vec2 { x: 1., y: 0. } },
    ///     Vertex { pos : Vec2 { x:  0.5, y:  0.5 }, uv: Vec2 { x: 1., y: 1. } },
    ///     Vertex { pos : Vec2 { x: -0.5, y:  0.5 }, uv: Vec2 { x: 0., y: 1. } },
    /// ];
    /// let buffer = Buffer::immutable(ctx, BufferType::VertexBuffer, &vertices);
    /// ```
    pub fn immutable<T>(ctx: &mut Context, buffer_type: BufferType, data: &[T]) -> Buffer {
        //println!("{} {}", mem::size_of::<T>(), mem::size_of_val(data));
        let index_type = if buffer_type == BufferType::IndexBuffer {
            Some(IndexType::for_type::<T>())
        } else {
            None
        };

        let gl_target = gl_buffer_target(&buffer_type);
        let gl_usage = gl_usage(&Usage::Immutable);
        let size = mem::size_of_val(data);
        let mut gl_buf: u32 = 0;

        unsafe {
            glGenBuffers(1, &mut gl_buf as *mut _);
            ctx.cache.store_buffer_binding(gl_target);
            ctx.cache.bind_buffer(gl_target, gl_buf, index_type);
            glBufferData(gl_target, size as _, std::ptr::null() as *const _, gl_usage);
            glBufferSubData(gl_target, 0, size as _, data.as_ptr() as *const _);
            ctx.cache.restore_buffer_binding(gl_target);
        }

        Buffer {
            gl_buf,
            buffer_type,
            size,
            index_type,
        }
    }

    pub fn stream(ctx: &mut Context, buffer_type: BufferType, size: usize) -> Buffer {
        let index_type = if buffer_type == BufferType::IndexBuffer {
            Some(IndexType::Short)
        } else {
            None
        };

        let gl_target = gl_buffer_target(&buffer_type);
        let gl_usage = gl_usage(&Usage::Stream);
        let mut gl_buf: u32 = 0;

        unsafe {
            glGenBuffers(1, &mut gl_buf as *mut _);
            ctx.cache.store_buffer_binding(gl_target);
            ctx.cache.bind_buffer(gl_target, gl_buf, None);
            glBufferData(gl_target, size as _, std::ptr::null() as *const _, gl_usage);
            ctx.cache.restore_buffer_binding(gl_target);
        }

        Buffer {
            gl_buf,
            buffer_type,
            size,
            index_type,
        }
    }

    pub fn index_stream(ctx: &mut Context, index_type: IndexType, size: usize) -> Buffer {
        let gl_target = gl_buffer_target(&BufferType::IndexBuffer);
        let gl_usage = gl_usage(&Usage::Stream);
        let mut gl_buf: u32 = 0;

        unsafe {
            glGenBuffers(1, &mut gl_buf as *mut _);
            ctx.cache.store_buffer_binding(gl_target);
            ctx.cache.bind_buffer(gl_target, gl_buf, None);
            glBufferData(gl_target, size as _, std::ptr::null() as *const _, gl_usage);
            ctx.cache.restore_buffer_binding(gl_target);
        }

        Buffer {
            gl_buf,
            buffer_type: BufferType::IndexBuffer,
            size,
            index_type: Some(index_type),
        }
    }
    pub fn update<T>(&self, ctx: &mut Context, data: &[T]) {
        //println!("{} {}", mem::size_of::<T>(), mem::size_of_val(data));
        if self.buffer_type == BufferType::IndexBuffer {
            assert!(self.index_type.is_some());
            assert!(self.index_type.unwrap() == IndexType::for_type::<T>());
        };

        let size = mem::size_of_val(data);

        assert!(size <= self.size);

        let gl_target = gl_buffer_target(&self.buffer_type);
        ctx.cache.store_buffer_binding(gl_target);
        ctx.cache
            .bind_buffer(gl_target, self.gl_buf, self.index_type);
        unsafe { glBufferSubData(gl_target, 0, size as _, data.as_ptr() as *const _) };
        ctx.cache.restore_buffer_binding(gl_target);
    }

    /// Size of buffer in bytes
    pub fn size(&self) -> usize {
        self.size
    }

    /// Delete GPU buffer, leaving handle unmodified.
    ///
    /// More high-level code on top of miniquad probably is going to call this in Drop implementation of some
    /// more RAII buffer object.
    ///
    /// There is no protection against using deleted textures later. However its not an UB in OpenGl and thats why
    /// this function is not marked as unsafe
    pub fn delete(&self) {
        unsafe { glDeleteBuffers(1, &self.gl_buf as *const _) }
    }
}

fn gl_buffer_target(buffer_type: &BufferType) -> GLenum {
    match buffer_type {
        BufferType::VertexBuffer => GL_ARRAY_BUFFER,
        BufferType::IndexBuffer => GL_ELEMENT_ARRAY_BUFFER,
    }
}

fn gl_usage(usage: &Usage) -> GLenum {
    match usage {
        Usage::Immutable => GL_STATIC_DRAW,
        Usage::Dynamic => GL_DYNAMIC_DRAW,
        Usage::Stream => GL_STREAM_DRAW,
    }
}
