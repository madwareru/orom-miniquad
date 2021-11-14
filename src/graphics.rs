use std::{ffi::CString};

mod texture;
mod shader_impl;
mod types_impl;
mod query_impl;
mod buffer_impl;

use crate::sapp::*;

pub use texture::{FilterMode, Texture, TextureAccess, TextureFormat, TextureParams, TextureWrap, TextureKind};
pub(crate) use shader_impl::{ShaderInternal};
pub use shader_impl::{Shader, ShaderMeta, ShaderImage, ShaderUniform, ShaderType, ShaderError, load_shader};
pub use types_impl::{
    UniformType, UniformDesc, UniformBlockLayout, VertexFormat, VertexStep, BufferLayout,
    VertexAttribute, PipelineLayout, BlendState, StencilState, StencilFaceState, StencilOp, CompareFunc,
    Equation, BlendValue, BlendFactor, CullFace, FrontFaceOrder, Comparison, PrimitiveType, IndexType,
    BufferType, Usage
};
pub use query_impl::*;
pub use buffer_impl::*;

type ColorMask = (bool, bool, bool, bool);

#[derive(Default, Copy, Clone)]
struct CachedAttribute {
    attribute: VertexAttributeInternal,
    gl_vbuf: GLuint,
}

struct GlCache {
    stored_index_buffer: GLuint,
    stored_index_type: Option<IndexType>,
    stored_vertex_buffer: GLuint,
    stored_texture: GLuint,
    index_buffer: GLuint,
    index_type: Option<IndexType>,
    vertex_buffer: GLuint,
    textures: [GLuint; MAX_SHADERSTAGE_IMAGES],
    cur_pipeline: Option<Pipeline>,
    color_blend: Option<BlendState>,
    alpha_blend: Option<BlendState>,
    stencil: Option<StencilState>,
    color_write: ColorMask,
    cull_face: CullFace,
    attributes: [Option<CachedAttribute>; MAX_VERTEX_ATTRIBUTES],
}

impl GlCache {
    fn bind_buffer(&mut self, target: GLenum, buffer: GLuint, index_type: Option<IndexType>) {
        if target == GL_ARRAY_BUFFER {
            if self.vertex_buffer != buffer {
                self.vertex_buffer = buffer;
                unsafe {
                    glBindBuffer(target, buffer);
                }
            }
        } else {
            if self.index_buffer != buffer {
                self.index_buffer = buffer;
                unsafe {
                    glBindBuffer(target, buffer);
                }
            }
            self.index_type = index_type;
        }
    }

    fn store_buffer_binding(&mut self, target: GLenum) {
        if target == GL_ARRAY_BUFFER {
            self.stored_vertex_buffer = self.vertex_buffer;
        } else {
            self.stored_index_buffer = self.index_buffer;
            self.stored_index_type = self.index_type;
        }
    }

    fn restore_buffer_binding(&mut self, target: GLenum) {
        if target == GL_ARRAY_BUFFER {
            if self.stored_vertex_buffer != 0 {
                self.bind_buffer(target, self.stored_vertex_buffer, None);
                self.stored_vertex_buffer = 0;
            }
        } else {
            if self.stored_index_buffer != 0 {
                self.bind_buffer(target, self.stored_index_buffer, self.stored_index_type);
                self.stored_index_buffer = 0;
            }
        }
    }

    fn bind_texture(&mut self, slot_index: usize, texture: GLuint) {
        unsafe {
            glActiveTexture(GL_TEXTURE0 + slot_index as GLuint);
            if self.textures[slot_index] != texture {
                glBindTexture(GL_TEXTURE_2D, texture);
                self.textures[slot_index] = texture;
            }
        }
    }

    fn store_texture_binding(&mut self, slot_index: usize) {
        self.stored_texture = self.textures[slot_index];
    }

    fn restore_texture_binding(&mut self, slot_index: usize) {
        self.bind_texture(slot_index, self.stored_texture);
    }

    fn clear_buffer_bindings(&mut self) {
        self.bind_buffer(GL_ARRAY_BUFFER, 0, None);
        self.vertex_buffer = 0;

        self.bind_buffer(GL_ELEMENT_ARRAY_BUFFER, 0, None);
        self.index_buffer = 0;
    }

    fn clear_texture_bindings(&mut self) {
        for ix in 0..MAX_SHADERSTAGE_IMAGES {
            if self.textures[ix] != 0 {
                self.bind_texture(ix, 0);
                self.textures[ix] = 0;
            }
        }
    }
}

pub enum PassAction {
    Nothing,
    Clear {
        color: Option<(f32, f32, f32, f32)>,
        depth: Option<f32>,
        stencil: Option<i32>,
    },
}

impl PassAction {
    pub fn clear_color(r: f32, g: f32, b: f32, a: f32) -> PassAction {
        PassAction::Clear {
            color: Some((r, g, b, a)),
            depth: Some(1.),
            stencil: None,
        }
    }
}

impl Default for PassAction {
    fn default() -> PassAction {
        PassAction::Clear {
            color: Some((0.0, 0.0, 0.0, 0.0)),
            depth: Some(1.),
            stencil: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RenderPass(usize);

struct RenderPassInternal {
    gl_fb: GLuint,
    texture: Texture,
    depth_texture: Option<Texture>,
}

impl RenderPass {
    pub fn new(
        context: &mut Context,
        color_img: Texture,
        depth_img: impl Into<Option<Texture>>,
    ) -> RenderPass {
        let mut gl_fb = 0;

        let depth_img = depth_img.into();

        unsafe {
            glGenFramebuffers(1, &mut gl_fb as *mut _);
            glBindFramebuffer(GL_FRAMEBUFFER, gl_fb);
            glFramebufferTexture2D(
                GL_FRAMEBUFFER,
                GL_COLOR_ATTACHMENT0,
                GL_TEXTURE_2D,
                color_img.texture,
                0,
            );
            if let Some(depth_img) = depth_img {
                glFramebufferTexture2D(
                    GL_FRAMEBUFFER,
                    GL_DEPTH_ATTACHMENT,
                    GL_TEXTURE_2D,
                    depth_img.texture,
                    0,
                );
            }
            glBindFramebuffer(GL_FRAMEBUFFER, context.default_framebuffer);
        }
        let pass = RenderPassInternal {
            gl_fb,
            texture: color_img,
            depth_texture: depth_img,
        };

        context.passes.push(pass);

        RenderPass(context.passes.len() - 1)
    }

    pub fn texture(&self, ctx: &mut Context) -> Texture {
        let render_pass = &mut ctx.passes[self.0];

        render_pass.texture
    }

    pub fn delete(&self, ctx: &mut Context) {
        let render_pass = &mut ctx.passes[self.0];

        unsafe { glDeleteFramebuffers(1, &mut render_pass.gl_fb as *mut _) }

        render_pass.texture.delete();
        if let Some(depth_texture) = render_pass.depth_texture {
            depth_texture.delete();
        }
    }
}

pub const MAX_VERTEX_ATTRIBUTES: usize = 16;
pub const MAX_SHADERSTAGE_IMAGES: usize = 12;

pub struct Context {
    shaders: Vec<ShaderInternal>,
    pipelines: Vec<PipelineInternal>,
    passes: Vec<RenderPassInternal>,
    default_framebuffer: GLuint,
    cache: GlCache,
}

impl Context {
    pub fn new() -> Context {
        unsafe {
            let mut default_framebuffer: GLuint = 0;
            glGetIntegerv(
                GL_FRAMEBUFFER_BINDING,
                &mut default_framebuffer as *mut _ as *mut _,
            );
            let mut vao = 0;

            glGenVertexArrays(1, &mut vao as *mut _);
            glBindVertexArray(vao);
            Context {
                default_framebuffer,
                shaders: vec![],
                pipelines: vec![],
                passes: vec![],
                cache: GlCache {
                    stored_index_buffer: 0,
                    stored_index_type: None,
                    stored_vertex_buffer: 0,
                    index_buffer: 0,
                    index_type: None,
                    vertex_buffer: 0,
                    cur_pipeline: None,
                    color_blend: None,
                    alpha_blend: None,
                    stencil: None,
                    color_write: (true, true, true, true),
                    cull_face: CullFace::Nothing,
                    stored_texture: 0,
                    textures: [0; MAX_SHADERSTAGE_IMAGES],
                    attributes: [None; MAX_VERTEX_ATTRIBUTES],
                },
            }
        }
    }

    /// The current framebuffer size in pixels
    /// NOTE: [High DPI Rendering](../conf/index.html#high-dpi-rendering)
    pub fn screen_size(&self) -> (f32, f32) {
        unsafe { (sapp_width() as f32, sapp_height() as f32) }
    }

    /// The dpi scaling factor (window pixels to framebuffer pixels)
    /// NOTE: [High DPI Rendering](../conf/index.html#high-dpi-rendering)
    pub fn dpi_scale(&self) -> f32 {
        unsafe { sapp_dpi_scale() }
    }

    /// True when high_dpi was requested and actually running in a high-dpi scenario
    /// NOTE: [High DPI Rendering](../conf/index.html#high-dpi-rendering)
    pub fn high_dpi(&self) -> bool {
        unsafe { sapp_high_dpi() }
    }

    pub fn apply_pipeline(&mut self, pipeline: &Pipeline) {
        self.cache.cur_pipeline = Some(*pipeline);

        {
            let pipeline = &self.pipelines[pipeline.0];
            let shader = &mut self.shaders[pipeline.shader.0];
            unsafe {
                glUseProgram(shader.program);
            }

            unsafe {
                glEnable(GL_SCISSOR_TEST);
            }

            if pipeline.params.depth_write {
                unsafe {
                    glEnable(GL_DEPTH_TEST);
                    glDepthFunc(pipeline.params.depth_test.into())
                }
            } else {
                unsafe {
                    glDisable(GL_DEPTH_TEST);
                }
            }

            match pipeline.params.front_face_order {
                FrontFaceOrder::Clockwise => unsafe {
                    glFrontFace(GL_CW);
                },
                FrontFaceOrder::CounterClockwise => unsafe {
                    glFrontFace(GL_CCW);
                },
            }
        }

        self.set_cull_face(self.pipelines[pipeline.0].params.cull_face);
        self.set_blend(
            self.pipelines[pipeline.0].params.color_blend,
            self.pipelines[pipeline.0].params.alpha_blend,
        );

        self.set_stencil(self.pipelines[pipeline.0].params.stencil_test);
        self.set_color_write(self.pipelines[pipeline.0].params.color_write);
    }

    pub fn set_cull_face(&mut self, cull_face: CullFace) {
        if self.cache.cull_face == cull_face {
            return;
        }

        match cull_face {
            CullFace::Nothing => unsafe {
                glDisable(GL_CULL_FACE);
            },
            CullFace::Front => unsafe {
                glEnable(GL_CULL_FACE);
                glCullFace(GL_FRONT);
            },
            CullFace::Back => unsafe {
                glEnable(GL_CULL_FACE);
                glCullFace(GL_BACK);
            },
        }
        self.cache.cull_face = cull_face;
    }

    pub fn set_color_write(&mut self, color_write: ColorMask) {
        if self.cache.color_write == color_write {
            return;
        }
        let (r, g, b, a) = color_write;
        unsafe { glColorMask(r as _, g as _, b as _, a as _) }
        self.cache.color_write = color_write;
    }

    pub fn set_blend(&mut self, color_blend: Option<BlendState>, alpha_blend: Option<BlendState>) {
        if color_blend.is_none() && alpha_blend.is_some() {
            panic!("AlphaBlend without ColorBlend");
        }
        if self.cache.color_blend == color_blend && self.cache.alpha_blend == alpha_blend {
            return;
        }

        unsafe {
            if let Some(color_blend) = color_blend {
                if self.cache.color_blend.is_none() {
                    glEnable(GL_BLEND);
                }

                let BlendState {
                    equation: eq_rgb,
                    sfactor: src_rgb,
                    dfactor: dst_rgb,
                } = color_blend;

                if let Some(BlendState {
                    equation: eq_alpha,
                    sfactor: src_alpha,
                    dfactor: dst_alpha,
                }) = alpha_blend
                {
                    glBlendFuncSeparate(
                        src_rgb.into(),
                        dst_rgb.into(),
                        src_alpha.into(),
                        dst_alpha.into(),
                    );
                    glBlendEquationSeparate(eq_rgb.into(), eq_alpha.into());
                } else {
                    glBlendFunc(src_rgb.into(), dst_rgb.into());
                    glBlendEquationSeparate(eq_rgb.into(), eq_rgb.into());
                }
            } else if self.cache.color_blend.is_some() {
                glDisable(GL_BLEND);
            }
        }

        self.cache.color_blend = color_blend;
        self.cache.alpha_blend = alpha_blend;
    }

    pub fn set_stencil(&mut self, stencil_test: Option<StencilState>) {
        if self.cache.stencil == stencil_test {
            return;
        }
        unsafe {
            if let Some(stencil) = stencil_test {
                if self.cache.stencil.is_none() {
                    glEnable(GL_STENCIL_TEST);
                }

                let front = &stencil.front;
                glStencilOpSeparate(
                    GL_FRONT,
                    front.fail_op.into(),
                    front.depth_fail_op.into(),
                    front.pass_op.into(),
                );
                glStencilFuncSeparate(
                    GL_FRONT,
                    front.test_func.into(),
                    front.test_ref,
                    front.test_mask,
                );
                glStencilMaskSeparate(GL_FRONT, front.write_mask);

                let back = &stencil.back;
                glStencilOpSeparate(
                    GL_BACK,
                    back.fail_op.into(),
                    back.depth_fail_op.into(),
                    back.pass_op.into(),
                );
                glStencilFuncSeparate(
                    GL_BACK,
                    back.test_func.into(),
                    back.test_ref.into(),
                    back.test_mask,
                );
                glStencilMaskSeparate(GL_BACK, back.write_mask);
            } else if self.cache.stencil.is_some() {
                glDisable(GL_STENCIL_TEST);
            }
        }

        self.cache.stencil = stencil_test;
    }

    /// Set a new viewport rectangle.
    /// Should be applied after begin_pass.
    pub fn apply_viewport(&mut self, x: i32, y: i32, w: i32, h: i32) {
        unsafe {
            glViewport(x, y, w, h);
        }
    }

    /// Set a new scissor rectangle.
    /// Should be applied after begin_pass.
    pub fn apply_scissor_rect(&mut self, x: i32, y: i32, w: i32, h: i32) {
        unsafe {
            glScissor(x, y, w, h);
        }
    }

    pub fn apply_bindings(&mut self, bindings: &Bindings) {
        let pip = &self.pipelines[self.cache.cur_pipeline.unwrap().0];
        let shader = &self.shaders[pip.shader.0];

        for (n, shader_image) in shader.images.iter().enumerate() {
            let bindings_image = bindings
                .images
                .get(n)
                .unwrap_or_else(|| panic!("Image count in bindings and shader did not match!"));
            if let Some(gl_loc) = shader_image.gl_loc {
                unsafe {
                    self.cache.bind_texture(n, bindings_image.texture);
                    glUniform1i(gl_loc, n as i32);
                }
            }
        }

        self.cache.bind_buffer(
            GL_ELEMENT_ARRAY_BUFFER,
            bindings.index_buffer.gl_buf,
            bindings.index_buffer.index_type,
        );

        let pip = &self.pipelines[self.cache.cur_pipeline.unwrap().0];

        for attr_index in 0..MAX_VERTEX_ATTRIBUTES {
            let cached_attr = &mut self.cache.attributes[attr_index];

            let pip_attribute = pip.layout.get(attr_index).copied();

            if let Some(Some(attribute)) = pip_attribute {
                let vb = bindings.vertex_buffers[attribute.buffer_index];

                if cached_attr.map_or(true, |cached_attr| {
                    attribute != cached_attr.attribute || cached_attr.gl_vbuf != vb.gl_buf
                }) {
                    self.cache
                        .bind_buffer(GL_ARRAY_BUFFER, vb.gl_buf, vb.index_type);

                    unsafe {
                        glVertexAttribPointer(
                            attr_index as GLuint,
                            attribute.size,
                            attribute.type_,
                            GL_FALSE as u8,
                            attribute.stride,
                            attribute.offset as *mut _,
                        );
                        glVertexAttribDivisor(attr_index as GLuint, attribute.divisor as u32);
                        glEnableVertexAttribArray(attr_index as GLuint);
                    };

                    let cached_attr = &mut self.cache.attributes[attr_index];
                    *cached_attr = Some(CachedAttribute {
                        attribute,
                        gl_vbuf: vb.gl_buf,
                    });
                }
            } else {
                if cached_attr.is_some() {
                    unsafe {
                        glDisableVertexAttribArray(attr_index as GLuint);
                    }
                    *cached_attr = None;
                }
            }
        }
    }

    pub fn apply_uniforms<U>(&mut self, uniforms: &U) {
        self.apply_uniforms_from_bytes(uniforms as *const _ as *const u8, std::mem::size_of::<U>())
    }

    #[doc(hidden)]
    /// Apply uniforms data from array of bytes with very special layout.
    /// Hidden because `apply_uniforms` is the recommended and safer way to work with uniforms.
    pub fn apply_uniforms_from_bytes(&mut self, uniform_ptr: *const u8, size: usize) {
        let pip = &self.pipelines[self.cache.cur_pipeline.unwrap().0];
        let shader = &self.shaders[pip.shader.0];

        let mut offset = 0;

        for (_, uniform) in shader.uniforms.iter().enumerate() {
            use UniformType::*;

            assert!(
                offset <= size - uniform.uniform_type.size() / 4,
                "Uniforms struct does not match shader uniforms layout"
            );

            unsafe {
                let data = (uniform_ptr as *const f32).offset(offset as isize);
                let data_int = (uniform_ptr as *const i32).offset(offset as isize);

                if let Some(gl_loc) = uniform.gl_loc {
                    match uniform.uniform_type {
                        Float1 => {
                            glUniform1fv(gl_loc, uniform.array_count, data);
                        }
                        Float2 => {
                            glUniform2fv(gl_loc, uniform.array_count, data);
                        }
                        Float3 => {
                            glUniform3fv(gl_loc, uniform.array_count, data);
                        }
                        Float4 => {
                            glUniform4fv(gl_loc, uniform.array_count, data);
                        }
                        Int1 => {
                            glUniform1iv(gl_loc, uniform.array_count, data_int);
                        }
                        Int2 => {
                            glUniform2iv(gl_loc, uniform.array_count, data_int);
                        }
                        Int3 => {
                            glUniform3iv(gl_loc, uniform.array_count, data_int);
                        }
                        Int4 => {
                            glUniform4iv(gl_loc, uniform.array_count, data_int);
                        }
                        Mat4 => {
                            glUniformMatrix4fv(gl_loc, uniform.array_count, 0, data);
                        }
                    }
                }
            }
            offset += uniform.uniform_type.size() / 4 * uniform.array_count as usize;
        }
    }

    pub fn clear(
        &self,
        color: Option<(f32, f32, f32, f32)>,
        depth: Option<f32>,
        stencil: Option<i32>,
    ) {
        let mut bits = 0;
        if let Some((r, g, b, a)) = color {
            bits |= GL_COLOR_BUFFER_BIT;
            unsafe {
                glClearColor(r, g, b, a);
            }
        }

        if let Some(v) = depth {
            bits |= GL_DEPTH_BUFFER_BIT;
            unsafe {
                glClearDepthf(v);
            }
        }

        if let Some(v) = stencil {
            bits |= GL_STENCIL_BUFFER_BIT;
            unsafe {
                glClearStencil(v);
            }
        }

        if bits != 0 {
            unsafe {
                glClear(bits);
            }
        }
    }

    /// start rendering to the default frame buffer
    pub fn begin_default_pass(&mut self, action: PassAction) {
        self.begin_pass(None, action);
    }

    /// start rendering to an offscreen framebuffer
    pub fn begin_pass(&mut self, pass: impl Into<Option<RenderPass>>, action: PassAction) {
        let (framebuffer, w, h) = match pass.into() {
            None => (
                self.default_framebuffer,
                unsafe { sapp_width() } as i32,
                unsafe { sapp_height() } as i32,
            ),
            Some(pass) => {
                let pass = &self.passes[pass.0];
                (
                    pass.gl_fb,
                    pass.texture.width as i32,
                    pass.texture.height as i32,
                )
            }
        };
        unsafe {
            glBindFramebuffer(GL_FRAMEBUFFER, framebuffer);
            glViewport(0, 0, w, h);
            glScissor(0, 0, w, h);
        }
        match action {
            PassAction::Nothing => {}
            PassAction::Clear {
                color,
                depth,
                stencil,
            } => {
                self.clear(color, depth, stencil);
            }
        }
    }

    pub fn end_render_pass(&mut self) {
        unsafe {
            glBindFramebuffer(GL_FRAMEBUFFER, self.default_framebuffer);
            self.cache.bind_buffer(GL_ARRAY_BUFFER, 0, None);
            self.cache.bind_buffer(GL_ELEMENT_ARRAY_BUFFER, 0, None);
        }
    }

    pub fn commit_frame(&mut self) {
        self.cache.clear_buffer_bindings();
        self.cache.clear_texture_bindings();
    }

    /// Draw elements using currently applied bindings and pipeline.
    ///
    /// + `base_element` specifies starting offset in `index_buffer`.
    /// + `num_elements` specifies length of the slice of `index_buffer` to draw.
    /// + `num_instances` specifies how many instances should be rendered.
    pub fn draw(&self, base_element: i32, num_elements: i32, num_instances: i32) {
        assert!(
            self.cache.cur_pipeline.is_some(),
            "Drawing without any binded pipeline"
        );

        let pip = &self.pipelines[self.cache.cur_pipeline.unwrap().0];
        let primitive_type = pip.params.primitive_type.into();
        let index_type = self.cache.index_type.expect("Unset index buffer type");

        unsafe {
            glDrawElementsInstanced(
                primitive_type,
                num_elements,
                index_type.into(),
                (index_type.size() as i32 * base_element) as *mut _,
                num_instances,
            );
        }
    }
}

#[derive(Debug, PartialEq, Clone, Copy)]
pub struct PipelineParams {
    pub cull_face: CullFace,
    pub front_face_order: FrontFaceOrder,
    pub depth_test: Comparison,
    pub depth_write: bool,
    pub depth_write_offset: Option<(f32, f32)>,
    /// Color (RGB) blend function. If None - blending will be disabled for this pipeline.
    /// Usual use case to get alpha-blending:
    ///```
    ///# use orom_miniquad::{PipelineParams, BlendState, BlendValue, BlendFactor, Equation};
    ///PipelineParams {
    ///    color_blend: Some(BlendState::new(
    ///        Equation::Add,
    ///        BlendFactor::Value(BlendValue::SourceAlpha),
    ///        BlendFactor::OneMinusValue(BlendValue::SourceAlpha))
    ///    ),
    ///    ..Default::default()
    ///};
    ///```
    pub color_blend: Option<BlendState>,
    /// Alpha blend function. If None - alpha will be blended with same equation than RGB colors.
    /// One of possible separate alpha channel blend settings is to avoid blending with WebGl background.
    /// On webgl canvas's resulting alpha channel will be used to blend the whole canvas background.
    /// To avoid modifying only alpha channel, but keep usual transparency:
    ///```
    ///# use orom_miniquad::{PipelineParams, BlendState, BlendValue, BlendFactor, Equation};
    ///PipelineParams {
    ///    color_blend: Some(BlendState::new(
    ///        Equation::Add,
    ///        BlendFactor::Value(BlendValue::SourceAlpha),
    ///        BlendFactor::OneMinusValue(BlendValue::SourceAlpha))
    ///    ),
    ///    alpha_blend: Some(BlendState::new(
    ///        Equation::Add,
    ///        BlendFactor::Zero,
    ///        BlendFactor::One)
    ///    ),
    ///    ..Default::default()
    ///};
    ///```
    /// The same results may be achieved with ColorMask(true, true, true, false)
    pub alpha_blend: Option<BlendState>,
    pub stencil_test: Option<StencilState>,
    pub color_write: ColorMask,
    pub primitive_type: PrimitiveType,
}

#[derive(Copy, Clone, Debug)]
pub struct Pipeline(usize);

impl Default for PipelineParams {
    fn default() -> PipelineParams {
        PipelineParams {
            cull_face: CullFace::Nothing,
            front_face_order: FrontFaceOrder::CounterClockwise,
            depth_test: Comparison::Always, // no depth test,
            depth_write: false,             // no depth write,
            depth_write_offset: None,
            color_blend: None,
            alpha_blend: None,
            stencil_test: None,
            color_write: (true, true, true, true),
            primitive_type: PrimitiveType::Triangles,
        }
    }
}

impl Pipeline {
    pub fn new(
        ctx: &mut Context,
        buffer_layout: &[BufferLayout],
        attributes: &[VertexAttribute],
        shader: Shader,
    ) -> Pipeline {
        Self::with_params(ctx, buffer_layout, attributes, shader, Default::default())
    }

    pub fn with_params(
        ctx: &mut Context,
        buffer_layout: &[BufferLayout],
        attributes: &[VertexAttribute],
        shader: Shader,
        params: PipelineParams,
    ) -> Pipeline {
        #[derive(Clone, Copy, Default)]
        struct BufferCacheData {
            stride: i32,
            offset: i64,
        }

        let mut buffer_cache: Vec<BufferCacheData> =
            vec![BufferCacheData::default(); buffer_layout.len()];

        for VertexAttribute {
            format,
            buffer_index,
            ..
        } in attributes
        {
            let layout = buffer_layout.get(*buffer_index).unwrap_or_else(|| panic!());
            let mut cache = buffer_cache
                .get_mut(*buffer_index)
                .unwrap_or_else(|| panic!());

            if layout.stride == 0 {
                cache.stride += format.byte_len();
            } else {
                cache.stride = layout.stride;
            }
            // WebGL 1 limitation
            assert!(cache.stride <= 255);
        }

        let program = ctx.shaders[shader.0].program;

        let attributes_len = attributes
            .iter()
            .map(|layout| match layout.format {
                VertexFormat::Mat4 => 4,
                _ => 1,
            })
            .sum();

        let mut vertex_layout: Vec<Option<VertexAttributeInternal>> = vec![None; attributes_len];

        for VertexAttribute {
            name,
            format,
            buffer_index,
        } in attributes
        {
            let mut buffer_data = &mut buffer_cache
                .get_mut(*buffer_index)
                .unwrap_or_else(|| panic!());
            let layout = buffer_layout.get(*buffer_index).unwrap_or_else(|| panic!());

            let cname = CString::new(*name).unwrap_or_else(|e| panic!("{}", e));
            let attr_loc = unsafe { glGetAttribLocation(program, cname.as_ptr() as *const _) };
            let attr_loc = if attr_loc == -1 { None } else { Some(attr_loc) };
            let divisor = if layout.step_func == VertexStep::PerVertex {
                0
            } else {
                layout.step_rate
            };

            let mut attributes_count: usize = 1;
            let mut format = *format;

            if format == VertexFormat::Mat4 {
                format = VertexFormat::Float4;
                attributes_count = 4;
            }
            for i in 0..attributes_count {
                if let Some(attr_loc) = attr_loc {
                    let attr_loc = attr_loc as GLuint + i as GLuint;

                    let attr = VertexAttributeInternal {
                        attr_loc,
                        size: format.size(),
                        type_: format.type_(),
                        offset: buffer_data.offset,
                        stride: buffer_data.stride,
                        buffer_index: *buffer_index,
                        divisor,
                    };
                    //println!("{}: {:?}", name, attr);

                    assert!(
                        attr_loc < vertex_layout.len() as u32,
                        "attribute: {} outside of allocated attributes array len: {}",
                        name,
                        vertex_layout.len()
                    );
                    vertex_layout[attr_loc as usize] = Some(attr);
                }
                buffer_data.offset += format.byte_len() as i64
            }
        }

        let pipeline = PipelineInternal {
            layout: vertex_layout,
            shader,
            params,
        };

        ctx.pipelines.push(pipeline);
        Pipeline(ctx.pipelines.len() - 1)
    }

    pub fn set_blend(&self, ctx: &mut Context, color_blend: Option<BlendState>) {
        let mut pipeline = &mut ctx.pipelines[self.0];
        pipeline.params.color_blend = color_blend;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Default)]
struct VertexAttributeInternal {
    attr_loc: GLuint,
    size: i32,
    type_: GLuint,
    offset: i64,
    stride: i32,
    buffer_index: usize,
    divisor: i32,
}

struct PipelineInternal {
    layout: Vec<Option<VertexAttributeInternal>>,
    shader: Shader,
    params: PipelineParams,
}

/// Geometry bindings
#[derive(Clone, Debug)]
pub struct Bindings {
    /// Vertex buffers. Data contained in the buffer must match layout
    /// specified in the `Pipeline`.
    ///
    /// Most commonly vertex buffer will contain `(x,y,z,w)` coordinates of the
    /// vertex in 3d space, as well as `(u,v)` coordinates that map the vertex
    /// to some position in the corresponding `Texture`.
    pub vertex_buffers: Vec<Buffer>,
    /// Index buffer which instructs the GPU in which order to draw vertices
    /// from a vertex buffer, with each subsequent 3 indices forming a
    /// triangle.
    pub index_buffer: Buffer,
    /// Textures to be used with when drawing the geometry in the fragment
    /// shader.
    pub images: Vec<Texture>,
}

impl Drop for Bindings {
    fn drop(&mut self) {
        for buffer in self.vertex_buffers.iter() {
            buffer.delete();
        }
        self.index_buffer.delete();
        for image in self.images.iter() {
            image.delete();
        }
    }
}