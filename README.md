# OROM Miniquad

This is a fork of [Miniquad](https://github.com/not-fl3/miniquad) made specifically for the needs of Open Rage Of Mages project. It's intended to remove web targets, as well as mobile, and provide reacher support for modern desktop target features. It's not intended to be backward compatible with Miniquad

API is highly inspired by [sokol-gfx](https://github.com/floooh/sokol) ([sokol overview](https://floooh.github.io/2017/07/29/sokol-gfx-tour.html), [2019 update](https://floooh.github.io/2019/01/12/sokol-apply-pipeline.html)). Implementation influenced by [crayon](https://docs.rs/crayon/0.7.1/crayon/video/index.html).

For context management and input "sokol-app" was used.

## Supported platforms

* Windows, OpenGl 3
* Linux, OpenGl 3
* macOS, OpenGL 3

# Building examples

## linux

```bash
# ubuntu system dependencies
apt install libx11-dev libxi-dev libgl1-mesa-dev

cargo run --example quad
```

## windows

```bash
# both MSVC and GNU target is supported:
rustup target add x86_64-pc-windows-msvc
# or
rustup target add x86_64-pc-windows-gnu

cargo run --example quad
```

# Goals

* Fast compilation time. Right now it is ~5s from "cargo clean".

* Cross platform. Amount of platform specific user code required should be kept as little as possible.

* Hackability. Working on your own game, highly probable some hardware incompability will be found. Working around that kind of bugs should be easy, implementation details should not be hidden under layers of abstraction.

# Non goals

* Ultimate type safety. Library should be entirely safe in Rust's definition of safe - no UB or memory unsafety. But correct GPU state is not type guaranteed. Feel free to provide safety abstraction in the user code then!

* sokol-gfx api compatibility. While sokol is absolutely great as an API design foundation, just reimplementing sokol in rust is not a goal. The idea is to learn from sokol, but make a library in a rust way when it is possible.
