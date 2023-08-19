// Compatible with Zig Version 0.11.0
const std = @import("std");
const ArrayList = std.ArrayList;
const Compile = std.Build.Step.Compile;
const ConfigHeader = std.Build.Step.ConfigHeader;
const Mode = std.builtin.Mode;
const CrossTarget = std.zig.CrossTarget;

const Maker = struct {
    builder: *std.build.Builder,
    target: CrossTarget,
    optimize: Mode,
    config_header: *ConfigHeader,
    enable_lto: bool,

    include_dirs: ArrayList([]const u8),
    cflags: ArrayList([]const u8),
    cxxflags: ArrayList([]const u8),
    objs: ArrayList(*Compile),

    fn addInclude(m: *Maker, dir: []const u8) !void {
        try m.include_dirs.append(dir);
    }
    fn addProjectInclude(m: *Maker, path: []const []const u8) !void {
        try m.addInclude(try m.builder.build_root.join(m.builder.allocator, path));
    }
    fn addCFlag(m: *Maker, flag: []const u8) !void {
        try m.cflags.append(flag);
    }
    fn addCxxFlag(m: *Maker, flag: []const u8) !void {
        try m.cxxflags.append(flag);
    }
    fn addFlag(m: *Maker, flag: []const u8) !void {
        try m.addCFlag(flag);
        try m.addCxxFlag(flag);
    }

    fn init(builder: *std.build.Builder) !Maker {
        const commit_hash = @embedFile(".git/refs/heads/master");
        const config_header = builder.addConfigHeader(
            .{ .style = .blank, .include_path = "build-info.h" },
            .{
                .BUILD_NUMBER = 0,
                .BUILD_COMMIT = commit_hash[0 .. commit_hash.len - 1], // omit newline
            },
        );
        var m = Maker{
            .builder = builder,
            .target = builder.standardTargetOptions(.{}),
            .optimize = builder.standardOptimizeOption(.{}),
            .config_header = config_header,
            .enable_lto = false,
            .include_dirs = ArrayList([]const u8).init(builder.allocator),
            .cflags = ArrayList([]const u8).init(builder.allocator),
            .cxxflags = ArrayList([]const u8).init(builder.allocator),
            .objs = ArrayList(*Compile).init(builder.allocator),
        };
        try m.addCFlag("-std=c11");
        try m.addCxxFlag("-std=c++11");
        try m.addProjectInclude(&.{});
        try m.addProjectInclude(&.{"examples"});
        return m;
    }

    fn obj(m: *const Maker, name: []const u8, src: []const u8) *Compile {
        const o = m.builder.addObject(.{ .name = name, .target = m.target, .optimize = m.optimize });
        if (std.mem.endsWith(u8, src, ".c")) {
            o.addCSourceFiles(&.{src}, m.cflags.items);
            o.linkLibC();
        } else {
            o.addCSourceFiles(&.{src}, m.cxxflags.items);
            o.linkLibCpp();
        }
        for (m.include_dirs.items) |i| o.addIncludePath(.{ .path = i });
        o.want_lto = m.enable_lto;
        return o;
    }

    fn exe(m: *const Maker, name: []const u8, src: []const u8, deps: []const *Compile) *Compile {
        const e = m.builder.addExecutable(.{ .name = name, .target = m.target, .optimize = m.optimize });
        e.addCSourceFiles(&.{src}, m.cxxflags.items);
        for (deps) |d| e.addObject(d);
        for (m.objs.items) |o| e.addObject(o);
        for (m.include_dirs.items) |i| e.addIncludePath(.{ .path = i });
        e.linkLibC();
        e.linkLibCpp();
        e.addConfigHeader(m.config_header);
        m.builder.installArtifact(e);
        e.want_lto = m.enable_lto;
        return e;
    }
};

pub fn build(b: *std.build.Builder) !void {
    var make = try Maker.init(b);
    make.enable_lto = b.option(bool, "lto", "Enable LTO optimization, (default: false)") orelse false;

    if (b.option(bool, "k-quants", "Enable K-quants, (default: true)") orelse true) {
        try make.addFlag("-DGGML_USE_K_QUANTS");
        const k_quants = make.obj("k_quants", "k_quants.c");
        try make.objs.append(k_quants);
    }

    const ggml = make.obj("ggml", "ggml.c");
    const ggml_alloc = make.obj("ggml-alloc", "ggml-alloc.c");
    const llama = make.obj("llama", "llama.cpp");
    const common = make.obj("common", "examples/common.cpp");
    const console = make.obj("common", "examples/console.cpp");
    const grammar_parser = make.obj("grammar-parser", "examples/grammar-parser.cpp");

    _ = make.exe("main", "examples/main/main.cpp", &.{ ggml, ggml_alloc, llama, common, console, grammar_parser });
    _ = make.exe("quantize", "examples/quantize/quantize.cpp", &.{ ggml, ggml_alloc, llama });
    _ = make.exe("perplexity", "examples/perplexity/perplexity.cpp", &.{ ggml, ggml_alloc, llama, common });
    _ = make.exe("embedding", "examples/embedding/embedding.cpp", &.{ ggml, ggml_alloc, llama, common });
    _ = make.exe("train-text-from-scratch", "examples/train-text-from-scratch/train-text-from-scratch.cpp", &.{ ggml, ggml_alloc, llama });

    const server = make.exe("server", "examples/server/server.cpp", &.{ ggml, ggml_alloc, llama, common, grammar_parser });
    if (server.target.isWindows()) {
        server.linkSystemLibrary("ws2_32");
    }
}
