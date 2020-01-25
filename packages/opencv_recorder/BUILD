load("//engine/build:isaac.bzl", "isaac_pkg", "isaac_subgraph")

py_binary(
    name = "opencv_recorder",
    srcs = [
        "__init__.py",
        "opencv_recorder.py",
    ],
    data = [
        "opencv_recorder.app.json",
        "//packages:py_init",
        "//packages/navsim/apps:navsim_subgraph",
        "//packages/viewers:libviewers_module.so",
    ],
    deps = [
        "//engine/pyalice",
    ],
    
    visibility = ["//visibility:public"],
)

isaac_subgraph(
    name = "opencv_recorder_subgraph",
    modules = [
    ],
    subgraph = "opencv_recorder.subgraph.json",
    visibility = ["//visibility:public"],
)

filegroup(
    name = "py_init",
    data = ["__init__.py"],
    visibility = ["//visibility:public"],
)

isaac_pkg(
    name = "opencv_recorder-pkg",
    srcs = ["opencv_recorder", "opencv_recorder_subgraph"],
)
