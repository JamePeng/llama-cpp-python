cmake_minimum_required(VERSION 3.21)

project(llama_cpp)

option(LLAMA_BUILD "Build llama.cpp shared library and install alongside python package" ON)
option(MTMD_BUILD "Build mtmd shared library and install alongside python package" ON)


set(CMAKE_INSTALL_BINDIR llama_cpp/bin CACHE PATH "" FORCE)
set(CMAKE_INSTALL_LIBDIR llama_cpp/lib CACHE PATH "" FORCE)
set(CMAKE_INSTALL_INCLUDEDIR llama_cpp/include CACHE PATH "" FORCE)


# Install a built target into the Python package runtime directory.
function(llama_cpp_python_install_target target)
    if(NOT TARGET ${target})
        return()
    endif()

    set(INSTALL_DIRS
        "${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp/lib"
        "${SKBUILD_PLATLIB_DIR}/llama_cpp/lib"
    )

    foreach(DIR ${INSTALL_DIRS})
        install(
            TARGETS ${target}
            LIBRARY DESTINATION ${DIR}
            RUNTIME DESTINATION ${DIR}
            ARCHIVE DESTINATION ${DIR}
            FRAMEWORK DESTINATION ${DIR}
            RESOURCE DESTINATION ${DIR}
        )

        # Copy runtime DLL dependencies of this target when available.
        # This does not replace explicit installation of dynamic backend
        # targets such as ggml-cpu-*; those are installed as targets below.
        # Automatically handle Windows DLL installation for each target
        if (WIN32)
            install(
                FILES $<TARGET_RUNTIME_DLLS:${target}>
                DESTINATION ${DIR}
                OPTIONAL # Prevent errors if the target has no DLLs
            )
        endif()
    endforeach()

    # Proper RPATH handling
    if(UNIX)
        set(INSTALL_RPATH_VAL "$ORIGIN")
        if(APPLE)
            set(INSTALL_RPATH_VAL "@loader_path")
        endif()

        set_target_properties(${target} PROPERTIES
            INSTALL_RPATH "${INSTALL_RPATH_VAL}"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    endif()
endfunction()


# Remove development-only artifacts from Python wheel runtime directories.
#
# Upstream install rules may place CMake package files, pkg-config files, and
# Windows import libraries under llama_cpp/lib because CMAKE_INSTALL_LIBDIR is
# redirected there for wheel builds. They are not needed at runtime.
function(llama_cpp_python_cleanup_dev_files)
    if(NOT WIN32)
        return()
    endif()

    set(INSTALL_DIRS
        "${CMAKE_CURRENT_SOURCE_DIR}/llama_cpp/lib"
        "${SKBUILD_PLATLIB_DIR}/llama_cpp/lib"
    )

    foreach(DIR ${INSTALL_DIRS})
        install(CODE "
            if(EXISTS \"${DIR}\")
                file(GLOB LLAMA_CPP_IMPORT_LIBS \"${DIR}/*.lib\")
                if(LLAMA_CPP_IMPORT_LIBS)
                    file(REMOVE \${LLAMA_CPP_IMPORT_LIBS})
                endif()

                file(REMOVE_RECURSE
                    \"${DIR}/cmake\"
                    \"${DIR}/pkgconfig\"
                )
            endif()
        ")
    endforeach()
endfunction()


if (LLAMA_BUILD)
    set(BUILD_SHARED_LIBS "On")

    set(CMAKE_SKIP_BUILD_RPATH FALSE)

    # When building, don't use the install RPATH already
    # (but later on when installing)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
 
    # Add the automatically determined parts of the RPATH
    # which point to directories outside the build tree to the install RPATH
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    set(CMAKE_SKIP_RPATH FALSE)

    # Enable building of the common library
    set(LLAMA_BUILD_COMMON ON CACHE BOOL "llama: build common utils library" FORCE)

    # Enable build and link OpenSSL
    set(LLAMA_OPENSSL ON CACHE BOOL "llama: use openssl to support HTTPS" FORCE)

    # Disable building of examples
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "llama: build examples" FORCE)

    # Disable building of tests
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "llama: build tests" FORCE)

    # Disable building of server
    set(LLAMA_BUILD_SERVER OFF CACHE BOOL "llama: build server example" FORCE)

    # Disable build the embedded Web UI for server
    set(LLAMA_BUILD_UI OFF CACHE BOOL "llama: build the embedded Web UI for server" FORCE)
    set(LLAMA_USE_PREBUILT_UI OFF CACHE BOOL "llama: use prebuilt UI from HF Bucket when available (requires LLAMA_BUILD_UI=ON)" FORCE)

    # Disable building curl support
    set(LLAMA_CURL OFF CACHE BOOL "llama.cpp: use libcurl to download model from an URL" FORCE)

    # Architecture detection and settings for Apple platforms
    if (APPLE)
        # If CMAKE_OSX_ARCHITECTURES is not set, use the host architecture
        if(NOT CMAKE_OSX_ARCHITECTURES)
            set(CMAKE_OSX_ARCHITECTURES ${CMAKE_HOST_SYSTEM_PROCESSOR} CACHE STRING "Build architecture for macOS" FORCE)
        endif()

        message(STATUS "Host architecture: ${CMAKE_HOST_SYSTEM_PROCESSOR}")
        message(STATUS "Target architecture: ${CMAKE_OSX_ARCHITECTURES}")

        # Configure based on target architecture
        if(CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
            # Intel Mac settings
            set(GGML_AVX "OFF" CACHE BOOL "ggml: enable AVX" FORCE)
            set(GGML_AVX2 "OFF" CACHE BOOL "ggml: enable AVX2" FORCE)
            set(GGML_FMA "OFF" CACHE BOOL "ggml: enable FMA" FORCE)
            set(GGML_F16C "OFF" CACHE BOOL "ggml: enable F16C" FORCE)
        endif()

        # Metal settings (enable for both architectures)
        set(GGML_METAL "ON" CACHE BOOL "ggml: enable Metal" FORCE)
        set(GGML_METAL_USE_BF16 "ON" CACHE BOOL "ggml: use bfloat if available" FORCE)
        set(GGML_METAL_EMBED_LIBRARY "ON" CACHE BOOL "ggml: embed metal library" FORCE)
    endif()

    add_subdirectory(vendor/llama.cpp)

    if (WIN32 AND TARGET llama)
        set_target_properties(llama PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
    endif()

    # Define list of LLAMA_CPP/GGML targets to install
    set(LLAMA_CPP_TARGETS
        llama
        llama-common
    )
    set(GGML_CORE_TARGETS
        ggml
        ggml-base
        ggml-blas
        ggml-cpu
        ggml-rpc
    )

    set(GGML_CPU_VARIANT_TARGETS
        ggml-cpu-x64
        ggml-cpu-sse42
        ggml-cpu-sandybridge
        ggml-cpu-ivybridge
        ggml-cpu-piledriver
        ggml-cpu-haswell
        ggml-cpu-skylakex
        ggml-cpu-cannonlake
        ggml-cpu-cascadelake
        ggml-cpu-cooperlake
        ggml-cpu-icelake
        ggml-cpu-alderlake
        ggml-cpu-sapphirerapids
        ggml-cpu-zen4
    )

    set(GGML_BACKEND_TARGETS
        ggml-cann
        ggml-cuda
        ggml-hexagon
        ggml-hip
        ggml-metal
        ggml-musa
        ggml-opencl
        ggml-openvino
        ggml-sycl
        ggml-virtgpu
        ggml-vulkan
        ggml-webgpu
        ggml-zdnn
        ggml-zendnn
    )

    foreach(TARGET_NAME
        ${LLAMA_CPP_TARGETS}
        ${GGML_CORE_TARGETS}
        ${GGML_CPU_VARIANT_TARGETS}
        ${GGML_BACKEND_TARGETS}
    )
        llama_cpp_python_install_target(${TARGET_NAME})
    endforeach()

    if (MTMD_BUILD)
        if (NOT DEFINED LLAMA_BUILD_NUMBER)
            set(LLAMA_BUILD_NUMBER        ${BUILD_NUMBER})
        endif()

        set(LLAMA_INSTALL_VERSION 0.0.${LLAMA_BUILD_NUMBER})

        add_subdirectory(vendor/llama.cpp/tools/mtmd)

        if (LLAMA_CUBLAS OR LLAMA_CUDA)
            add_compile_definitions(GGML_USE_CUBLAS)
            add_compile_definitions(GGML_USE_CUDA)
        endif()

        if (LLAMA_METAL)
            add_compile_definitions(GGML_USE_METAL)
        endif()

        if (WIN32)
            set_target_properties(mtmd PROPERTIES CUDA_ARCHITECTURES OFF)
        endif()

        llama_cpp_python_install_target(mtmd)
    endif()

    # Run after all runtime targets are installed, including mtmd.
    llama_cpp_python_cleanup_dev_files()

endif()
