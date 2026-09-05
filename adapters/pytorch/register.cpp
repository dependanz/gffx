/* Private PyTorch Stable-ABI loader and operation registration.
 *
 * Phase 3 step 1 registers the first advertised operation, mesh.face_geometry, against the
 * LibTorch Stable ABI. Behaviour is specified by TORCH_ADAPTER_ACCEPTANCE_V0_1.md in the project
 * record; FACE_GEOMETRY_ACCEPTANCE_V0_1.md remains authoritative for what the kernel computes and
 * nothing here reinterprets it.
 *
 * Division of labour with the Python layer. This translation unit does the conversion and the
 * kernel call; src/gffx/torch/mesh.py does argument validation and exception typing. That split is
 * not arbitrary: the Stable ABI offers no way to raise a specific Python exception type, so a
 * failure here is thrown as a std::runtime_error carrying a machine-readable prefix that the
 * Python layer maps onto the exception table in section 5 of the record. Validation that can be
 * performed before the call is therefore performed in Python, where the correct exception can be
 * raised directly and the message can be written for a human.
 *
 * Autograd is registered in Python through torch.autograd.Function rather than here, because
 * torch::autograd::Function is not part of the Stable ABI. That is a supported custom-operation
 * mechanism rather than monkey-patching. Its interaction with torch.compile and torch.export is
 * Phase 3 step 4 and is deliberately not pre-decided here.
 *
 * Tensors are converted by borrowing. No input is copied, no torch memory is freed, and no pointer
 * outlives the call.
 */

#include <Python.h>

#if defined(_WIN32)
/* Declared rather than included. <windows.h> defines macros that collide with the standard
 * library headers torch pulls in, and this needs exactly two functions. */
extern "C" __declspec(dllimport) void *__stdcall GetModuleHandleA(const char *name);
extern "C" __declspec(dllimport) void *__stdcall GetProcAddress(void *module, const char *name);
#else
#include <dlfcn.h>
#endif

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

#include <gffx/execution.h>
#include <gffx/io.h>
#include <gffx/mesh.h>
#include <gffx/points.h>
#include <gffx/render.h>
#include <gffx/transforms.h>
#include <gffx/status.h>
#include <gffx/tensor.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

using torch::headeronly::ScalarType;
using torch::stable::Tensor;

namespace {

/* A kernel failure is thrown with a machine-readable prefix so the Python layer can map the ABI
 * status onto the documented exception type without parsing prose. The diagnostic text is always
 * appended, so it reaches the user's message rather than being discarded. */
[[noreturn]] void raise_status(gffx_status status, const char *diagnostic) {
    std::string message = "gffx-status:";
    message += std::to_string(static_cast<unsigned long>(status));
    message += ": ";
    message += (diagnostic != nullptr && diagnostic[0] != '\0')
        ? diagnostic
        : "the operation reported no diagnostic";
    throw std::runtime_error(message);
}

/* Conditions the Python layer is expected to have rejected already. Reaching one means the two
 * layers disagree, which is an adapter defect rather than a caller error, and is reported as such
 * rather than silently accommodated. */
[[noreturn]] void raise_internal(const char *detail) {
    std::string message = "gffx-internal: ";
    message += detail;
    throw std::runtime_error(message);
}

/*
 * The caller's CUDA stream.
 *
 * `API_CONTRACT_V0_1.md` requires a CUDA adapter to enqueue on the stream the framework is using,
 * not on a stream of its own, so the raw handle is needed rather than an identifier. The Stable
 * ABI does not offer one: `torch::stable::accelerator::getCurrentStream` returns a `StreamId`, an
 * integer whose conversion back to a `cudaStream_t` lives in ATen internals that the stable surface
 * deliberately excludes. The only C entry point that returns the handle,
 * `aoti_torch_get_current_cuda_stream`, is declared behind `#ifdef USE_CUDA` and lives in the
 * framework's CUDA library.
 *
 * Linking it is not an option. This extension is built once and must load against both CPU-only
 * and CUDA builds of PyTorch, a portability property already verified by running the whole suite
 * against `2.10.0+cpu` and `2.10.0+cu128` with the same binary; a link-time reference to a
 * CUDA-only symbol would break that for every CPU user.
 *
 * So the symbol is resolved at run time from whatever is already loaded in the process, and its
 * absence is reported rather than worked around. Falling back to the default stream would enqueue
 * a caller's work on a stream they did not choose, which reorders against their other work
 * silently; that is precisely the class of hidden behaviour this project refuses, so an
 * unavailable stream accessor is a refusal rather than a guess.
 */
using gffx_current_cuda_stream_fn = int (*)(int32_t, void **);

gffx_current_cuda_stream_fn resolve_current_cuda_stream() {
    static gffx_current_cuda_stream_fn resolved = nullptr;
    static bool attempted = false;
    if (attempted) return resolved;
    attempted = true;
#if defined(_WIN32)
    /* Already-loaded modules only: importing torch is the caller's business, and this must never
     * cause a library to load as a side effect of asking a question. */
    static const char *const modules[] = {"torch_cuda.dll", "torch_cpu.dll", "torch.dll"};
    for (const char *name : modules) {
        void *handle = GetModuleHandleA(name);
        if (handle == nullptr) continue;
        auto symbol = reinterpret_cast<gffx_current_cuda_stream_fn>(
            GetProcAddress(handle, "aoti_torch_get_current_cuda_stream"));
        if (symbol != nullptr) { resolved = symbol; break; }
    }
#else
    resolved = reinterpret_cast<gffx_current_cuda_stream_fn>(
        dlsym(RTLD_DEFAULT, "aoti_torch_get_current_cuda_stream"));
#endif
    return resolved;
}

gffx_execution_context cuda_context(int32_t device_index) {
    gffx_execution_context context{};
    context.struct_size = static_cast<uint32_t>(sizeof(context));
    context.abi_version = GFFX_ABI_VERSION;
    context.device_type = GFFX_DEVICE_CUDA;
    context.device_index = device_index;

    gffx_current_cuda_stream_fn accessor = resolve_current_cuda_stream();
    if (accessor == nullptr) {
        raise_internal(
            "this PyTorch build does not expose aoti_torch_get_current_cuda_stream, so the "
            "caller's CUDA stream cannot be honoured. Running on a stream the caller did not "
            "choose would silently reorder their work, so the call is refused instead");
    }
    void *stream = nullptr;
    if (accessor(device_index, &stream) != 0) {
        raise_internal("could not read the current CUDA stream from the framework");
    }
    context.stream = stream;
    return context;
}

/* Selects the context for whichever device the caller's tensors live on.
 *
 * The CPU and CUDA paths of every operation differ in exactly one thing, the execution context, so
 * the implementations take the device from their inputs rather than existing twice. Duplicating
 * seven function bodies to change one line would have been seven places for them to drift apart.
 */
gffx_execution_context context_for(const Tensor &reference);

gffx_execution_context cpu_context() {
    gffx_execution_context context{};
    context.struct_size = static_cast<uint32_t>(sizeof(context));
    context.abi_version = GFFX_ABI_VERSION;
    context.device_type = GFFX_DEVICE_CPU;
    context.device_index = 0;
    return context;
}

gffx_dtype dtype_of(const Tensor &tensor) {
    switch (tensor.scalar_type()) {
        case ScalarType::Float: return GFFX_DTYPE_FLOAT32;
        case ScalarType::Double: return GFFX_DTYPE_FLOAT64;
        case ScalarType::Int: return GFFX_DTYPE_INT32;
        case ScalarType::Bool: return GFFX_DTYPE_BOOL;
        case ScalarType::UInt32: return GFFX_DTYPE_UINT32;
        default: raise_internal("an unsupported tensor dtype reached the adapter");
    }
}

/* Borrows the tensor's storage. The shape and stride arrays must outlive the call, which is why
 * the caller owns them as locals rather than this returning them. Strides are in elements in both
 * torch and the GFFX ABI, so they transfer without scaling. */
gffx_tensor_view borrow(
    const Tensor &tensor, uint32_t flags, int64_t *shape, int64_t *strides, bool writable
) {
    const auto sizes = tensor.sizes();
    const auto tensor_strides = tensor.strides();
    const int64_t rank = tensor.dim();
    for (int64_t index = 0; index < rank; ++index) {
        shape[index] = sizes[index];
        strides[index] = tensor_strides[index];
    }
    gffx_tensor_view view{};
    view.struct_size = static_cast<uint32_t>(sizeof(view));
    view.abi_version = GFFX_ABI_VERSION;
    view.data = writable ? tensor.mutable_data_ptr()
                         : const_cast<void *>(tensor.const_data_ptr());
    view.rank = static_cast<uint32_t>(rank);
    view.shape = shape;
    view.strides = strides;
    view.dtype = dtype_of(tensor);
    /* The view follows the tensor rather than assuming a device, which is what lets one borrow
     * helper serve both backends. */
    view.device_type = tensor.is_cuda() ? GFFX_DEVICE_CUDA : GFFX_DEVICE_CPU;
    view.device_index = tensor.is_cuda() ? tensor.get_device_index() : 0;
    view.flags = flags;
    return view;
}

/* Shape and stride storage for borrowed views.
 *
 * A gffx_tensor_view holds pointers to its shape and stride arrays rather than copying them, so
 * that storage must outlive the kernel call. Slots live in a fixed array rather than a growing
 * vector because a reallocation would invalidate pointers already handed to the ABI, which is the
 * kind of defect that shows up as corrupted geometry rather than as a crash.
 */
class ViewArena {
public:
    gffx_tensor_view read(const Tensor &tensor) {
        return take(tensor, GFFX_TENSOR_READ_ONLY, false);
    }
    gffx_tensor_view write(const Tensor &tensor) {
        return take(tensor, GFFX_TENSOR_OUTPUT, true);
    }

private:
    static constexpr int kMaxViews = 24;
    static constexpr int kMaxRank = 6;
    int64_t shapes_[kMaxViews][kMaxRank]{};
    int64_t strides_[kMaxViews][kMaxRank]{};
    int used_ = 0;

    gffx_tensor_view take(const Tensor &tensor, uint32_t flags, bool writable) {
        if (used_ >= kMaxViews) {
            raise_internal("more borrowed views in one call than the arena holds");
        }
        if (tensor.dim() > kMaxRank) {
            raise_internal("a tensor rank exceeded what the arena holds");
        }
        const int slot = used_++;
        return borrow(tensor, flags, shapes_[slot], strides_[slot], writable);
    }
};

/* The workspace is a torch uint8 tensor, so every allocation stays on the caller's allocator and
 * the functional surface's internal scratch is the same object type the streaming surface asks a
 * caller to supply. */
Tensor make_workspace(
    const Tensor &reference, int64_t vertex_count, int64_t face_count, gffx_dtype dtype,
    uint64_t *out_bytes
) {
    /* The context follows the reference tensor. Asking with a CPU context while running on the
     * GPU would return the host requirement, which is zero for this operation, and the device
     * path would then fail its own workspace check for a reason that looks unrelated. */
    gffx_execution_context context = context_for(reference);
    gffx_diagnostic_buffer diagnostic{};
    uint64_t required_bytes = 0;
    uint64_t required_alignment = 0;
    diagnostic.struct_size = static_cast<uint32_t>(sizeof(diagnostic));
    diagnostic.abi_version = GFFX_ABI_VERSION;

    const gffx_status status = gffx_mesh_face_geometry_workspace(
        vertex_count, face_count, dtype, &context, &required_bytes, &required_alignment,
        &diagnostic);
    if (status != GFFX_STATUS_OK) {
        raise_status(status, "");
    }
    *out_bytes = required_bytes;
    const std::vector<int64_t> size{static_cast<int64_t>(required_bytes)};
    return torch::stable::new_empty(reference, size, ScalarType::Byte);
}

gffx_buffer workspace_buffer(const Tensor &workspace, uint64_t bytes) {
    gffx_buffer buffer{};
    buffer.struct_size = static_cast<uint32_t>(sizeof(buffer));
    buffer.abi_version = GFFX_ABI_VERSION;
    buffer.data = bytes > 0 ? workspace.mutable_data_ptr() : nullptr;
    buffer.capacity_bytes = bytes;
    buffer.alignment = 8;
    buffer.device_type = workspace.is_cuda() ? GFFX_DEVICE_CUDA : GFFX_DEVICE_CPU;
    buffer.device_index = workspace.is_cuda() ? workspace.get_device_index() : 0;
    return buffer;
}

/* Shared by the functional and the preallocated surfaces so the two cannot drift: the streaming
 * entry point is the same kernel call with the allocation removed. */
void run_forward(
    const Tensor &vertices, const Tensor &faces, double eps,
    Tensor &normals, Tensor &areas, Tensor &valid, const gffx_buffer *workspace
) {
    gffx_execution_context context = context_for(vertices);
    gffx_diagnostic_buffer diagnostic{};
    std::string message(512, '\0');
    int64_t vertex_shape[2], vertex_strides[2];
    int64_t face_shape[2], face_strides[2];
    int64_t normal_shape[2], normal_strides[2];
    int64_t area_shape[1], area_strides[1];
    int64_t valid_shape[1], valid_strides[1];

    diagnostic.struct_size = static_cast<uint32_t>(sizeof(diagnostic));
    diagnostic.abi_version = GFFX_ABI_VERSION;
    diagnostic.data = &message[0];
    diagnostic.capacity_bytes = message.size();

    gffx_tensor_view vertices_view =
        borrow(vertices, GFFX_TENSOR_READ_ONLY, vertex_shape, vertex_strides, false);
    gffx_tensor_view faces_view =
        borrow(faces, GFFX_TENSOR_READ_ONLY, face_shape, face_strides, false);
    gffx_tensor_view normals_view =
        borrow(normals, GFFX_TENSOR_OUTPUT, normal_shape, normal_strides, true);
    gffx_tensor_view areas_view =
        borrow(areas, GFFX_TENSOR_OUTPUT, area_shape, area_strides, true);
    gffx_tensor_view valid_view =
        borrow(valid, GFFX_TENSOR_OUTPUT, valid_shape, valid_strides, true);

    const gffx_status status = gffx_mesh_face_geometry(
        &vertices_view, &faces_view, eps, &context, &normals_view, &areas_view, &valid_view,
        workspace, &diagnostic);
    if (status != GFFX_STATUS_OK) {
        raise_status(status, message.c_str());
    }
}

std::tuple<Tensor, Tensor, Tensor> face_geometry(
    const Tensor &vertices, const Tensor &faces, double eps
) {
    const int64_t vertex_count = vertices.size(0);
    const int64_t face_count = faces.size(0);
    const gffx_dtype dtype = dtype_of(vertices);

    const std::vector<int64_t> normal_size{face_count, 3};
    const std::vector<int64_t> scalar_size{face_count};
    Tensor normals = torch::stable::new_empty(vertices, normal_size);
    Tensor areas = torch::stable::new_empty(vertices, scalar_size);
    Tensor valid = torch::stable::new_empty(vertices, scalar_size, ScalarType::Bool);

    uint64_t workspace_bytes = 0;
    Tensor workspace =
        make_workspace(vertices, vertex_count, face_count, dtype, &workspace_bytes);
    const gffx_buffer buffer = workspace_buffer(workspace, workspace_bytes);

    run_forward(vertices, faces, eps, normals, areas, valid,
                workspace_bytes > 0 ? &buffer : nullptr);
    return std::make_tuple(normals, areas, valid);
}

/* Cotangent presence is passed as explicit flags rather than as optional tensors. The ABI
 * expresses an absent cotangent as a null view, and a flag maps onto that directly, whereas a
 * zero-filled tensor would be a different statement: it asserts a cotangent of zero rather than
 * the absence of one. */
Tensor face_geometry_backward(
    const Tensor &vertices, const Tensor &faces, double eps,
    const Tensor &grad_normals, const Tensor &grad_areas,
    bool has_grad_normals, bool has_grad_areas
) {
    gffx_execution_context context = context_for(vertices);
    gffx_diagnostic_buffer diagnostic{};
    std::string message(512, '\0');
    int64_t vertex_shape[2], vertex_strides[2];
    int64_t face_shape[2], face_strides[2];
    int64_t grad_normal_shape[2], grad_normal_strides[2];
    int64_t grad_area_shape[1], grad_area_strides[1];
    int64_t out_shape[2], out_strides[2];

    if (!has_grad_normals && !has_grad_areas) {
        raise_internal("the backward pass requires at least one cotangent");
    }

    diagnostic.struct_size = static_cast<uint32_t>(sizeof(diagnostic));
    diagnostic.abi_version = GFFX_ABI_VERSION;
    diagnostic.data = &message[0];
    diagnostic.capacity_bytes = message.size();

    const std::vector<int64_t> grad_size{vertices.size(0), 3};
    Tensor grad_vertices = torch::stable::new_empty(vertices, grad_size);

    gffx_tensor_view vertices_view =
        borrow(vertices, GFFX_TENSOR_READ_ONLY, vertex_shape, vertex_strides, false);
    gffx_tensor_view faces_view =
        borrow(faces, GFFX_TENSOR_READ_ONLY, face_shape, face_strides, false);
    gffx_tensor_view grad_normals_view =
        borrow(grad_normals, GFFX_TENSOR_READ_ONLY, grad_normal_shape, grad_normal_strides,
               false);
    gffx_tensor_view grad_areas_view =
        borrow(grad_areas, GFFX_TENSOR_READ_ONLY, grad_area_shape, grad_area_strides, false);
    gffx_tensor_view grad_vertices_view =
        borrow(grad_vertices, GFFX_TENSOR_OUTPUT, out_shape, out_strides, true);

    uint64_t workspace_bytes = 0;
    Tensor workspace = make_workspace(vertices, vertices.size(0), faces.size(0),
                                      dtype_of(vertices), &workspace_bytes);
    const gffx_buffer buffer = workspace_buffer(workspace, workspace_bytes);

    const gffx_status status = gffx_mesh_face_geometry_backward(
        &vertices_view, &faces_view, eps,
        has_grad_normals ? &grad_normals_view : nullptr,
        has_grad_areas ? &grad_areas_view : nullptr,
        &context, &grad_vertices_view,
        workspace_bytes > 0 ? &buffer : nullptr, &diagnostic);
    if (status != GFFX_STATUS_OK) {
        raise_status(status, message.c_str());
    }
    return grad_vertices;
}

/* The inference-only surface: the same kernel, writing into buffers the caller already owns. */
void face_geometry_out(
    const Tensor &vertices, const Tensor &faces, double eps,
    Tensor &normals, Tensor &areas, Tensor &valid, Tensor &workspace
) {
    const gffx_buffer buffer =
        workspace_buffer(workspace, static_cast<uint64_t>(workspace.numel()));
    run_forward(vertices, faces, eps, normals, areas, valid,
                workspace.numel() > 0 ? &buffer : nullptr);
}


/* ------------------------------------------------------------------------------------------
 * The remaining ten primitives.
 *
 * Each follows the pattern the first operation established: borrow inputs, allocate outputs on
 * the caller's allocator, query and allocate the workspace, call the kernel, and translate a
 * non-OK status into a prefixed throw the Python layer maps. Validation stays in Python for the
 * reason given at the top of this file, so these functions assume shapes and dtypes have already
 * been checked and report a disagreement as internal rather than accommodating it.
 * ------------------------------------------------------------------------------------------ */

/* A workspace sized by an operation's own query. Each query has a different signature, so the
 * caller passes the already-computed byte count; this only allocates and describes it. */
Tensor workspace_of(const Tensor &reference, uint64_t bytes) {
    const std::vector<int64_t> size{static_cast<int64_t>(bytes)};
    return torch::stable::new_empty(reference, size, ScalarType::Byte);
}

#define GFFX_CHECK(call, message)                    \
    do {                                             \
        const gffx_status status_ = (call);          \
        if (status_ != GFFX_STATUS_OK) {             \
            raise_status(status_, (message));        \
        }                                            \
    } while (0)

/* A diagnostic buffer plus its storage, so every call reports a message rather than an empty
 * string. Declared as one object because the two must stay together. */
struct Diagnostic {
    std::string storage;
    gffx_diagnostic_buffer buffer{};

    Diagnostic() : storage(512, '\0') {
        buffer.struct_size = static_cast<uint32_t>(sizeof(buffer));
        buffer.abi_version = GFFX_ABI_VERSION;
        buffer.data = &storage[0];
        buffer.capacity_bytes = storage.size();
    }
    const char *text() const { return storage.c_str(); }
};

/* ------------------------------------------------------------------- mesh.vertex_normals */

Tensor vertex_normals(
    const Tensor &vertices, const Tensor &faces, double eps, int64_t weighting
) {
    ViewArena arena;
    Diagnostic diagnostic;
    gffx_execution_context context = context_for(vertices);
    uint64_t bytes = 0, alignment = 0;

    GFFX_CHECK(gffx_mesh_vertex_normals_workspace(
                   vertices.size(0), faces.size(0), dtype_of(vertices), &context, &bytes,
                   &alignment, &diagnostic.buffer),
               diagnostic.text());
    Tensor workspace = workspace_of(vertices, bytes);
    const gffx_buffer buffer = workspace_buffer(workspace, bytes);

    const std::vector<int64_t> size{vertices.size(0), 3};
    Tensor normals = torch::stable::new_empty(vertices, size);

    gffx_tensor_view vertices_view = arena.read(vertices);
    gffx_tensor_view faces_view = arena.read(faces);
    gffx_tensor_view normals_view = arena.write(normals);
    GFFX_CHECK(gffx_mesh_vertex_normals(
                   &vertices_view, &faces_view, eps, static_cast<uint32_t>(weighting), &context,
                   &normals_view, bytes > 0 ? &buffer : nullptr, &diagnostic.buffer),
               diagnostic.text());
    return normals;
}

Tensor vertex_normals_backward(
    const Tensor &vertices, const Tensor &faces, double eps, int64_t weighting,
    const Tensor &grad_normals
) {
    ViewArena arena;
    Diagnostic diagnostic;
    gffx_execution_context context = context_for(vertices);
    uint64_t bytes = 0, alignment = 0;

    GFFX_CHECK(gffx_mesh_vertex_normals_workspace(
                   vertices.size(0), faces.size(0), dtype_of(vertices), &context, &bytes,
                   &alignment, &diagnostic.buffer),
               diagnostic.text());
    Tensor workspace = workspace_of(vertices, bytes);
    const gffx_buffer buffer = workspace_buffer(workspace, bytes);

    const std::vector<int64_t> size{vertices.size(0), 3};
    Tensor grad_vertices = torch::stable::new_empty(vertices, size);

    gffx_tensor_view vertices_view = arena.read(vertices);
    gffx_tensor_view faces_view = arena.read(faces);
    gffx_tensor_view grad_normals_view = arena.read(grad_normals);
    gffx_tensor_view grad_vertices_view = arena.write(grad_vertices);
    GFFX_CHECK(gffx_mesh_vertex_normals_backward(
                   &vertices_view, &faces_view, eps, static_cast<uint32_t>(weighting),
                   &grad_normals_view, &context, &grad_vertices_view,
                   bytes > 0 ? &buffer : nullptr, &diagnostic.buffer),
               diagnostic.text());
    return grad_vertices;
}

/* --------------------------------------------------------------------- mesh.gather_faces */

Tensor gather_faces(const Tensor &vertices, const Tensor &faces) {
    ViewArena arena;
    Diagnostic diagnostic;
    gffx_execution_context context = context_for(vertices);
    uint64_t bytes = 0, alignment = 0;

    /* The workspace is queried rather than assumed absent. This operation needs none on the CPU,
     * but the CUDA backend needs a device word to report whether the index-validation kernel found
     * an out-of-range face, since host code cannot read device-resident indices. Passing null
     * unconditionally is what made the CUDA path fail its own workspace check. */
    GFFX_CHECK(gffx_mesh_gather_faces_workspace(vertices.size(0), faces.size(0),
                                                dtype_of(vertices), &context, &bytes, &alignment,
                                                &diagnostic.buffer),
               diagnostic.text());
    Tensor workspace = workspace_of(vertices, bytes);
    const gffx_buffer buffer = workspace_buffer(workspace, bytes);

    const std::vector<int64_t> size{faces.size(0), 3, 3};
    Tensor gathered = torch::stable::new_empty(vertices, size);

    gffx_tensor_view vertices_view = arena.read(vertices);
    gffx_tensor_view faces_view = arena.read(faces);
    gffx_tensor_view gathered_view = arena.write(gathered);
    GFFX_CHECK(gffx_mesh_gather_faces(&vertices_view, &faces_view, &context, &gathered_view,
                                      &buffer, &diagnostic.buffer),
               diagnostic.text());
    return gathered;
}

Tensor gather_faces_backward(
    const Tensor &vertices, const Tensor &faces, const Tensor &grad_gathered
) {
    ViewArena arena;
    Diagnostic diagnostic;
    gffx_execution_context context = context_for(vertices);

    const std::vector<int64_t> size{vertices.size(0), 3};
    Tensor grad_vertices = torch::stable::new_empty(vertices, size);

    gffx_tensor_view vertices_view = arena.read(vertices);
    gffx_tensor_view faces_view = arena.read(faces);
    gffx_tensor_view grad_gathered_view = arena.read(grad_gathered);
    gffx_tensor_view grad_vertices_view = arena.write(grad_vertices);
    GFFX_CHECK(gffx_mesh_gather_faces_backward(&vertices_view, &faces_view, &grad_gathered_view,
                                               &context, &grad_vertices_view, nullptr,
                                               &diagnostic.buffer),
               diagnostic.text());
    return grad_vertices;
}

/* -------------------------------------------------------------- transforms.transform_points */

Tensor transform_points(
    const Tensor &points, const Tensor &matrices, const Tensor &point_offsets
) {
    ViewArena arena;
    Diagnostic diagnostic;
    gffx_execution_context context = context_for(points);
    uint64_t bytes = 0, alignment = 0;

    GFFX_CHECK(gffx_transforms_transform_points_workspace(
                   points.size(0), matrices.size(0), dtype_of(points), &context, &bytes,
                   &alignment, &diagnostic.buffer),
               diagnostic.text());
    Tensor workspace = workspace_of(points, bytes);
    const gffx_buffer buffer = workspace_buffer(workspace, bytes);

    const std::vector<int64_t> size{points.size(0), 4};
    Tensor homogeneous = torch::stable::new_empty(points, size);

    gffx_tensor_view points_view = arena.read(points);
    gffx_tensor_view matrices_view = arena.read(matrices);
    gffx_tensor_view offsets_view = arena.read(point_offsets);
    gffx_tensor_view homogeneous_view = arena.write(homogeneous);
    GFFX_CHECK(gffx_transforms_transform_points(&points_view, &matrices_view, &offsets_view,
                                                &context, &homogeneous_view,
                                                bytes > 0 ? &buffer : nullptr,
                                                &diagnostic.buffer),
               diagnostic.text());
    return homogeneous;
}

std::tuple<Tensor, Tensor> transform_points_backward(
    const Tensor &points, const Tensor &matrices, const Tensor &point_offsets,
    const Tensor &grad_homogeneous
) {
    ViewArena arena;
    Diagnostic diagnostic;
    gffx_execution_context context = context_for(points);
    uint64_t bytes = 0, alignment = 0;

    GFFX_CHECK(gffx_transforms_transform_points_workspace(
                   points.size(0), matrices.size(0), dtype_of(points), &context, &bytes,
                   &alignment, &diagnostic.buffer),
               diagnostic.text());
    Tensor workspace = workspace_of(points, bytes);
    const gffx_buffer buffer = workspace_buffer(workspace, bytes);

    const std::vector<int64_t> point_size{points.size(0), 3};
    const std::vector<int64_t> matrix_size{matrices.size(0), 4, 4};
    Tensor grad_points = torch::stable::new_empty(points, point_size);
    Tensor grad_matrices = torch::stable::new_empty(points, matrix_size);

    gffx_tensor_view points_view = arena.read(points);
    gffx_tensor_view matrices_view = arena.read(matrices);
    gffx_tensor_view offsets_view = arena.read(point_offsets);
    gffx_tensor_view grad_homogeneous_view = arena.read(grad_homogeneous);
    gffx_tensor_view grad_points_view = arena.write(grad_points);
    gffx_tensor_view grad_matrices_view = arena.write(grad_matrices);
    GFFX_CHECK(gffx_transforms_transform_points_backward(
                   &points_view, &matrices_view, &offsets_view, &grad_homogeneous_view, &context,
                   &grad_points_view, &grad_matrices_view, bytes > 0 ? &buffer : nullptr,
                   &diagnostic.buffer),
               diagnostic.text());
    return std::make_tuple(grad_points, grad_matrices);
}

/* ------------------------------------------------------------ transforms.perspective_divide */

std::tuple<Tensor, Tensor> perspective_divide(const Tensor &homogeneous, double eps) {
    ViewArena arena;
    Diagnostic diagnostic;
    gffx_execution_context context = context_for(homogeneous);
    uint64_t bytes = 0, alignment = 0;

    GFFX_CHECK(gffx_transforms_perspective_divide_workspace(
                   homogeneous.size(0), dtype_of(homogeneous), &context, &bytes, &alignment,
                   &diagnostic.buffer),
               diagnostic.text());
    Tensor workspace = workspace_of(homogeneous, bytes);
    const gffx_buffer buffer = workspace_buffer(workspace, bytes);

    const std::vector<int64_t> ndc_size{homogeneous.size(0), 3};
    const std::vector<int64_t> valid_size{homogeneous.size(0)};
    Tensor ndc = torch::stable::new_empty(homogeneous, ndc_size);
    Tensor valid = torch::stable::new_empty(homogeneous, valid_size, ScalarType::Bool);

    gffx_tensor_view homogeneous_view = arena.read(homogeneous);
    gffx_tensor_view ndc_view = arena.write(ndc);
    gffx_tensor_view valid_view = arena.write(valid);
    GFFX_CHECK(gffx_transforms_perspective_divide(&homogeneous_view, eps, &context, &ndc_view,
                                                  &valid_view, bytes > 0 ? &buffer : nullptr,
                                                  &diagnostic.buffer),
               diagnostic.text());
    return std::make_tuple(ndc, valid);
}

Tensor perspective_divide_backward(
    const Tensor &homogeneous, double eps, const Tensor &grad_ndc
) {
    ViewArena arena;
    Diagnostic diagnostic;
    gffx_execution_context context = context_for(homogeneous);
    uint64_t bytes = 0, alignment = 0;

    GFFX_CHECK(gffx_transforms_perspective_divide_workspace(
                   homogeneous.size(0), dtype_of(homogeneous), &context, &bytes, &alignment,
                   &diagnostic.buffer),
               diagnostic.text());
    Tensor workspace = workspace_of(homogeneous, bytes);
    const gffx_buffer buffer = workspace_buffer(workspace, bytes);

    const std::vector<int64_t> size{homogeneous.size(0), 4};
    Tensor grad_homogeneous = torch::stable::new_empty(homogeneous, size);

    gffx_tensor_view homogeneous_view = arena.read(homogeneous);
    gffx_tensor_view grad_ndc_view = arena.read(grad_ndc);
    gffx_tensor_view grad_homogeneous_view = arena.write(grad_homogeneous);
    GFFX_CHECK(gffx_transforms_perspective_divide_backward(
                   &homogeneous_view, eps, &grad_ndc_view, &context, &grad_homogeneous_view,
                   bytes > 0 ? &buffer : nullptr, &diagnostic.buffer),
               diagnostic.text());
    return grad_homogeneous;
}

/* -------------------------------------------------------------- mesh.build_edge_topology */

std::tuple<Tensor, Tensor, Tensor, Tensor> build_edge_topology(
    const Tensor &faces, const Tensor &face_offsets
) {
    ViewArena arena;
    Diagnostic diagnostic;
    gffx_execution_context context = context_for(faces);
    uint64_t bytes = 0, alignment = 0;
    const int64_t face_count = faces.size(0);
    const int64_t batch_count = face_offsets.size(0) - 1;

    GFFX_CHECK(gffx_mesh_build_edge_topology_workspace(
                   face_count, batch_count, &context, &bytes, &alignment, &diagnostic.buffer),
               diagnostic.text());
    Tensor workspace = workspace_of(faces, bytes);
    const gffx_buffer buffer = workspace_buffer(workspace, bytes);

    /* Capacities are exact: each face contributes three half-edges, so 3F bounds every output. */
    const std::vector<int64_t> edge_size{face_count * 3, 2};
    const std::vector<int64_t> offset_size{face_count * 3 + 1};
    const std::vector<int64_t> incidence_size{face_count * 3};
    const std::vector<int64_t> mesh_size{batch_count + 1};
    Tensor edges = torch::stable::new_empty(faces, edge_size);
    Tensor edge_face_offsets = torch::stable::new_empty(faces, offset_size);
    Tensor edge_faces = torch::stable::new_empty(faces, incidence_size);
    Tensor mesh_edge_offsets = torch::stable::new_empty(faces, mesh_size);

    gffx_tensor_view faces_view = arena.read(faces);
    gffx_tensor_view face_offsets_view = arena.read(face_offsets);
    gffx_tensor_view edges_view = arena.write(edges);
    gffx_tensor_view edge_face_offsets_view = arena.write(edge_face_offsets);
    gffx_tensor_view edge_faces_view = arena.write(edge_faces);
    gffx_tensor_view mesh_edge_offsets_view = arena.write(mesh_edge_offsets);
    GFFX_CHECK(gffx_mesh_build_edge_topology(
                   &faces_view, &face_offsets_view, &context, &edges_view,
                   &edge_face_offsets_view, &edge_faces_view, &mesh_edge_offsets_view,
                   bytes > 0 ? &buffer : nullptr, &diagnostic.buffer),
               diagnostic.text());
    return std::make_tuple(edges, edge_face_offsets, edge_faces, mesh_edge_offsets);
}

/* ------------------------------------------------------------------------------ points.knn */

std::tuple<Tensor, Tensor, Tensor> knn(
    const Tensor &query, const Tensor &reference, const Tensor &query_offsets,
    const Tensor &reference_offsets, int64_t neighbor_count
) {
    ViewArena arena;
    Diagnostic diagnostic;
    gffx_execution_context context = context_for(query);
    uint64_t bytes = 0, alignment = 0;

    GFFX_CHECK(gffx_points_knn_workspace(
                   query.size(0), reference.size(0), neighbor_count, dtype_of(query), &context,
                   &bytes, &alignment, &diagnostic.buffer),
               diagnostic.text());
    Tensor workspace = workspace_of(query, bytes);
    const gffx_buffer buffer = workspace_buffer(workspace, bytes);

    const std::vector<int64_t> size{query.size(0), neighbor_count};
    Tensor distance_squared = torch::stable::new_empty(query, size);
    Tensor reference_index = torch::stable::new_empty(query, size, ScalarType::Int);
    Tensor valid = torch::stable::new_empty(query, size, ScalarType::Bool);

    gffx_tensor_view query_view = arena.read(query);
    gffx_tensor_view reference_view = arena.read(reference);
    gffx_tensor_view query_offsets_view = arena.read(query_offsets);
    gffx_tensor_view reference_offsets_view = arena.read(reference_offsets);
    gffx_tensor_view distance_view = arena.write(distance_squared);
    gffx_tensor_view index_view = arena.write(reference_index);
    gffx_tensor_view valid_view = arena.write(valid);
    GFFX_CHECK(gffx_points_knn(&query_view, &reference_view, &query_offsets_view,
                               &reference_offsets_view, neighbor_count, &context, &distance_view,
                               &index_view, &valid_view, bytes > 0 ? &buffer : nullptr,
                               &diagnostic.buffer),
               diagnostic.text());
    return std::make_tuple(distance_squared, reference_index, valid);
}

std::tuple<Tensor, Tensor> knn_backward(
    const Tensor &query, const Tensor &reference, const Tensor &reference_index,
    const Tensor &valid, const Tensor &grad_distance_squared
) {
    ViewArena arena;
    Diagnostic diagnostic;
    gffx_execution_context context = context_for(query);

    const std::vector<int64_t> query_size{query.size(0), 3};
    const std::vector<int64_t> reference_size{reference.size(0), 3};
    Tensor grad_query = torch::stable::new_empty(query, query_size);
    Tensor grad_reference = torch::stable::new_empty(query, reference_size);

    gffx_tensor_view query_view = arena.read(query);
    gffx_tensor_view reference_view = arena.read(reference);
    gffx_tensor_view index_view = arena.read(reference_index);
    gffx_tensor_view valid_view = arena.read(valid);
    gffx_tensor_view grad_distance_view = arena.read(grad_distance_squared);
    gffx_tensor_view grad_query_view = arena.write(grad_query);
    gffx_tensor_view grad_reference_view = arena.write(grad_reference);
    GFFX_CHECK(gffx_points_knn_backward(&query_view, &reference_view, &index_view, &valid_view,
                                        &grad_distance_view, &context, &grad_query_view,
                                        &grad_reference_view, nullptr, &diagnostic.buffer),
               diagnostic.text());
    return std::make_tuple(grad_query, grad_reference);
}

/* ------------------------------------------------------- points.closest_point_on_mesh */

std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> closest_point_on_mesh(
    const Tensor &points, const Tensor &vertices, const Tensor &faces,
    const Tensor &point_offsets, const Tensor &vertex_offsets, const Tensor &face_offsets,
    double eps
) {
    ViewArena arena;
    Diagnostic diagnostic;
    gffx_execution_context context = context_for(points);
    uint64_t bytes = 0, alignment = 0;

    GFFX_CHECK(gffx_points_closest_point_on_mesh_workspace(
                   points.size(0), vertices.size(0), faces.size(0), dtype_of(points), &context,
                   &bytes, &alignment, &diagnostic.buffer),
               diagnostic.text());
    Tensor workspace = workspace_of(points, bytes);
    const gffx_buffer buffer = workspace_buffer(workspace, bytes);

    const std::vector<int64_t> scalar_size{points.size(0)};
    const std::vector<int64_t> triple_size{points.size(0), 3};
    Tensor distance_squared = torch::stable::new_empty(points, scalar_size);
    Tensor face_index = torch::stable::new_empty(points, scalar_size, ScalarType::Int);
    Tensor barycentric = torch::stable::new_empty(points, triple_size);
    Tensor closest = torch::stable::new_empty(points, triple_size);
    Tensor valid = torch::stable::new_empty(points, scalar_size, ScalarType::Bool);

    gffx_tensor_view points_view = arena.read(points);
    gffx_tensor_view vertices_view = arena.read(vertices);
    gffx_tensor_view faces_view = arena.read(faces);
    gffx_tensor_view point_offsets_view = arena.read(point_offsets);
    gffx_tensor_view vertex_offsets_view = arena.read(vertex_offsets);
    gffx_tensor_view face_offsets_view = arena.read(face_offsets);
    gffx_tensor_view distance_view = arena.write(distance_squared);
    gffx_tensor_view face_index_view = arena.write(face_index);
    gffx_tensor_view barycentric_view = arena.write(barycentric);
    gffx_tensor_view closest_view = arena.write(closest);
    gffx_tensor_view valid_view = arena.write(valid);
    GFFX_CHECK(gffx_points_closest_point_on_mesh(
                   &points_view, &vertices_view, &faces_view, &point_offsets_view,
                   &vertex_offsets_view, &face_offsets_view, eps, &context, &distance_view,
                   &face_index_view, &barycentric_view, &closest_view, &valid_view,
                   bytes > 0 ? &buffer : nullptr, &diagnostic.buffer),
               diagnostic.text());
    return std::make_tuple(distance_squared, face_index, barycentric, closest, valid);
}

std::tuple<Tensor, Tensor> closest_point_on_mesh_backward(
    const Tensor &points, const Tensor &vertices, const Tensor &faces, const Tensor &face_index,
    const Tensor &barycentric, const Tensor &closest, const Tensor &valid,
    const Tensor &grad_distance_squared
) {
    ViewArena arena;
    Diagnostic diagnostic;
    gffx_execution_context context = context_for(points);

    const std::vector<int64_t> point_size{points.size(0), 3};
    const std::vector<int64_t> vertex_size{vertices.size(0), 3};
    Tensor grad_points = torch::stable::new_empty(points, point_size);
    Tensor grad_vertices = torch::stable::new_empty(points, vertex_size);

    gffx_tensor_view points_view = arena.read(points);
    gffx_tensor_view vertices_view = arena.read(vertices);
    gffx_tensor_view faces_view = arena.read(faces);
    gffx_tensor_view face_index_view = arena.read(face_index);
    gffx_tensor_view barycentric_view = arena.read(barycentric);
    gffx_tensor_view closest_view = arena.read(closest);
    gffx_tensor_view valid_view = arena.read(valid);
    gffx_tensor_view grad_distance_view = arena.read(grad_distance_squared);
    gffx_tensor_view grad_points_view = arena.write(grad_points);
    gffx_tensor_view grad_vertices_view = arena.write(grad_vertices);
    GFFX_CHECK(gffx_points_closest_point_on_mesh_backward(
                   &points_view, &vertices_view, &faces_view, &face_index_view,
                   &barycentric_view, &closest_view, &valid_view, &grad_distance_view, &context,
                   &grad_points_view, &grad_vertices_view, nullptr, &diagnostic.buffer),
               diagnostic.text());
    return std::make_tuple(grad_points, grad_vertices);
}

/* ------------------------------------------------------------------- mesh.sample_surface */

std::tuple<Tensor, Tensor, Tensor, Tensor> sample_surface(
    const Tensor &vertices, const Tensor &faces, const Tensor &vertex_offsets,
    const Tensor &face_offsets, int64_t sample_count, const Tensor &rng_key,
    const Tensor &rng_counter, double eps
) {
    ViewArena arena;
    Diagnostic diagnostic;
    gffx_execution_context context = context_for(vertices);
    uint64_t bytes = 0, alignment = 0;
    const int64_t batch_count = face_offsets.size(0) - 1;

    GFFX_CHECK(gffx_mesh_sample_surface_workspace(
                   vertices.size(0), faces.size(0), sample_count, dtype_of(vertices), &context,
                   &bytes, &alignment, &diagnostic.buffer),
               diagnostic.text());
    Tensor workspace = workspace_of(vertices, bytes);
    const gffx_buffer buffer = workspace_buffer(workspace, bytes);

    const std::vector<int64_t> point_size{batch_count, sample_count, 3};
    const std::vector<int64_t> index_size{batch_count, sample_count};
    const std::vector<int64_t> counter_size{2};
    Tensor points = torch::stable::new_empty(vertices, point_size);
    Tensor face_index = torch::stable::new_empty(vertices, index_size, ScalarType::Int);
    Tensor barycentric = torch::stable::new_empty(vertices, point_size);
    Tensor next_counter = torch::stable::new_empty(vertices, counter_size, ScalarType::UInt32);

    gffx_tensor_view vertices_view = arena.read(vertices);
    gffx_tensor_view faces_view = arena.read(faces);
    gffx_tensor_view vertex_offsets_view = arena.read(vertex_offsets);
    gffx_tensor_view face_offsets_view = arena.read(face_offsets);
    gffx_tensor_view key_view = arena.read(rng_key);
    gffx_tensor_view counter_view = arena.read(rng_counter);
    gffx_tensor_view points_view = arena.write(points);
    gffx_tensor_view face_index_view = arena.write(face_index);
    gffx_tensor_view barycentric_view = arena.write(barycentric);
    gffx_tensor_view next_counter_view = arena.write(next_counter);
    GFFX_CHECK(gffx_mesh_sample_surface(
                   &vertices_view, &faces_view, &vertex_offsets_view, &face_offsets_view,
                   sample_count, &key_view, &counter_view, eps, &context, &points_view,
                   &face_index_view, &barycentric_view, &next_counter_view,
                   bytes > 0 ? &buffer : nullptr, &diagnostic.buffer),
               diagnostic.text());
    return std::make_tuple(points, face_index, barycentric, next_counter);
}

Tensor sample_surface_backward(
    const Tensor &vertices, const Tensor &faces, const Tensor &face_index,
    const Tensor &barycentric, const Tensor &grad_points
) {
    ViewArena arena;
    Diagnostic diagnostic;
    gffx_execution_context context = context_for(vertices);

    const std::vector<int64_t> size{vertices.size(0), 3};
    Tensor grad_vertices = torch::stable::new_empty(vertices, size);

    gffx_tensor_view faces_view = arena.read(faces);
    gffx_tensor_view face_index_view = arena.read(face_index);
    gffx_tensor_view barycentric_view = arena.read(barycentric);
    gffx_tensor_view grad_points_view = arena.read(grad_points);
    gffx_tensor_view grad_vertices_view = arena.write(grad_vertices);
    GFFX_CHECK(gffx_mesh_sample_surface_backward(
                   &faces_view, &face_index_view, &barycentric_view, &grad_points_view, &context,
                   &grad_vertices_view, nullptr, &diagnostic.buffer),
               diagnostic.text());
    return grad_vertices;
}

/* ----------------------------------------------------------------------- render.rasterize */

std::tuple<Tensor, Tensor, Tensor, Tensor> rasterize(
    const Tensor &ndc_vertices, const Tensor &faces, const Tensor &vertex_offsets,
    const Tensor &face_offsets, int64_t image_height, int64_t image_width,
    int64_t faces_per_pixel, double blur_radius_px, int64_t cull_mode, double eps
) {
    ViewArena arena;
    Diagnostic diagnostic;
    gffx_execution_context context = context_for(ndc_vertices);
    uint64_t bytes = 0, alignment = 0;
    const int64_t batch_count = face_offsets.size(0) - 1;

    GFFX_CHECK(gffx_render_rasterize_workspace(
                   ndc_vertices.size(0), faces.size(0), image_height, image_width,
                   faces_per_pixel, dtype_of(ndc_vertices), &context, &bytes, &alignment,
                   &diagnostic.buffer),
               diagnostic.text());
    Tensor workspace = workspace_of(ndc_vertices, bytes);
    const gffx_buffer buffer = workspace_buffer(workspace, bytes);

    const std::vector<int64_t> fragment_size{
        batch_count, image_height, image_width, faces_per_pixel};
    const std::vector<int64_t> bary_size{
        batch_count, image_height, image_width, faces_per_pixel, 3};
    Tensor face_index = torch::stable::new_empty(ndc_vertices, fragment_size, ScalarType::Int);
    Tensor barycentric = torch::stable::new_empty(ndc_vertices, bary_size);
    Tensor depth = torch::stable::new_empty(ndc_vertices, fragment_size);
    Tensor signed_distance = torch::stable::new_empty(ndc_vertices, fragment_size);

    gffx_tensor_view ndc_view = arena.read(ndc_vertices);
    gffx_tensor_view faces_view = arena.read(faces);
    gffx_tensor_view vertex_offsets_view = arena.read(vertex_offsets);
    gffx_tensor_view face_offsets_view = arena.read(face_offsets);
    gffx_tensor_view face_index_view = arena.write(face_index);
    gffx_tensor_view barycentric_view = arena.write(barycentric);
    gffx_tensor_view depth_view = arena.write(depth);
    gffx_tensor_view signed_distance_view = arena.write(signed_distance);
    GFFX_CHECK(gffx_render_rasterize(
                   &ndc_view, &faces_view, &vertex_offsets_view, &face_offsets_view,
                   image_height, image_width, faces_per_pixel, blur_radius_px,
                   static_cast<uint32_t>(cull_mode), eps, &context, &face_index_view,
                   &barycentric_view, &depth_view, &signed_distance_view,
                   bytes > 0 ? &buffer : nullptr, &diagnostic.buffer),
               diagnostic.text());
    return std::make_tuple(face_index, barycentric, depth, signed_distance);
}

Tensor rasterize_backward(
    const Tensor &ndc_vertices, const Tensor &faces, int64_t image_height, int64_t image_width,
    const Tensor &face_index, const Tensor &grad_barycentric, const Tensor &grad_depth,
    const Tensor &grad_signed_distance
) {
    ViewArena arena;
    Diagnostic diagnostic;
    gffx_execution_context context = context_for(ndc_vertices);

    const std::vector<int64_t> size{ndc_vertices.size(0), 3};
    Tensor grad_ndc = torch::stable::new_empty(ndc_vertices, size);

    gffx_tensor_view ndc_view = arena.read(ndc_vertices);
    gffx_tensor_view faces_view = arena.read(faces);
    gffx_tensor_view face_index_view = arena.read(face_index);
    gffx_tensor_view grad_barycentric_view = arena.read(grad_barycentric);
    gffx_tensor_view grad_depth_view = arena.read(grad_depth);
    gffx_tensor_view grad_signed_distance_view = arena.read(grad_signed_distance);
    gffx_tensor_view grad_ndc_view = arena.write(grad_ndc);
    GFFX_CHECK(gffx_render_rasterize_backward(
                   &ndc_view, &faces_view, image_height, image_width, &face_index_view,
                   &grad_barycentric_view, &grad_depth_view, &grad_signed_distance_view,
                   &context, &grad_ndc_view, nullptr, &diagnostic.buffer),
               diagnostic.text());
    return grad_ndc;
}

/* --------------------------------------------------------------------- render.interpolate */

Tensor interpolate(
    const Tensor &face_index, const Tensor &barycentric, const Tensor &face_attributes
) {
    ViewArena arena;
    Diagnostic diagnostic;
    gffx_execution_context context = context_for(face_attributes);

    std::vector<int64_t> size;
    for (int64_t index = 0; index < face_index.dim(); ++index) {
        size.push_back(face_index.size(index));
    }
    size.push_back(face_attributes.size(2));
    Tensor attributes = torch::stable::new_empty(face_attributes, size);

    gffx_tensor_view face_index_view = arena.read(face_index);
    gffx_tensor_view barycentric_view = arena.read(barycentric);
    gffx_tensor_view face_attributes_view = arena.read(face_attributes);
    gffx_tensor_view attributes_view = arena.write(attributes);
    GFFX_CHECK(gffx_render_interpolate(&face_index_view, &barycentric_view,
                                       &face_attributes_view, &context, &attributes_view,
                                       nullptr, &diagnostic.buffer),
               diagnostic.text());
    return attributes;
}

std::tuple<Tensor, Tensor> interpolate_backward(
    const Tensor &face_index, const Tensor &barycentric, const Tensor &face_attributes,
    const Tensor &grad_attributes
) {
    ViewArena arena;
    Diagnostic diagnostic;
    gffx_execution_context context = context_for(face_attributes);

    std::vector<int64_t> bary_size;
    for (int64_t index = 0; index < barycentric.dim(); ++index) {
        bary_size.push_back(barycentric.size(index));
    }
    const std::vector<int64_t> attribute_size{
        face_attributes.size(0), face_attributes.size(1), face_attributes.size(2)};
    Tensor grad_barycentric = torch::stable::new_empty(face_attributes, bary_size);
    Tensor grad_face_attributes = torch::stable::new_empty(face_attributes, attribute_size);

    gffx_tensor_view face_index_view = arena.read(face_index);
    gffx_tensor_view barycentric_view = arena.read(barycentric);
    gffx_tensor_view face_attributes_view = arena.read(face_attributes);
    gffx_tensor_view grad_attributes_view = arena.read(grad_attributes);
    gffx_tensor_view grad_barycentric_view = arena.write(grad_barycentric);
    gffx_tensor_view grad_face_attributes_view = arena.write(grad_face_attributes);
    GFFX_CHECK(gffx_render_interpolate_backward(
                   &face_index_view, &barycentric_view, &face_attributes_view,
                   &grad_attributes_view, &context, &grad_barycentric_view,
                   &grad_face_attributes_view, nullptr, &diagnostic.buffer),
               diagnostic.text());
    return std::make_tuple(grad_barycentric, grad_face_attributes);
}


/* ---------------------------------------------------------------- render.texture_pyramid */

/* The level chain, computed on the host because the pyramid must be allocated before the kernel
 * that would report its size can run. This mirrors the rule in API_CONTRACT_V0_1.md section 4.7:
 * level 0 is the texture, each later level halves by integer division so an odd dimension drops
 * its trailing row or column, and a dimension already at 1 is carried through. The duplication is
 * real, and it is bounded by a cross-check the kernel already performs between level_offsets[1]
 * and the extents and channel count, so a drift here fails loudly at the first sample rather than
 * producing a quietly wrong pyramid. */
int64_t pyramid_level_count(int64_t height, int64_t width, int64_t levels) {
    if (levels > 0) {
        return levels;
    }
    int64_t extent = height > width ? height : width;
    int64_t count = 1;
    while (extent > 1) {
        extent /= 2;
        ++count;
    }
    return count;
}

int64_t pyramid_element_count(
    int64_t height, int64_t width, int64_t channels, int64_t level_count
) {
    int64_t total = 0;
    int64_t level_height = height;
    int64_t level_width = width;
    for (int64_t level = 0; level < level_count; ++level) {
        total += level_height * level_width * channels;
        level_height = level_height > 1 ? level_height / 2 : 1;
        level_width = level_width > 1 ? level_width / 2 : 1;
    }
    return total;
}

std::tuple<Tensor, Tensor> texture_pyramid(const Tensor &texture, int64_t levels) {
    ViewArena arena;
    Diagnostic diagnostic;
    gffx_execution_context context = context_for(texture);

    const int64_t height = texture.size(0);
    const int64_t width = texture.size(1);
    const int64_t channels = texture.size(2);
    const int64_t level_count = pyramid_level_count(height, width, levels);
    const std::vector<int64_t> pyramid_size{
        pyramid_element_count(height, width, channels, level_count)};
    const std::vector<int64_t> offset_size{level_count + 1};

    uint64_t bytes = 0;
    uint64_t alignment = 0;
    GFFX_CHECK(gffx_render_texture_pyramid_workspace(
                   height, width, channels, dtype_of(texture), &context, &bytes, &alignment,
                   &diagnostic.buffer),
               diagnostic.text());
    const std::vector<int64_t> workspace_size{static_cast<int64_t>(bytes)};
    Tensor workspace = torch::stable::new_empty(texture, workspace_size, ScalarType::Byte);
    const gffx_buffer buffer = workspace_buffer(workspace, bytes);

    Tensor pyramid = torch::stable::new_empty(texture, pyramid_size);
    Tensor level_offsets = torch::stable::new_empty(texture, offset_size, ScalarType::Int);

    gffx_tensor_view texture_view = arena.read(texture);
    gffx_tensor_view pyramid_view = arena.write(pyramid);
    gffx_tensor_view offsets_view = arena.write(level_offsets);
    GFFX_CHECK(gffx_render_texture_pyramid(&texture_view, levels, &context, &pyramid_view,
                                           &offsets_view, bytes > 0 ? &buffer : nullptr,
                                           &diagnostic.buffer),
               diagnostic.text());
    return std::make_tuple(pyramid, level_offsets);
}

Tensor texture_pyramid_backward(
    const Tensor &level_offsets, int64_t texture_height, int64_t texture_width,
    int64_t channel_count, const Tensor &grad_pyramid
) {
    ViewArena arena;
    Diagnostic diagnostic;
    gffx_execution_context context = context_for(grad_pyramid);

    const std::vector<int64_t> size{texture_height, texture_width, channel_count};
    Tensor grad_texture = torch::stable::new_empty(grad_pyramid, size);

    gffx_tensor_view offsets_view = arena.read(level_offsets);
    gffx_tensor_view grad_pyramid_view = arena.read(grad_pyramid);
    gffx_tensor_view grad_texture_view = arena.write(grad_texture);
    GFFX_CHECK(gffx_render_texture_pyramid_backward(
                   &offsets_view, texture_height, texture_width, channel_count,
                   &grad_pyramid_view, &context, &grad_texture_view, nullptr,
                   &diagnostic.buffer),
               diagnostic.text());
    return grad_texture;
}


/* ------------------------------------------------------------------------- render.texture */

/* derivatives, lod and border each carry a presence flag rather than arriving as an optional
 * tensor, matching the convention established for cotangents above. The ABI expresses absence as a
 * null view, and a zero-filled tensor would assert a value of zero rather than the absence of one.
 * The distinction is load-bearing here: a lod of zero selects level 0 deliberately, while an
 * absent lod means the sampler derives nothing and reads level 0 for an entirely different
 * reason. */
Tensor texture(
    const Tensor &pyramid, const Tensor &level_offsets, int64_t texture_height,
    int64_t texture_width, int64_t channel_count, const Tensor &coordinates,
    const Tensor &derivatives, const Tensor &lod, bool has_derivatives, bool has_lod,
    int64_t filter, int64_t mip_filter, int64_t wrap_u, int64_t wrap_v, const Tensor &border,
    bool has_border
) {
    ViewArena arena;
    Diagnostic diagnostic;
    gffx_execution_context context = context_for(pyramid);

    const int64_t sample_count = coordinates.size(0);
    const std::vector<int64_t> size{sample_count, channel_count};
    Tensor samples = torch::stable::new_empty(pyramid, size);

    uint64_t bytes = 0;
    uint64_t alignment = 0;
    GFFX_CHECK(gffx_render_texture_workspace(sample_count, channel_count, dtype_of(pyramid),
                                             &context, &bytes, &alignment, &diagnostic.buffer),
               diagnostic.text());
    const std::vector<int64_t> workspace_size{static_cast<int64_t>(bytes)};
    Tensor workspace = torch::stable::new_empty(pyramid, workspace_size, ScalarType::Byte);
    const gffx_buffer buffer = workspace_buffer(workspace, bytes);

    gffx_tensor_view pyramid_view = arena.read(pyramid);
    gffx_tensor_view offsets_view = arena.read(level_offsets);
    gffx_tensor_view coordinates_view = arena.read(coordinates);
    gffx_tensor_view derivatives_view = arena.read(derivatives);
    gffx_tensor_view lod_view = arena.read(lod);
    gffx_tensor_view border_view = arena.read(border);
    gffx_tensor_view samples_view = arena.write(samples);
    GFFX_CHECK(gffx_render_texture(
                   &pyramid_view, &offsets_view, texture_height, texture_width,
                   &coordinates_view, has_derivatives ? &derivatives_view : nullptr,
                   has_lod ? &lod_view : nullptr, static_cast<uint32_t>(filter),
                   static_cast<uint32_t>(mip_filter), static_cast<uint32_t>(wrap_u),
                   static_cast<uint32_t>(wrap_v), has_border ? &border_view : nullptr, &context,
                   &samples_view, bytes > 0 ? &buffer : nullptr, &diagnostic.buffer),
               diagnostic.text());
    return samples;
}

std::tuple<Tensor, Tensor> texture_backward(
    const Tensor &pyramid, const Tensor &level_offsets, int64_t texture_height,
    int64_t texture_width, const Tensor &coordinates, const Tensor &derivatives,
    const Tensor &lod, bool has_derivatives, bool has_lod, int64_t filter, int64_t mip_filter,
    int64_t wrap_u, int64_t wrap_v, const Tensor &border, bool has_border,
    const Tensor &grad_samples
) {
    ViewArena arena;
    Diagnostic diagnostic;
    gffx_execution_context context = context_for(pyramid);

    std::vector<int64_t> pyramid_size;
    for (int64_t index = 0; index < pyramid.dim(); ++index) {
        pyramid_size.push_back(pyramid.size(index));
    }
    std::vector<int64_t> coordinate_size;
    for (int64_t index = 0; index < coordinates.dim(); ++index) {
        coordinate_size.push_back(coordinates.size(index));
    }
    Tensor grad_pyramid = torch::stable::new_empty(pyramid, pyramid_size);
    Tensor grad_coordinates = torch::stable::new_empty(coordinates, coordinate_size);

    gffx_tensor_view pyramid_view = arena.read(pyramid);
    gffx_tensor_view offsets_view = arena.read(level_offsets);
    gffx_tensor_view coordinates_view = arena.read(coordinates);
    gffx_tensor_view derivatives_view = arena.read(derivatives);
    gffx_tensor_view lod_view = arena.read(lod);
    gffx_tensor_view border_view = arena.read(border);
    gffx_tensor_view grad_samples_view = arena.read(grad_samples);
    gffx_tensor_view grad_pyramid_view = arena.write(grad_pyramid);
    gffx_tensor_view grad_coordinates_view = arena.write(grad_coordinates);
    GFFX_CHECK(gffx_render_texture_backward(
                   &pyramid_view, &offsets_view, texture_height, texture_width,
                   &coordinates_view, has_derivatives ? &derivatives_view : nullptr,
                   has_lod ? &lod_view : nullptr, static_cast<uint32_t>(filter),
                   static_cast<uint32_t>(mip_filter), static_cast<uint32_t>(wrap_u),
                   static_cast<uint32_t>(wrap_v), has_border ? &border_view : nullptr,
                   &grad_samples_view, &context, &grad_pyramid_view, &grad_coordinates_view,
                   nullptr, &diagnostic.buffer),
               diagnostic.text());
    return std::make_tuple(grad_pyramid, grad_coordinates);
}


/* --------------------------------------------------------------------------------- io.ply
 *
 * The reader takes bytes, not a path. That is the core's boundary and it is kept here rather than
 * flattened: Python opens the file, this parses the buffer. The same op therefore serves a file, a
 * memory-mapped region, an archive member, or a network download with no second entry point, and
 * the adapter needs no string in its schema.
 */

std::tuple<int64_t, int64_t, int64_t> ply_probe(const Tensor &data) {
    Diagnostic diagnostic;
    /* The reader is CPU-only: it parses a host byte buffer, and there is no device variant. */
    gffx_execution_context context = cpu_context();
    gffx_ply_header header{};
    header.struct_size = static_cast<uint32_t>(sizeof(header));
    header.abi_version = GFFX_ABI_VERSION;

    GFFX_CHECK(gffx_io_ply_probe(data.const_data_ptr(), data.numel(), &context, &header,
                                 &diagnostic.buffer),
               diagnostic.text());
    return std::make_tuple(static_cast<int64_t>(header.format), header.vertex_count,
                           header.face_count);
}

std::tuple<Tensor, Tensor> ply_read(const Tensor &data, bool double_precision) {
    ViewArena arena;
    Diagnostic diagnostic;
    gffx_execution_context context = cpu_context();
    gffx_ply_header header{};
    header.struct_size = static_cast<uint32_t>(sizeof(header));
    header.abi_version = GFFX_ABI_VERSION;

    /* Probed here rather than trusting a count passed down from Python, so the buffer that is
     * parsed is the buffer that was measured. */
    GFFX_CHECK(gffx_io_ply_probe(data.const_data_ptr(), data.numel(), &context, &header,
                                 &diagnostic.buffer),
               diagnostic.text());

    const std::vector<int64_t> vertex_size{header.vertex_count, 3};
    const std::vector<int64_t> face_size{header.face_count, 3};
    Tensor vertices = torch::stable::new_empty(
        data, vertex_size, double_precision ? ScalarType::Double : ScalarType::Float);
    Tensor faces = torch::stable::new_empty(data, face_size, ScalarType::Int);

    gffx_tensor_view vertices_view = arena.write(vertices);
    gffx_tensor_view faces_view = arena.write(faces);
    GFFX_CHECK(gffx_io_ply_read(data.const_data_ptr(), data.numel(), &header, &context,
                                &vertices_view, &faces_view, nullptr, &diagnostic.buffer),
               diagnostic.text());
    return std::make_tuple(vertices, faces);
}

/* ------------------------------------------------------------------------- mesh.validate */

/* Returned as a tensor of counts rather than a struct, because the Stable ABI schema has no
 * struct type. The Python layer names the fields, so a caller never indexes this by hand. */
Tensor mesh_validate(
    const Tensor &vertices, const Tensor &faces, const Tensor &vertex_offsets,
    const Tensor &face_offsets, double eps, int64_t flags
) {
    ViewArena arena;
    Diagnostic diagnostic;
    gffx_execution_context context = context_for(vertices);
    gffx_mesh_validation_report report{};
    report.struct_size = static_cast<uint32_t>(sizeof(report));
    report.abi_version = GFFX_ABI_VERSION;

    gffx_tensor_view vertices_view = arena.read(vertices);
    gffx_tensor_view faces_view = arena.read(faces);
    gffx_tensor_view vertex_offsets_view = arena.read(vertex_offsets);
    gffx_tensor_view face_offsets_view = arena.read(face_offsets);
    GFFX_CHECK(gffx_mesh_validate(&vertices_view, &faces_view, &vertex_offsets_view,
                                  &face_offsets_view, eps, static_cast<uint32_t>(flags),
                                  &context, &report, nullptr, &diagnostic.buffer),
               diagnostic.text());

    const std::vector<int64_t> size{7};
    Tensor out = torch::stable::new_empty(faces, size, ScalarType::Long);
    int64_t *values = out.mutable_data_ptr<int64_t>();
    values[0] = static_cast<int64_t>(report.findings);
    values[1] = report.first_bad_face;
    values[2] = report.first_bad_offset_batch;
    values[3] = report.degenerate_face_count;
    values[4] = report.nonfinite_vertex_count;
    values[5] = report.unreferenced_vertex_count;
    values[6] = 0;
    return out;
}


/* -------------------------------------------------------------------------------------------
 * The inference-only preallocated surface.
 *
 * API_CONTRACT_V0_1.md section 11 fixes what these are: the same kernels with the allocation
 * removed. Each takes the outputs and the workspace the caller already owns and performs no
 * allocation of its own, which is what makes the no-allocation-after-warm-up rule reachable for a
 * frame loop.
 *
 * mesh.build_edge_topology has no _out variant, and that is a scoping decision rather than an
 * omission: its own contract classifies it as setup-class work that is not permitted inside a
 * claimed real-time frame path, so a preallocated variant would advertise a capability the record
 * withholds. The same reasoning excludes mesh.validate and the PLY reader.
 * ------------------------------------------------------------------------------------------- */

void vertex_normals_out(
    const Tensor &vertices, const Tensor &faces, double eps, int64_t weighting,
    Tensor &normals, Tensor &workspace
) {
    ViewArena arena;
    Diagnostic diagnostic;
    gffx_execution_context context = context_for(vertices);
    const gffx_buffer buffer =
        workspace_buffer(workspace, static_cast<uint64_t>(workspace.numel()));
    gffx_tensor_view vertices_view = arena.read(vertices);
    gffx_tensor_view faces_view = arena.read(faces);
    gffx_tensor_view normals_view = arena.write(normals);
    GFFX_CHECK(gffx_mesh_vertex_normals(
                   &vertices_view, &faces_view, eps, static_cast<uint32_t>(weighting), &context,
                   &normals_view, workspace.numel() > 0 ? &buffer : nullptr, &diagnostic.buffer),
               diagnostic.text());
}

void gather_faces_out(
    const Tensor &vertices, const Tensor &faces, Tensor &gathered, Tensor &workspace
) {
    ViewArena arena;
    Diagnostic diagnostic;
    gffx_execution_context context = context_for(vertices);
    const gffx_buffer buffer =
        workspace_buffer(workspace, static_cast<uint64_t>(workspace.numel()));
    gffx_tensor_view vertices_view = arena.read(vertices);
    gffx_tensor_view faces_view = arena.read(faces);
    gffx_tensor_view gathered_view = arena.write(gathered);
    GFFX_CHECK(gffx_mesh_gather_faces(&vertices_view, &faces_view, &context, &gathered_view,
                                      workspace.numel() > 0 ? &buffer : nullptr,
                                      &diagnostic.buffer),
               diagnostic.text());
}

void transform_points_out(
    const Tensor &points, const Tensor &matrices, const Tensor &point_offsets,
    Tensor &homogeneous, Tensor &workspace
) {
    ViewArena arena;
    Diagnostic diagnostic;
    gffx_execution_context context = context_for(points);
    const gffx_buffer buffer =
        workspace_buffer(workspace, static_cast<uint64_t>(workspace.numel()));
    gffx_tensor_view points_view = arena.read(points);
    gffx_tensor_view matrices_view = arena.read(matrices);
    gffx_tensor_view offsets_view = arena.read(point_offsets);
    gffx_tensor_view homogeneous_view = arena.write(homogeneous);
    GFFX_CHECK(gffx_transforms_transform_points(
                   &points_view, &matrices_view, &offsets_view, &context, &homogeneous_view,
                   workspace.numel() > 0 ? &buffer : nullptr, &diagnostic.buffer),
               diagnostic.text());
}

void perspective_divide_out(
    const Tensor &homogeneous, double eps, Tensor &ndc, Tensor &valid, Tensor &workspace
) {
    ViewArena arena;
    Diagnostic diagnostic;
    gffx_execution_context context = context_for(homogeneous);
    const gffx_buffer buffer =
        workspace_buffer(workspace, static_cast<uint64_t>(workspace.numel()));
    gffx_tensor_view homogeneous_view = arena.read(homogeneous);
    gffx_tensor_view ndc_view = arena.write(ndc);
    gffx_tensor_view valid_view = arena.write(valid);
    GFFX_CHECK(gffx_transforms_perspective_divide(
                   &homogeneous_view, eps, &context, &ndc_view, &valid_view,
                   workspace.numel() > 0 ? &buffer : nullptr, &diagnostic.buffer),
               diagnostic.text());
}

void knn_out(
    const Tensor &query, const Tensor &reference, const Tensor &query_offsets,
    const Tensor &reference_offsets, int64_t neighbor_count, Tensor &distance_squared,
    Tensor &reference_index, Tensor &valid, Tensor &workspace
) {
    ViewArena arena;
    Diagnostic diagnostic;
    gffx_execution_context context = context_for(query);
    const gffx_buffer buffer =
        workspace_buffer(workspace, static_cast<uint64_t>(workspace.numel()));
    gffx_tensor_view query_view = arena.read(query);
    gffx_tensor_view reference_view = arena.read(reference);
    gffx_tensor_view query_offsets_view = arena.read(query_offsets);
    gffx_tensor_view reference_offsets_view = arena.read(reference_offsets);
    gffx_tensor_view distance_view = arena.write(distance_squared);
    gffx_tensor_view index_view = arena.write(reference_index);
    gffx_tensor_view valid_view = arena.write(valid);
    GFFX_CHECK(gffx_points_knn(&query_view, &reference_view, &query_offsets_view,
                               &reference_offsets_view, neighbor_count, &context, &distance_view,
                               &index_view, &valid_view,
                               workspace.numel() > 0 ? &buffer : nullptr, &diagnostic.buffer),
               diagnostic.text());
}

void closest_point_on_mesh_out(
    const Tensor &points, const Tensor &vertices, const Tensor &faces,
    const Tensor &point_offsets, const Tensor &vertex_offsets, const Tensor &face_offsets,
    double eps, Tensor &distance_squared, Tensor &face_index, Tensor &barycentric,
    Tensor &closest, Tensor &valid, Tensor &workspace
) {
    ViewArena arena;
    Diagnostic diagnostic;
    gffx_execution_context context = context_for(points);
    const gffx_buffer buffer =
        workspace_buffer(workspace, static_cast<uint64_t>(workspace.numel()));
    gffx_tensor_view points_view = arena.read(points);
    gffx_tensor_view vertices_view = arena.read(vertices);
    gffx_tensor_view faces_view = arena.read(faces);
    gffx_tensor_view point_offsets_view = arena.read(point_offsets);
    gffx_tensor_view vertex_offsets_view = arena.read(vertex_offsets);
    gffx_tensor_view face_offsets_view = arena.read(face_offsets);
    gffx_tensor_view distance_view = arena.write(distance_squared);
    gffx_tensor_view face_index_view = arena.write(face_index);
    gffx_tensor_view barycentric_view = arena.write(barycentric);
    gffx_tensor_view closest_view = arena.write(closest);
    gffx_tensor_view valid_view = arena.write(valid);
    GFFX_CHECK(gffx_points_closest_point_on_mesh(
                   &points_view, &vertices_view, &faces_view, &point_offsets_view,
                   &vertex_offsets_view, &face_offsets_view, eps, &context, &distance_view,
                   &face_index_view, &barycentric_view, &closest_view, &valid_view,
                   workspace.numel() > 0 ? &buffer : nullptr, &diagnostic.buffer),
               diagnostic.text());
}

void sample_surface_out(
    const Tensor &vertices, const Tensor &faces, const Tensor &vertex_offsets,
    const Tensor &face_offsets, int64_t sample_count, const Tensor &rng_key,
    const Tensor &rng_counter, double eps, Tensor &points, Tensor &face_index,
    Tensor &barycentric, Tensor &next_counter, Tensor &workspace
) {
    ViewArena arena;
    Diagnostic diagnostic;
    gffx_execution_context context = context_for(vertices);
    const gffx_buffer buffer =
        workspace_buffer(workspace, static_cast<uint64_t>(workspace.numel()));
    gffx_tensor_view vertices_view = arena.read(vertices);
    gffx_tensor_view faces_view = arena.read(faces);
    gffx_tensor_view vertex_offsets_view = arena.read(vertex_offsets);
    gffx_tensor_view face_offsets_view = arena.read(face_offsets);
    gffx_tensor_view key_view = arena.read(rng_key);
    gffx_tensor_view counter_view = arena.read(rng_counter);
    gffx_tensor_view points_view = arena.write(points);
    gffx_tensor_view face_index_view = arena.write(face_index);
    gffx_tensor_view barycentric_view = arena.write(barycentric);
    gffx_tensor_view next_counter_view = arena.write(next_counter);
    GFFX_CHECK(gffx_mesh_sample_surface(
                   &vertices_view, &faces_view, &vertex_offsets_view, &face_offsets_view,
                   sample_count, &key_view, &counter_view, eps, &context, &points_view,
                   &face_index_view, &barycentric_view, &next_counter_view,
                   workspace.numel() > 0 ? &buffer : nullptr, &diagnostic.buffer),
               diagnostic.text());
}

void rasterize_out(
    const Tensor &ndc_vertices, const Tensor &faces, const Tensor &vertex_offsets,
    const Tensor &face_offsets, int64_t image_height, int64_t image_width,
    int64_t faces_per_pixel, double blur_radius_px, int64_t cull_mode, double eps,
    Tensor &face_index, Tensor &barycentric, Tensor &depth, Tensor &signed_distance,
    Tensor &workspace
) {
    ViewArena arena;
    Diagnostic diagnostic;
    gffx_execution_context context = context_for(ndc_vertices);
    const gffx_buffer buffer =
        workspace_buffer(workspace, static_cast<uint64_t>(workspace.numel()));
    gffx_tensor_view ndc_view = arena.read(ndc_vertices);
    gffx_tensor_view faces_view = arena.read(faces);
    gffx_tensor_view vertex_offsets_view = arena.read(vertex_offsets);
    gffx_tensor_view face_offsets_view = arena.read(face_offsets);
    gffx_tensor_view face_index_view = arena.write(face_index);
    gffx_tensor_view barycentric_view = arena.write(barycentric);
    gffx_tensor_view depth_view = arena.write(depth);
    gffx_tensor_view signed_distance_view = arena.write(signed_distance);
    GFFX_CHECK(gffx_render_rasterize(
                   &ndc_view, &faces_view, &vertex_offsets_view, &face_offsets_view,
                   image_height, image_width, faces_per_pixel, blur_radius_px,
                   static_cast<uint32_t>(cull_mode), eps, &context, &face_index_view,
                   &barycentric_view, &depth_view, &signed_distance_view,
                   workspace.numel() > 0 ? &buffer : nullptr, &diagnostic.buffer),
               diagnostic.text());
}

void interpolate_out(
    const Tensor &face_index, const Tensor &barycentric, const Tensor &face_attributes,
    Tensor &attributes, Tensor &workspace
) {
    ViewArena arena;
    Diagnostic diagnostic;
    gffx_execution_context context = context_for(face_index);
    const gffx_buffer buffer =
        workspace_buffer(workspace, static_cast<uint64_t>(workspace.numel()));
    gffx_tensor_view face_index_view = arena.read(face_index);
    gffx_tensor_view barycentric_view = arena.read(barycentric);
    gffx_tensor_view face_attributes_view = arena.read(face_attributes);
    gffx_tensor_view attributes_view = arena.write(attributes);
    GFFX_CHECK(gffx_render_interpolate(&face_index_view, &barycentric_view,
                                       &face_attributes_view, &context, &attributes_view,
                                       workspace.numel() > 0 ? &buffer : nullptr,
                                       &diagnostic.buffer),
               diagnostic.text());
}

/* Workspace byte requirements, so a streaming host can size its scratch once at setup. Returned
 * as int64 because the Stable ABI schema has no unsigned integer type; the values are small. */
std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t> workspace_sizes(
    int64_t vertex_count, int64_t face_count, int64_t point_count, int64_t neighbor_count,
    int64_t sample_count, int64_t batch_count, int64_t image_height, int64_t image_width,
    int64_t faces_per_pixel, bool double_precision
) {
    Diagnostic diagnostic;
    gffx_execution_context context = cpu_context();
    const gffx_dtype dtype = double_precision ? GFFX_DTYPE_FLOAT64 : GFFX_DTYPE_FLOAT32;
    uint64_t bytes = 0, alignment = 0;
    int64_t normals = 0, transform = 0, divide = 0, nearest = 0, sample = 0, raster = 0;

    GFFX_CHECK(gffx_mesh_vertex_normals_workspace(vertex_count, face_count, dtype, &context,
                                                  &bytes, &alignment, &diagnostic.buffer),
               diagnostic.text());
    normals = static_cast<int64_t>(bytes);
    GFFX_CHECK(gffx_transforms_transform_points_workspace(point_count, batch_count, dtype,
                                                          &context, &bytes, &alignment,
                                                          &diagnostic.buffer),
               diagnostic.text());
    transform = static_cast<int64_t>(bytes);
    GFFX_CHECK(gffx_transforms_perspective_divide_workspace(point_count, dtype, &context, &bytes,
                                                            &alignment, &diagnostic.buffer),
               diagnostic.text());
    divide = static_cast<int64_t>(bytes);
    GFFX_CHECK(gffx_points_closest_point_on_mesh_workspace(point_count, vertex_count, face_count,
                                                           dtype, &context, &bytes, &alignment,
                                                           &diagnostic.buffer),
               diagnostic.text());
    nearest = static_cast<int64_t>(bytes);
    GFFX_CHECK(gffx_mesh_sample_surface_workspace(vertex_count, face_count, sample_count, dtype,
                                                  &context, &bytes, &alignment,
                                                  &diagnostic.buffer),
               diagnostic.text());
    sample = static_cast<int64_t>(bytes);
    GFFX_CHECK(gffx_render_rasterize_workspace(vertex_count, face_count, image_height,
                                               image_width, faces_per_pixel, dtype, &context,
                                               &bytes, &alignment, &diagnostic.buffer),
               diagnostic.text());
    raster = static_cast<int64_t>(bytes);
    (void)neighbor_count;
    return std::make_tuple(normals, transform, divide, nearest, sample, raster);
}

gffx_execution_context context_for(const Tensor &reference) {
    return reference.is_cuda() ? cuda_context(reference.get_device_index()) : cpu_context();
}

}  // namespace

STABLE_TORCH_LIBRARY(gffx_internal, m) {
    m.def("_foundation_probe() -> ()");
}

STABLE_TORCH_LIBRARY(gffx, m) {
    m.def("face_geometry(Tensor vertices, Tensor faces, float eps) -> (Tensor, Tensor, Tensor)");
    m.def(
        "face_geometry_backward(Tensor vertices, Tensor faces, float eps, "
        "Tensor grad_normals, Tensor grad_areas, bool has_grad_normals, bool has_grad_areas) "
        "-> Tensor");
    m.def(
        "face_geometry_out(Tensor vertices, Tensor faces, float eps, Tensor(a!) normals, "
        "Tensor(b!) areas, Tensor(c!) valid, Tensor(d!) workspace) -> ()");
    m.def("vertex_normals(Tensor vertices, Tensor faces, float eps, int weighting) -> Tensor");
    m.def(
        "vertex_normals_backward(Tensor vertices, Tensor faces, float eps, int weighting, "
        "Tensor grad_normals) -> Tensor");
    m.def("gather_faces(Tensor vertices, Tensor faces) -> Tensor");
    m.def("gather_faces_backward(Tensor vertices, Tensor faces, Tensor grad_gathered) -> Tensor");
    m.def(
        "transform_points(Tensor points, Tensor matrices, Tensor point_offsets) -> Tensor");
    m.def(
        "transform_points_backward(Tensor points, Tensor matrices, Tensor point_offsets, "
        "Tensor grad_homogeneous) -> (Tensor, Tensor)");
    m.def("perspective_divide(Tensor homogeneous, float eps) -> (Tensor, Tensor)");
    m.def(
        "perspective_divide_backward(Tensor homogeneous, float eps, Tensor grad_ndc) -> Tensor");
    m.def(
        "build_edge_topology(Tensor faces, Tensor face_offsets) "
        "-> (Tensor, Tensor, Tensor, Tensor)");
    m.def(
        "knn(Tensor query, Tensor reference, Tensor query_offsets, Tensor reference_offsets, "
        "int neighbor_count) -> (Tensor, Tensor, Tensor)");
    m.def(
        "knn_backward(Tensor query, Tensor reference, Tensor reference_index, Tensor valid, "
        "Tensor grad_distance_squared) -> (Tensor, Tensor)");
    m.def(
        "closest_point_on_mesh(Tensor points, Tensor vertices, Tensor faces, "
        "Tensor point_offsets, Tensor vertex_offsets, Tensor face_offsets, float eps) "
        "-> (Tensor, Tensor, Tensor, Tensor, Tensor)");
    m.def(
        "closest_point_on_mesh_backward(Tensor points, Tensor vertices, Tensor faces, "
        "Tensor face_index, Tensor barycentric, Tensor closest, Tensor valid, "
        "Tensor grad_distance_squared) -> (Tensor, Tensor)");
    m.def(
        "sample_surface(Tensor vertices, Tensor faces, Tensor vertex_offsets, "
        "Tensor face_offsets, int sample_count, Tensor rng_key, Tensor rng_counter, float eps) "
        "-> (Tensor, Tensor, Tensor, Tensor)");
    m.def(
        "sample_surface_backward(Tensor vertices, Tensor faces, Tensor face_index, "
        "Tensor barycentric, Tensor grad_points) -> Tensor");
    m.def(
        "rasterize(Tensor ndc_vertices, Tensor faces, Tensor vertex_offsets, "
        "Tensor face_offsets, int image_height, int image_width, int faces_per_pixel, "
        "float blur_radius_px, int cull_mode, float eps) -> (Tensor, Tensor, Tensor, Tensor)");
    m.def(
        "rasterize_backward(Tensor ndc_vertices, Tensor faces, int image_height, "
        "int image_width, Tensor face_index, Tensor grad_barycentric, Tensor grad_depth, "
        "Tensor grad_signed_distance) -> Tensor");
    m.def(
        "interpolate(Tensor face_index, Tensor barycentric, Tensor face_attributes) -> Tensor");
    m.def(
        "interpolate_backward(Tensor face_index, Tensor barycentric, Tensor face_attributes, "
        "Tensor grad_attributes) -> (Tensor, Tensor)");
    m.def("texture_pyramid(Tensor texture, int levels) -> (Tensor, Tensor)");
    m.def(
        "texture_pyramid_backward(Tensor level_offsets, int texture_height, int texture_width, "
        "int channel_count, Tensor grad_pyramid) -> Tensor");
    m.def(
        "texture(Tensor pyramid, Tensor level_offsets, int texture_height, int texture_width, "
        "int channel_count, Tensor coordinates, Tensor derivatives, Tensor lod, "
        "bool has_derivatives, bool has_lod, int filter, int mip_filter, int wrap_u, int wrap_v, "
        "Tensor border, bool has_border) -> Tensor");
    m.def(
        "texture_backward(Tensor pyramid, Tensor level_offsets, int texture_height, "
        "int texture_width, Tensor coordinates, Tensor derivatives, Tensor lod, "
        "bool has_derivatives, bool has_lod, int filter, int mip_filter, int wrap_u, int wrap_v, "
        "Tensor border, bool has_border, Tensor grad_samples) -> (Tensor, Tensor)");
    m.def("ply_probe(Tensor data) -> (int, int, int)");
    m.def("ply_read(Tensor data, bool double_precision) -> (Tensor, Tensor)");
    m.def(
        "mesh_validate(Tensor vertices, Tensor faces, Tensor vertex_offsets, "
        "Tensor face_offsets, float eps, int flags) -> Tensor");
    m.def(
        "vertex_normals_out(Tensor vertices, Tensor faces, float eps, int weighting, "
        "Tensor(a!) normals, Tensor(b!) workspace) -> ()");
    m.def(
        "gather_faces_out(Tensor vertices, Tensor faces, Tensor(a!) gathered, "
        "Tensor(b!) workspace) -> ()");
    m.def(
        "transform_points_out(Tensor points, Tensor matrices, Tensor point_offsets, "
        "Tensor(a!) homogeneous, Tensor(b!) workspace) -> ()");
    m.def(
        "perspective_divide_out(Tensor homogeneous, float eps, Tensor(a!) ndc, "
        "Tensor(b!) valid, Tensor(c!) workspace) -> ()");
    m.def(
        "knn_out(Tensor query, Tensor reference, Tensor query_offsets, "
        "Tensor reference_offsets, int neighbor_count, Tensor(a!) distance_squared, "
        "Tensor(b!) reference_index, Tensor(c!) valid, Tensor(d!) workspace) -> ()");
    m.def(
        "closest_point_on_mesh_out(Tensor points, Tensor vertices, Tensor faces, "
        "Tensor point_offsets, Tensor vertex_offsets, Tensor face_offsets, float eps, "
        "Tensor(a!) distance_squared, Tensor(b!) face_index, Tensor(c!) barycentric, "
        "Tensor(d!) closest, Tensor(e!) valid, Tensor(f!) workspace) -> ()");
    m.def(
        "sample_surface_out(Tensor vertices, Tensor faces, Tensor vertex_offsets, "
        "Tensor face_offsets, int sample_count, Tensor rng_key, Tensor rng_counter, float eps, "
        "Tensor(a!) points, Tensor(b!) face_index, Tensor(c!) barycentric, "
        "Tensor(d!) next_counter, Tensor(e!) workspace) -> ()");
    m.def(
        "rasterize_out(Tensor ndc_vertices, Tensor faces, Tensor vertex_offsets, "
        "Tensor face_offsets, int image_height, int image_width, int faces_per_pixel, "
        "float blur_radius_px, int cull_mode, float eps, Tensor(a!) face_index, "
        "Tensor(b!) barycentric, Tensor(c!) depth, Tensor(d!) signed_distance, "
        "Tensor(e!) workspace) -> ()");
    m.def(
        "interpolate_out(Tensor face_index, Tensor barycentric, Tensor face_attributes, "
        "Tensor(a!) attributes, Tensor(b!) workspace) -> ()");
    m.def(
        "workspace_sizes(int vertex_count, int face_count, int point_count, int neighbor_count, "
        "int sample_count, int batch_count, int image_height, int image_width, "
        "int faces_per_pixel, bool double_precision) -> (int, int, int, int, int, int)");
}

/* workspace_sizes takes no tensor, so the dispatcher has no device to route on and the CPU key
 * would never be reached. It is a pure function of integers and is registered device-agnostically. */
STABLE_TORCH_LIBRARY_IMPL(gffx, CompositeExplicitAutograd, m) {
    m.impl("workspace_sizes", TORCH_BOX(&workspace_sizes));
}

/* Only operations with a CUDA kernel are registered here. An unregistered operation makes torch
 * report that no CUDA implementation exists, which is earlier and clearer than dispatching into a
 * function that would fail on a NULL table entry. */
/*
 * The CUDA key registers the same functions as the CPU key.
 *
 * Each body takes its execution context from its own inputs, so one implementation serves both
 * devices and there is no second copy to drift. What differs between backends is decided inside
 * the core, which forwards to the plugin when the context is CUDA and runs the reference kernel
 * otherwise.
 *
 * Only operations with a CUDA kernel appear here. An unregistered operation makes torch itself
 * report that no CUDA implementation exists, which is earlier and clearer than dispatching into a
 * function that would fail on a NULL table entry deeper down.
 */
STABLE_TORCH_LIBRARY_IMPL(gffx, CUDA, m) {
    /* Every operation whose CUDA kernel is verified against the host by the device parity
     * fixtures. This list stood at eight forwards and no backwards until 2026-09-04, dating from
     * when only the order-independent group had kernels; twelve backwards and three further
     * forwards had landed since without being registered here, so a gradient on a CUDA tensor was
     * refused by torch even though the kernel existed and was bit-identical to the host. The
     * omission is invisible from C, where the operation table is complete, and invisible from a
     * CPU-only PyTorch build, where nothing dispatches to this key. */
    m.impl("face_geometry", TORCH_BOX(&face_geometry));
    m.impl("face_geometry_backward", TORCH_BOX(&face_geometry_backward));
    m.impl("vertex_normals", TORCH_BOX(&vertex_normals));
    m.impl("vertex_normals_backward", TORCH_BOX(&vertex_normals_backward));
    m.impl("gather_faces", TORCH_BOX(&gather_faces));
    m.impl("gather_faces_backward", TORCH_BOX(&gather_faces_backward));
    m.impl("transform_points", TORCH_BOX(&transform_points));
    m.impl("transform_points_backward", TORCH_BOX(&transform_points_backward));
    m.impl("perspective_divide", TORCH_BOX(&perspective_divide));
    m.impl("perspective_divide_backward", TORCH_BOX(&perspective_divide_backward));
    m.impl("knn", TORCH_BOX(&knn));
    m.impl("knn_backward", TORCH_BOX(&knn_backward));
    m.impl("closest_point_on_mesh", TORCH_BOX(&closest_point_on_mesh));
    m.impl("closest_point_on_mesh_backward", TORCH_BOX(&closest_point_on_mesh_backward));
    m.impl("sample_surface", TORCH_BOX(&sample_surface));
    m.impl("sample_surface_backward", TORCH_BOX(&sample_surface_backward));
    m.impl("rasterize", TORCH_BOX(&rasterize));
    m.impl("rasterize_backward", TORCH_BOX(&rasterize_backward));
    m.impl("interpolate", TORCH_BOX(&interpolate));
    m.impl("interpolate_backward", TORCH_BOX(&interpolate_backward));
    m.impl("texture_pyramid", TORCH_BOX(&texture_pyramid));
    m.impl("texture_pyramid_backward", TORCH_BOX(&texture_pyramid_backward));
    m.impl("texture", TORCH_BOX(&texture));
    m.impl("texture_backward", TORCH_BOX(&texture_backward));

    /* mesh.build_edge_topology is the one operation table entry without a CUDA forward, so it
     * stays unregistered here and torch reports the absence itself. The preallocated _out
     * variants also stay CPU-only: their kernels are the same functions and would very likely
     * work, but "would work" is not what this list is for. */
}

STABLE_TORCH_LIBRARY_IMPL(gffx, CPU, m) {
    m.impl("face_geometry", TORCH_BOX(&face_geometry));
    m.impl("face_geometry_backward", TORCH_BOX(&face_geometry_backward));
    m.impl("face_geometry_out", TORCH_BOX(&face_geometry_out));
    m.impl("vertex_normals", TORCH_BOX(&vertex_normals));
    m.impl("vertex_normals_backward", TORCH_BOX(&vertex_normals_backward));
    m.impl("gather_faces", TORCH_BOX(&gather_faces));
    m.impl("gather_faces_backward", TORCH_BOX(&gather_faces_backward));
    m.impl("transform_points", TORCH_BOX(&transform_points));
    m.impl("transform_points_backward", TORCH_BOX(&transform_points_backward));
    m.impl("perspective_divide", TORCH_BOX(&perspective_divide));
    m.impl("perspective_divide_backward", TORCH_BOX(&perspective_divide_backward));
    m.impl("build_edge_topology", TORCH_BOX(&build_edge_topology));
    m.impl("knn", TORCH_BOX(&knn));
    m.impl("knn_backward", TORCH_BOX(&knn_backward));
    m.impl("closest_point_on_mesh", TORCH_BOX(&closest_point_on_mesh));
    m.impl("closest_point_on_mesh_backward", TORCH_BOX(&closest_point_on_mesh_backward));
    m.impl("sample_surface", TORCH_BOX(&sample_surface));
    m.impl("sample_surface_backward", TORCH_BOX(&sample_surface_backward));
    m.impl("rasterize", TORCH_BOX(&rasterize));
    m.impl("rasterize_backward", TORCH_BOX(&rasterize_backward));
    m.impl("interpolate", TORCH_BOX(&interpolate));
    m.impl("interpolate_backward", TORCH_BOX(&interpolate_backward));
    m.impl("texture_pyramid", TORCH_BOX(&texture_pyramid));
    m.impl("texture_pyramid_backward", TORCH_BOX(&texture_pyramid_backward));
    m.impl("texture", TORCH_BOX(&texture));
    m.impl("texture_backward", TORCH_BOX(&texture_backward));
    m.impl("ply_probe", TORCH_BOX(&ply_probe));
    m.impl("ply_read", TORCH_BOX(&ply_read));
    m.impl("mesh_validate", TORCH_BOX(&mesh_validate));
    m.impl("vertex_normals_out", TORCH_BOX(&vertex_normals_out));
    m.impl("gather_faces_out", TORCH_BOX(&gather_faces_out));
    m.impl("transform_points_out", TORCH_BOX(&transform_points_out));
    m.impl("perspective_divide_out", TORCH_BOX(&perspective_divide_out));
    m.impl("knn_out", TORCH_BOX(&knn_out));
    m.impl("closest_point_on_mesh_out", TORCH_BOX(&closest_point_on_mesh_out));
    m.impl("sample_surface_out", TORCH_BOX(&sample_surface_out));
    m.impl("rasterize_out", TORCH_BOX(&rasterize_out));
    m.impl("interpolate_out", TORCH_BOX(&interpolate_out));
}

static struct PyModuleDef gffx_torch_module = {
    PyModuleDef_HEAD_INIT,
    "_torch",
    "Private GFFX PyTorch Stable-ABI registration loader.",
    0,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit__torch(void) { return PyModule_Create(&gffx_torch_module); }
