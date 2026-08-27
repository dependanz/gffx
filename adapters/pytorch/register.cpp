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

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

#include <gffx/execution.h>
#include <gffx/mesh.h>
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
    view.device_type = GFFX_DEVICE_CPU;
    view.device_index = 0;
    view.flags = flags;
    return view;
}

/* The workspace is a torch uint8 tensor, so every allocation stays on the caller's allocator and
 * the functional surface's internal scratch is the same object type the streaming surface asks a
 * caller to supply. */
Tensor make_workspace(
    const Tensor &reference, int64_t vertex_count, int64_t face_count, gffx_dtype dtype,
    uint64_t *out_bytes
) {
    gffx_execution_context context = cpu_context();
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
    buffer.device_type = GFFX_DEVICE_CPU;
    buffer.device_index = 0;
    return buffer;
}

/* Shared by the functional and the preallocated surfaces so the two cannot drift: the streaming
 * entry point is the same kernel call with the allocation removed. */
void run_forward(
    const Tensor &vertices, const Tensor &faces, double eps,
    Tensor &normals, Tensor &areas, Tensor &valid, const gffx_buffer *workspace
) {
    gffx_execution_context context = cpu_context();
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
    gffx_execution_context context = cpu_context();
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
}

STABLE_TORCH_LIBRARY_IMPL(gffx, CPU, m) {
    m.impl("face_geometry", TORCH_BOX(&face_geometry));
    m.impl("face_geometry_backward", TORCH_BOX(&face_geometry_backward));
    m.impl("face_geometry_out", TORCH_BOX(&face_geometry_out));
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
