/* Stable-ABI allocation/dispatch only; arithmetic stays in the shared C family. */
Tensor image_transfer_call(const Tensor &input, const Tensor *gradient, int64_t transfer) {
    if (transfer < 0 || transfer > 1) raise_internal("unknown image transfer mode");
    GffxCudaDeviceScope device_scope(input);
    ViewArena arena;
    Diagnostic diagnostic;
    auto context = context_for(input);
    uint64_t bytes = 0, alignment = 0;
    GFFX_CHECK(gffx_image_transfer_workspace(input.numel(), dtype_of(input), &context,
        &bytes, &alignment, &diagnostic.buffer), diagnostic.text());
    Tensor workspace = workspace_of(input, bytes);
    auto buffer = workspace_buffer(workspace, bytes);
    Tensor output = torch::stable::new_empty(input, input.sizes());
    auto iv = arena.read(input), ov = arena.write(output);
    if (gradient == nullptr) {
        GFFX_CHECK(gffx_image_transfer(&iv, static_cast<uint32_t>(transfer), &context,
            &ov, &buffer, &diagnostic.buffer), diagnostic.text());
    } else {
        auto gv = arena.read(*gradient);
        GFFX_CHECK(gffx_image_transfer_backward(&iv, &gv, static_cast<uint32_t>(transfer),
            &context, &ov, &buffer, &diagnostic.buffer), diagnostic.text());
    }
    device_scope.close();
    return output;
}

Tensor image_transfer(const Tensor &input, int64_t transfer) {
    return image_transfer_call(input, nullptr, transfer);
}

Tensor image_transfer_backward(const Tensor &input, const Tensor &gradient, int64_t transfer) {
    return image_transfer_call(input, &gradient, transfer);
}
