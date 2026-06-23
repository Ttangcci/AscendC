#include "softmax_cross_entropy_with_logits.h"

enum class SoftmaxCrossEntropyWithLogitsTilingKey : uint32_t {
    TILING_KEY_FLOAT   = 0,
    TILING_KEY_FLOAT16 = 1,
    TILING_KEY_BF16    = 2,
};

template <uint32_t schMode>
__global__ __aicore__ void softmax_cross_entropy_with_logits(
    GM_ADDR features, GM_ADDR labels, GM_ADDR loss, GM_ADDR backprop, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SoftmaxCrossEntropyWithLogitsTilingData);
    GET_TILING_DATA_WITH_STRUCT(SoftmaxCrossEntropyWithLogitsTilingData, tilingData, tiling);

    if constexpr (schMode == static_cast<uint32_t>(SoftmaxCrossEntropyWithLogitsTilingKey::TILING_KEY_FLOAT)) {
        NsSoftmaxCrossEntropyWithLogits::KernelSoftmaxCrossEntropyWithLogits<float> op;
        op.Init(features, labels, loss, backprop,
                tilingData.batchSize, tilingData.numClasses,
                tilingData.blockLength, tilingData.tileNum, tilingData.tileLength);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(SoftmaxCrossEntropyWithLogitsTilingKey::TILING_KEY_FLOAT16)) {
        NsSoftmaxCrossEntropyWithLogits::KernelSoftmaxCrossEntropyWithLogits<half> op;
        op.Init(features, labels, loss, backprop,
                tilingData.batchSize, tilingData.numClasses,
                tilingData.blockLength, tilingData.tileNum, tilingData.tileLength);
        op.Process();
    }
    if constexpr (schMode == static_cast<uint32_t>(SoftmaxCrossEntropyWithLogitsTilingKey::TILING_KEY_BF16)) {
        NsSoftmaxCrossEntropyWithLogits::KernelSoftmaxCrossEntropyWithLogits<bfloat16_t> op;
        op.Init(features, labels, loss, backprop,
                tilingData.batchSize, tilingData.numClasses,
                tilingData.blockLength, tilingData.tileNum, tilingData.tileLength);
        op.Process();
    }
}
// #include "softmax_cross_entropy_with_logits.h"

// template <uint32_t schMode>
// __global__ __aicore__ void softmax_cross_entropy_with_logits(
//     GM_ADDR features, GM_ADDR labels, GM_ADDR loss, GM_ADDR backprop, GM_ADDR workspace, GM_ADDR tiling)
// {
//     REGISTER_TILING_DEFAULT(SoftmaxCrossEntropyWithLogitsTilingData);
//     GET_TILING_DATA_WITH_STRUCT(SoftmaxCrossEntropyWithLogitsTilingData, tilingData, tiling);
//     NsSoftmaxCrossEntropyWithLogits::KernelSoftmaxCrossEntropyWithLogits<float> op;
//     op.Init(features, labels, loss, backprop,
//             tilingData.batchSize, tilingData.numClasses,
//             tilingData.blockLength, tilingData.tileNum, tilingData.tileLength);
//     op.Process();
// }