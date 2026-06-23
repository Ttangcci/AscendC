/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */

#include "log/log.h"
#include "util/math_util.h"
#include "tiling/platform/platform_ascendc.h"
#include "register/op_impl_registry.h"
#include "util/platform_util.h"
#include "../op_kernel/softmax_cross_entropy_with_logits_tiling_data.h"
#include "../op_kernel/softmax_cross_entropy_with_logits_tiling_key.h"

namespace optiling {

struct SoftmaxCrossEntropyWithLogitsCompileInfo {};

static ge::graphStatus TilingParseForSoftmaxCrossEntropyWithLogits(
    [[maybe_unused]] gert::TilingParseContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus SoftmaxCrossEntropyWithLogitsTilingFunc(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE(context, "context is nullptr"), return ge::GRAPH_FAILED);

    SoftmaxCrossEntropyWithLogitsTilingData* tiling =
        context->GetTilingData<SoftmaxCrossEntropyWithLogitsTilingData>();
    OP_CHECK_NULL_WITH_CONTEXT(context, tiling);
    OP_CHECK_IF(
        memset_s(tiling, sizeof(SoftmaxCrossEntropyWithLogitsTilingData), 0,
                 sizeof(SoftmaxCrossEntropyWithLogitsTilingData)) != EOK,
        OP_LOGE(context, "memset tiling data error"), return ge::GRAPH_FAILED);

    auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint64_t ubSize = 0;
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
    int64_t coreNum = ascendcPlatform.GetCoreNum();
    OP_CHECK_IF(coreNum <= 0, OP_LOGE(context, "coreNum <= 0"), return ge::GRAPH_FAILED);
    OP_CHECK_IF(ubSize == 0, OP_LOGE(context, "ubSize == 0"), return ge::GRAPH_FAILED);

    const gert::StorageShape* featuresStorageShape = context->GetInputShape(0);
    OP_CHECK_NULL_WITH_CONTEXT(context, featuresStorageShape);
    const gert::Shape& featuresShape = featuresStorageShape->GetStorageShape();

    int64_t dimNum = featuresShape.GetDimNum();
    OP_CHECK_IF(dimNum < 2, OP_LOGE(context, "features must be at least 2D"), return ge::GRAPH_FAILED);

    int64_t batchSize = 1;
    for (int64_t i = 0; i < dimNum - 1; i++) {
        batchSize *= featuresShape.GetDim(i);
    }
    int64_t numClasses = featuresShape.GetDim(dimNum - 1);

    auto dtype = context->GetInputDesc(0)->GetDataType();
    int64_t typeLength = (dtype == ge::DT_FLOAT) ? 4 : 2;

    int64_t bytesPerRow = 6 * numClasses * typeLength + typeLength * 2 + numClasses * 12;
    int64_t maxRowsPerTile = static_cast<int64_t>(ubSize) / bytesPerRow;
    OP_LOGE(context, "DEBUG: ubSize=%lu, coreNum=%ld, bytesPerRow=%ld, maxRowsPerTile=%ld", ubSize, coreNum, bytesPerRow, maxRowsPerTile);
    if (maxRowsPerTile <= 0) maxRowsPerTile = 1;

    int64_t usedCoreNum = batchSize < coreNum ? batchSize : coreNum; //实际用到的核数
    int64_t blockLength = (batchSize + usedCoreNum - 1) / usedCoreNum; //每个核分摊到多少行数据

    int64_t tileNum = (blockLength + maxRowsPerTile - 1) / maxRowsPerTile; //每个核要分几批搬运数据
    if (tileNum <= 0) tileNum = 1;

    int64_t tileLength = blockLength / tileNum; //每一批处理多少行
    if (tileLength <= 0) tileLength = 1;

    tiling->batchSize   = static_cast<uint64_t>(batchSize);
    tiling->numClasses  = static_cast<uint64_t>(numClasses);
    tiling->blockLength = static_cast<uint64_t>(blockLength);
    
    tiling->tileNum     = static_cast<uint64_t>(tileNum);
    tiling->tileLength  = static_cast<uint64_t>(tileLength);
    
    // tiling->tileLength  = 1;
    // tiling->tileNum     = static_cast<uint64_t>(blockLength);

    size_t* workspaces = context->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context, workspaces);
    workspaces[0] = ascendcPlatform.GetLibApiWorkSpaceSize();

    uint64_t tilingKey = 0;
    if (dtype == ge::DT_FLOAT) {
        tilingKey = GET_TPL_TILING_KEY(0);
    } else if (dtype == ge::DT_FLOAT16) {
        tilingKey = GET_TPL_TILING_KEY(1);
    } else if (dtype == ge::DT_BF16) {
        tilingKey = GET_TPL_TILING_KEY(2);
    }
    // uint64_t tilingKey = GET_TPL_TILING_KEY(0);
    context->SetTilingKey(tilingKey);
    
    context->SetBlockDim(usedCoreNum);

    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(SoftmaxCrossEntropyWithLogits)
    .Tiling(SoftmaxCrossEntropyWithLogitsTilingFunc)
    .TilingParse<SoftmaxCrossEntropyWithLogitsCompileInfo>(TilingParseForSoftmaxCrossEntropyWithLogits);

} // namespace optiling