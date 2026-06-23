/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file softmax_cross_entropy_with_logits_graph_infer.cpp
 * \brief softmax_cross_entropy_with_logits operator graph infer resource
 */
#include "register/op_impl_registry.h"
#include "log/log.h"

namespace ops {
using namespace ge;

static constexpr int64_t IDX_0 = 0;
static constexpr int64_t IDX_1 = 1;

static ge::graphStatus InferDataTypeSoftmaxCrossEntropyWithLogits(gert::InferDataTypeContext* context)
{
    OP_LOGD(context->GetNodeName(), "Begin to do InferDataTypeSoftmaxCrossEntropyWithLogits");

    // 输入 features 的 dtype 决定输出 dtype
    ge::DataType inputDtype = context->GetInputDataType(IDX_0);

    // loss 和 backprop 的 dtype 和输入一致
    context->SetOutputDataType(IDX_0, inputDtype);
    context->SetOutputDataType(IDX_1, inputDtype);

    OP_LOGD(context->GetNodeName(), "End to do InferDataTypeSoftmaxCrossEntropyWithLogits");
    return GRAPH_SUCCESS;
}

IMPL_OP(SoftmaxCrossEntropyWithLogits).InferDataType(InferDataTypeSoftmaxCrossEntropyWithLogits);

} // namespace ops