/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_TILING_DATA_H
#define SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_TILING_DATA_H

struct SoftmaxCrossEntropyWithLogitsTilingData {
    uint64_t batchSize;
    uint64_t numClasses;
    uint64_t blockLength;
    uint64_t tileNum;
    uint64_t tileLength;
};

#endif