#ifndef SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_H
#define SOFTMAX_CROSS_ENTROPY_WITH_LOGITS_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "softmax_cross_entropy_with_logits_tiling_data.h"
#include "softmax_cross_entropy_with_logits_tiling_key.h"

namespace NsSoftmaxCrossEntropyWithLogits {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class KernelSoftmaxCrossEntropyWithLogits {
public:
    __aicore__ inline KernelSoftmaxCrossEntropyWithLogits() {};
    __aicore__ inline void Init(GM_ADDR features, GM_ADDR labels, GM_ADDR loss, GM_ADDR backprop,
                                uint64_t batchSize, uint64_t numClasses,
                                uint64_t blockLength, uint64_t tileNum, uint64_t tileLength);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int32_t rowOffset, int32_t rowCount);
    __aicore__ inline void Compute(int32_t rowOffset, int32_t rowCount);
    __aicore__ inline void CopyOut(int32_t rowOffset, int32_t rowCount);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> featuresQueue;
    TQue<QuePosition::VECIN, BUFFER_NUM> labelsQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> lossQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> backpropQueue;
    TBuf<QuePosition::VECCALC> lnBuf;
    TBuf<QuePosition::VECCALC> featuresFloatBuf;
    TBuf<QuePosition::VECCALC> labelsFloatBuf;
    TBuf<QuePosition::VECCALC> tmpFloatBuf;
    TBuf<QuePosition::VECCALC> rowTmpBuf;   // 每行单独使用的小buffer，从offset0开始，避免带偏移调用向量API
    TBuf<QuePosition::VECCALC> lossOneBuf;  // 专门用于half/bf16场景下，单个loss值的安全Cast中转

    GlobalTensor<T> featuresGm;
    GlobalTensor<T> labelsGm;
    GlobalTensor<T> lossGm;
    GlobalTensor<T> backpropGm;

    uint64_t batchSize;
    uint64_t numClasses;
    uint64_t blockLength;
    uint64_t tileNum;
    uint64_t tileLength;
    uint64_t blockOffset;
    uint64_t validRows;
};

template <typename T>
__aicore__ inline void KernelSoftmaxCrossEntropyWithLogits<T>::Init(
    GM_ADDR features, GM_ADDR labels, GM_ADDR loss, GM_ADDR backprop,
    uint64_t batchSize, uint64_t numClasses,
    uint64_t blockLength, uint64_t tileNum, uint64_t tileLength)
{
    this->batchSize   = batchSize;
    this->numClasses  = numClasses;
    this->blockLength = blockLength;
    this->tileNum     = tileNum;
    this->tileLength  = tileLength;
    this->blockOffset = blockLength * GetBlockIdx();

    if (this->blockOffset >= this->batchSize) {
        this->validRows = 0;
    } else {
        uint64_t remain = this->batchSize - this->blockOffset;
        this->validRows = (this->blockLength < remain) ? this->blockLength : remain;
    }

    featuresGm.SetGlobalBuffer((__gm__ T*)features + blockOffset * numClasses, validRows * numClasses);
    labelsGm.SetGlobalBuffer((__gm__ T*)labels + blockOffset * numClasses, validRows * numClasses);
    lossGm.SetGlobalBuffer((__gm__ T*)loss + blockOffset, validRows);
    backpropGm.SetGlobalBuffer((__gm__ T*)backprop + blockOffset * numClasses, validRows * numClasses);

    pipe.InitBuffer(featuresQueue, BUFFER_NUM, tileLength * numClasses * sizeof(T));
    pipe.InitBuffer(labelsQueue,   BUFFER_NUM, tileLength * numClasses * sizeof(T));
    pipe.InitBuffer(lossQueue,     BUFFER_NUM, tileLength * sizeof(T));
    pipe.InitBuffer(backpropQueue, BUFFER_NUM, tileLength * numClasses * sizeof(T));
    pipe.InitBuffer(lnBuf,             32 * sizeof(float));
    pipe.InitBuffer(featuresFloatBuf,  tileLength * numClasses * sizeof(float));
    pipe.InitBuffer(labelsFloatBuf,    tileLength * numClasses * sizeof(float));
    pipe.InitBuffer(tmpFloatBuf,       tileLength * numClasses * sizeof(float));
    pipe.InitBuffer(rowTmpBuf,         numClasses * sizeof(float));
    pipe.InitBuffer(lossOneBuf, 32);  // 固定32字节，够存1个任意类型的元素，且满足对齐
}

template <typename T>
__aicore__ inline void KernelSoftmaxCrossEntropyWithLogits<T>::CopyIn(int32_t rowOffset, int32_t rowCount)
{
    LocalTensor<T> featuresLocal = featuresQueue.AllocTensor<T>();
    LocalTensor<T> labelsLocal   = labelsQueue.AllocTensor<T>();

    DataCopyExtParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = rowCount * numClasses * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    DataCopyPadExtParams<T> padParams{false, 0, 0, 0};

    DataCopyPad(featuresLocal, featuresGm[rowOffset * numClasses], copyParams, padParams);
    DataCopyPad(labelsLocal,   labelsGm[rowOffset * numClasses],   copyParams, padParams);

    featuresQueue.EnQue(featuresLocal);
    labelsQueue.EnQue(labelsLocal);
}

template <typename T>
__aicore__ inline void KernelSoftmaxCrossEntropyWithLogits<T>::Compute(int32_t rowOffset, int32_t rowCount)
{
    LocalTensor<T> featuresLocal = featuresQueue.DeQue<T>();
    LocalTensor<T> labelsLocal   = labelsQueue.DeQue<T>();
    LocalTensor<T> backpropLocal = backpropQueue.AllocTensor<T>();
    LocalTensor<T> lossLocal     = lossQueue.AllocTensor<T>();

    LocalTensor<float> featuresFloat = featuresFloatBuf.Get<float>();
    LocalTensor<float> labelsFloat   = labelsFloatBuf.Get<float>();
    LocalTensor<float> tmpFloat      = tmpFloatBuf.Get<float>();
    LocalTensor<float> rowTmp        = rowTmpBuf.Get<float>();

    int32_t totalCount = rowCount * (int32_t)numClasses;
    int32_t nc = (int32_t)numClasses;

    if constexpr (std::is_same_v<T, float>) {
        for (int32_t k = 0; k < totalCount; k++) {
            featuresFloat.SetValue(k, featuresLocal.GetValue(k));
            labelsFloat.SetValue(k, labelsLocal.GetValue(k));
        }
    } else {
        Cast(featuresFloat, featuresLocal, RoundMode::CAST_NONE, totalCount);
        Cast(labelsFloat,   labelsLocal,   RoundMode::CAST_NONE, totalCount);
    }

    for (int32_t i = 0; i < rowCount; i++) {
        int32_t rowStart = i * nc;

        // 把这一行数据拷到独立的rowTmp（offset从0开始），避免后续Exp/Ln带偏移调用
        for (int32_t j = 0; j < nc; j++) {
            rowTmp.SetValue(j, featuresFloat.GetValue(rowStart + j));
        }

        float maxVal = rowTmp.GetValue(0);
        for (int32_t j = 1; j < nc; j++) {
            float v = rowTmp.GetValue(j);
            if (v > maxVal) maxVal = v;
        }
        for (int32_t j = 0; j < nc; j++) {
            rowTmp.SetValue(j, rowTmp.GetValue(j) - maxVal);
        }
        Exp(rowTmp, rowTmp, nc);  // 不带偏移，安全调用

        float sumVal = 0.0f;
        for (int32_t j = 0; j < nc; j++) {
            sumVal += rowTmp.GetValue(j);
        }

        LocalTensor<float> lnTensor = lnBuf.Get<float>();
        lnTensor.SetValue(0, sumVal);
        Ln(lnTensor, lnTensor, 1);  // 不带偏移，安全调用
        float logSumVal = lnTensor.GetValue(0);

        float lossVal = 0.0f;
        for (int32_t j = 0; j < nc; j++) {
            float softmax_j = rowTmp.GetValue(j) / sumVal;
            float label_j   = labelsFloat.GetValue(rowStart + j);
            float feat_j    = featuresFloat.GetValue(rowStart + j);
            // 把backprop结果直接写回tmpFloat对应的全局位置（这里是SetValue标量操作，不是向量API，没有偏移限制问题）
            tmpFloat.SetValue(rowStart + j, softmax_j - label_j);
            lossVal += -label_j * (feat_j - maxVal - logSumVal);
        }

        if constexpr (std::is_same_v<T, float>) {
            for (int32_t j = 0; j < nc; j++) {
                backpropLocal.SetValue(rowStart + j, tmpFloat.GetValue(rowStart + j));
            }
            lossLocal.SetValue(i, lossVal);
        } else {
            LocalTensor<float> lossFloatTensor = lnBuf.Get<float>();
            LocalTensor<T> lossOneTensor = lossOneBuf.Get<T>();  // 用独立buffer，不复用lnBuf
            lossFloatTensor.SetValue(0, lossVal);
            Cast(lossOneTensor, lossFloatTensor, RoundMode::CAST_ROUND, 1);  // 不带偏移，安全调用
            lossLocal.SetValue(i, lossOneTensor.GetValue(0));  // 标量赋值，写到正确位置，没有偏移问题
        }
    }

    if constexpr (!std::is_same_v<T, float>) {
        Cast(backpropLocal, tmpFloat, RoundMode::CAST_ROUND, totalCount);
    }

    backpropQueue.EnQue<T>(backpropLocal);
    lossQueue.EnQue<T>(lossLocal);
    featuresQueue.FreeTensor(featuresLocal);
    labelsQueue.FreeTensor(labelsLocal);
}

template <typename T>
__aicore__ inline void KernelSoftmaxCrossEntropyWithLogits<T>::CopyOut(int32_t rowOffset, int32_t rowCount)
{
    LocalTensor<T> backpropLocal = backpropQueue.DeQue<T>();
    LocalTensor<T> lossLocal     = lossQueue.DeQue<T>();

    DataCopyExtParams copyParamsBackprop;
    copyParamsBackprop.blockCount = 1;
    copyParamsBackprop.blockLen = rowCount * numClasses * sizeof(T);
    copyParamsBackprop.srcStride = 0;
    copyParamsBackprop.dstStride = 0;
    DataCopyPad(backpropGm[rowOffset * numClasses], backpropLocal, copyParamsBackprop);

    DataCopyExtParams copyParamsLoss;
    copyParamsLoss.blockCount = 1;
    copyParamsLoss.blockLen = rowCount * sizeof(T);
    copyParamsLoss.srcStride = 0;
    copyParamsLoss.dstStride = 0;
    DataCopyPad(lossGm[rowOffset], lossLocal, copyParamsLoss);

    backpropQueue.FreeTensor(backpropLocal);
    lossQueue.FreeTensor(lossLocal);
}

template <typename T>
__aicore__ inline void KernelSoftmaxCrossEntropyWithLogits<T>::Process()
{
    if (validRows == 0) {
        return;
    }

    int32_t fullTileNum = (int32_t)(validRows / tileLength);
    for (int32_t i = 0; i < fullTileNum; i++) {
        int32_t rowOffset = i * tileLength;
        CopyIn(rowOffset, tileLength);
        Compute(rowOffset, tileLength);
        CopyOut(rowOffset, tileLength);
    }
    int32_t tail = (int32_t)validRows - fullTileNum * (int32_t)tileLength;
    if (tail > 0) {
        int32_t rowOffset = fullTileNum * (int32_t)tileLength;
        CopyIn(rowOffset, tail);
        Compute(rowOffset, tail);
        CopyOut(rowOffset, tail);
    }
}

} // namespace NsSoftmaxCrossEntropyWithLogits
#endif