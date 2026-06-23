/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include "acl/acl.h"
#include "aclnnop/aclnn_softmax_cross_entropy_with_logits.h"

#define CHECK_RET(cond, return_expr) \
  do {                               \
    if (!(cond)) {                   \
      return_expr;                   \
    }                                \
  } while (0)

#define LOG_PRINT(message, ...)     \
  do {                              \
    printf(message, ##__VA_ARGS__); \
  } while (0)

uint16_t FloatToBf16(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  return static_cast<uint16_t>(bits >> 16);
}

float Bf16ToFloat(uint16_t bf16) {
  uint32_t bits = static_cast<uint32_t>(bf16) << 16;
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

int64_t GetShapeSize(const std::vector<int64_t>& shape) {
  int64_t shapeSize = 1;
  for (auto i : shape) {
    shapeSize *= i;
  }
  return shapeSize;
}

int Init(int32_t deviceId, aclrtStream* stream) {
  auto ret = aclInit(nullptr);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
  ret = aclrtSetDevice(deviceId);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
  ret = aclrtCreateStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
  return 0;
}

template <typename T>
int CreateAclTensor(const std::vector<T>& hostData, const std::vector<int64_t>& shape, void** deviceAddr,
                    aclDataType dataType, aclTensor** tensor) {
  auto size = GetShapeSize(shape) * sizeof(T);
  auto ret = aclrtMalloc(deviceAddr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMalloc failed. ERROR: %d\n", ret); return ret);
  ret = aclrtMemcpy(*deviceAddr, size, hostData.data(), size, ACL_MEMCPY_HOST_TO_DEVICE);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy failed. ERROR: %d\n", ret); return ret);

  std::vector<int64_t> strides(shape.size(), 1);
  for (int64_t i = shape.size() - 2; i >= 0; i--) {
    strides[i] = shape[i + 1] * strides[i + 1];
  }

  *tensor = aclCreateTensor(shape.data(), shape.size(), dataType, strides.data(), 0, aclFormat::ACL_FORMAT_ND,
                            shape.data(), shape.size(), *deviceAddr);
  return 0;
}

int main() {
  int32_t deviceId = 0;
  aclrtStream stream;
  auto ret = Init(deviceId, &stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("Init acl failed. ERROR: %d\n", ret); return ret);

  int64_t batch = 200;
  int64_t numClasses = 10;
  std::vector<int64_t> featuresShape = {batch, numClasses};
  std::vector<int64_t> labelsShape = {batch, numClasses};
  std::vector<int64_t> lossShape = {batch};
  std::vector<int64_t> backpropShape = {batch, numClasses};

  std::vector<float> featuresHostFloat(batch * numClasses);
  std::vector<float> labelsHostFloat(batch * numClasses, 0.0f);

  srand(42);
  for (int64_t i = 0; i < batch; i++) {
    for (int64_t j = 0; j < numClasses; j++) {
      featuresHostFloat[i * numClasses + j] = static_cast<float>((j - numClasses / 2) + (i % 5) * 0.3f);
    }
    int64_t labelClass = rand() % numClasses;
    labelsHostFloat[i * numClasses + labelClass] = 1.0f;
  }

  std::vector<uint16_t> featuresHostData(batch * numClasses);
  std::vector<uint16_t> labelsHostData(batch * numClasses);
  for (int64_t k = 0; k < batch * numClasses; k++) {
    featuresHostData[k] = FloatToBf16(featuresHostFloat[k]);
    labelsHostData[k] = FloatToBf16(labelsHostFloat[k]);
  }

  std::vector<uint16_t> lossHostData(batch, FloatToBf16(0.0f));
  std::vector<uint16_t> backpropHostData(batch * numClasses, FloatToBf16(0.0f));

  void* featuresDeviceAddr = nullptr;
  void* labelsDeviceAddr = nullptr;
  void* lossDeviceAddr = nullptr;
  void* backpropDeviceAddr = nullptr;
  aclTensor* features = nullptr;
  aclTensor* labels = nullptr;
  aclTensor* loss = nullptr;
  aclTensor* backprop = nullptr;

  ret = CreateAclTensor(featuresHostData, featuresShape, &featuresDeviceAddr, aclDataType::ACL_BF16, &features);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(labelsHostData, labelsShape, &labelsDeviceAddr, aclDataType::ACL_BF16, &labels);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(lossHostData, lossShape, &lossDeviceAddr, aclDataType::ACL_BF16, &loss);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(backpropHostData, backpropShape, &backpropDeviceAddr, aclDataType::ACL_BF16, &backprop);
  CHECK_RET(ret == ACL_SUCCESS, return ret);

  uint64_t workspaceSize = 0;
  aclOpExecutor* executor;
  ret = aclnnSoftmaxCrossEntropyWithLogitsGetWorkspaceSize(features, labels, loss, backprop, &workspaceSize, &executor);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("GetWorkspaceSize failed. ERROR: %d\n", ret); return ret);

  void* workspaceAddr = nullptr;
  if (workspaceSize > 0) {
    ret = aclrtMalloc(&workspaceAddr, workspaceSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return ret);
  }

  ret = aclnnSoftmaxCrossEntropyWithLogits(workspaceAddr, workspaceSize, executor, stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclnnSoftmaxCrossEntropyWithLogits failed. ERROR: %d\n", ret); return ret);

  ret = aclrtSynchronizeStream(stream);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", ret); return ret);

  std::vector<uint16_t> lossResult(batch, FloatToBf16(0.0f));
  ret = aclrtMemcpy(lossResult.data(), lossResult.size() * sizeof(uint16_t), lossDeviceAddr,
                    batch * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy loss failed. ERROR: %d\n", ret); return ret);

  std::vector<uint16_t> backpropResult(batch * numClasses, FloatToBf16(0.0f));
  ret = aclrtMemcpy(backpropResult.data(), backpropResult.size() * sizeof(uint16_t), backpropDeviceAddr,
                    batch * numClasses * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy backprop failed. ERROR: %d\n", ret); return ret);

  std::vector<float> goldenLoss(batch);
  std::vector<float> goldenBackprop(batch * numClasses);
  for (int64_t i = 0; i < batch; i++) {
    float maxVal = featuresHostFloat[i * numClasses];
    for (int64_t j = 1; j < numClasses; j++) {
      float v = featuresHostFloat[i * numClasses + j];
      if (v > maxVal) maxVal = v;
    }
    float sumExp = 0.0f;
    std::vector<float> expVals(numClasses);
    for (int64_t j = 0; j < numClasses; j++) {
      expVals[j] = std::exp(featuresHostFloat[i * numClasses + j] - maxVal);
      sumExp += expVals[j];
    }
    float logSum = std::log(sumExp);
    float lossVal = 0.0f;
    for (int64_t j = 0; j < numClasses; j++) {
      float softmaxVal = expVals[j] / sumExp;
      float labelVal = labelsHostFloat[i * numClasses + j];
      goldenBackprop[i * numClasses + j] = softmaxVal - labelVal;
      lossVal += -labelVal * (featuresHostFloat[i * numClasses + j] - maxVal - logSum);
    }
    goldenLoss[i] = lossVal;
  }

  float maxLossErr = 0.0f;
  float maxBackpropErr = 0.0f;
  for (int64_t i = 0; i < batch; i++) {
    float deviceVal = Bf16ToFloat(lossResult[i]);
    float err = std::fabs(deviceVal - goldenLoss[i]);
    if (err > maxLossErr) maxLossErr = err;
  }
  for (int64_t i = 0; i < batch * numClasses; i++) {
    float deviceVal = Bf16ToFloat(backpropResult[i]);
    float err = std::fabs(deviceVal - goldenBackprop[i]);
    if (err > maxBackpropErr) maxBackpropErr = err;
  }

  LOG_PRINT("batch=%ld, numClasses=%ld (bf16)\n", batch, numClasses);
  LOG_PRINT("Max loss error: %e\n", maxLossErr);
  LOG_PRINT("Max backprop error: %e\n", maxBackpropErr);

  LOG_PRINT("\n--- First 5 rows ---\n");
  for (int64_t i = 0; i < 5 && i < batch; i++) {
    LOG_PRINT("loss[%ld]: device=%f, golden=%f\n", i, Bf16ToFloat(lossResult[i]), goldenLoss[i]);
  }
  LOG_PRINT("\n--- Last 5 rows ---\n");
  for (int64_t i = batch - 5; i < batch; i++) {
    if (i < 0) continue;
    LOG_PRINT("loss[%ld]: device=%f, golden=%f\n", i, Bf16ToFloat(lossResult[i]), goldenLoss[i]);
  }

  // bf16精度比float16更低，阈值再放宽一些
  if (maxLossErr < 1e-1f && maxBackpropErr < 1e-1f) {
    LOG_PRINT("\n*** TEST PASSED ***\n");
  } else {
    LOG_PRINT("\n*** TEST FAILED ***\n");
  }

  aclDestroyTensor(features);
  aclDestroyTensor(labels);
  aclDestroyTensor(loss);
  aclDestroyTensor(backprop);
  aclrtFree(featuresDeviceAddr);
  aclrtFree(labelsDeviceAddr);
  aclrtFree(lossDeviceAddr);
  aclrtFree(backpropDeviceAddr);
  if (workspaceSize > 0) {
    aclrtFree(workspaceAddr);
  }
  aclrtDestroyStream(stream);
  aclrtResetDevice(deviceId);
  aclFinalize();
  return 0;
}