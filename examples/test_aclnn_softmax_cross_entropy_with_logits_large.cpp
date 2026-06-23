/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 */
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
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

  // 大规模测试：batch=200, numClasses=10
  int64_t batch = 200;
  int64_t numClasses = 10;
  std::vector<int64_t> featuresShape = {batch, numClasses};
  std::vector<int64_t> labelsShape = {batch, numClasses};
  std::vector<int64_t> lossShape = {batch};
  std::vector<int64_t> backpropShape = {batch, numClasses};

  // 用循环生成数据：features用简单规律数值，labels用one-hot（每行随机选一个类别为1）
  std::vector<float> featuresHostData(batch * numClasses);
  std::vector<float> labelsHostData(batch * numClasses, 0.0f);
  std::vector<int64_t> labelIndex(batch);

  srand(42); // 固定随机种子，保证每次运行结果可复现
  for (int64_t i = 0; i < batch; i++) {
    for (int64_t j = 0; j < numClasses; j++) {
      // 用一个简单规律生成数值，范围在[-2, 8]之间，带一点行间变化
      featuresHostData[i * numClasses + j] = static_cast<float>((j - numClasses / 2) + (i % 5) * 0.3f);
    }
    int64_t labelClass = rand() % numClasses;
    labelIndex[i] = labelClass;
    labelsHostData[i * numClasses + labelClass] = 1.0f;
  }

  std::vector<float> lossHostData(batch, 0.0f);
  std::vector<float> backpropHostData(batch * numClasses, 0.0f);

  void* featuresDeviceAddr = nullptr;
  void* labelsDeviceAddr = nullptr;
  void* lossDeviceAddr = nullptr;
  void* backpropDeviceAddr = nullptr;
  aclTensor* features = nullptr;
  aclTensor* labels = nullptr;
  aclTensor* loss = nullptr;
  aclTensor* backprop = nullptr;

  ret = CreateAclTensor(featuresHostData, featuresShape, &featuresDeviceAddr, aclDataType::ACL_FLOAT, &features);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(labelsHostData, labelsShape, &labelsDeviceAddr, aclDataType::ACL_FLOAT, &labels);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(lossHostData, lossShape, &lossDeviceAddr, aclDataType::ACL_FLOAT, &loss);
  CHECK_RET(ret == ACL_SUCCESS, return ret);
  ret = CreateAclTensor(backpropHostData, backpropShape, &backpropDeviceAddr, aclDataType::ACL_FLOAT, &backprop);
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

  std::vector<float> lossResult(batch, 0.0f);
  ret = aclrtMemcpy(lossResult.data(), lossResult.size() * sizeof(float), lossDeviceAddr,
                    batch * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy loss failed. ERROR: %d\n", ret); return ret);

  std::vector<float> backpropResult(batch * numClasses, 0.0f);
  ret = aclrtMemcpy(backpropResult.data(), backpropResult.size() * sizeof(float), backpropDeviceAddr,
                    batch * numClasses * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
  CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("copy backprop failed. ERROR: %d\n", ret); return ret);

  // ===== 在CPU上用同样的算法重新算一遍golden值，自动对比 =====
  std::vector<float> goldenLoss(batch);
  std::vector<float> goldenBackprop(batch * numClasses);
  for (int64_t i = 0; i < batch; i++) {
    float maxVal = featuresHostData[i * numClasses];
    for (int64_t j = 1; j < numClasses; j++) {
      float v = featuresHostData[i * numClasses + j];
      if (v > maxVal) maxVal = v;
    }
    float sumExp = 0.0f;
    std::vector<float> expVals(numClasses);
    for (int64_t j = 0; j < numClasses; j++) {
      expVals[j] = std::exp(featuresHostData[i * numClasses + j] - maxVal);
      sumExp += expVals[j];
    }
    float logSum = std::log(sumExp);
    float lossVal = 0.0f;
    for (int64_t j = 0; j < numClasses; j++) {
      float softmaxVal = expVals[j] / sumExp;
      float labelVal = labelsHostData[i * numClasses + j];
      goldenBackprop[i * numClasses + j] = softmaxVal - labelVal;
      lossVal += -labelVal * (featuresHostData[i * numClasses + j] - maxVal - logSum);
    }
    goldenLoss[i] = lossVal;
  }

  // 对比误差
  float maxLossErr = 0.0f;
  float maxBackpropErr = 0.0f;
  for (int64_t i = 0; i < batch; i++) {
    float err = std::fabs(lossResult[i] - goldenLoss[i]);
    if (err > maxLossErr) maxLossErr = err;
  }
  for (int64_t i = 0; i < batch * numClasses; i++) {
    float err = std::fabs(backpropResult[i] - goldenBackprop[i]);
    if (err > maxBackpropErr) maxBackpropErr = err;
  }

  LOG_PRINT("batch=%ld, numClasses=%ld\n", batch, numClasses);
  LOG_PRINT("Max loss error: %e\n", maxLossErr);
  LOG_PRINT("Max backprop error: %e\n", maxBackpropErr);

  // 打印前5行和最后5行作为抽样检查
  LOG_PRINT("\n--- First 5 rows ---\n");
  for (int64_t i = 0; i < 5 && i < batch; i++) {
    LOG_PRINT("loss[%ld]: device=%f, golden=%f\n", i, lossResult[i], goldenLoss[i]);
  }
  LOG_PRINT("\n--- Last 5 rows ---\n");
  for (int64_t i = batch - 5; i < batch; i++) {
    if (i < 0) continue;
    LOG_PRINT("loss[%ld]: device=%f, golden=%f\n", i, lossResult[i], goldenLoss[i]);
  }

  if (maxLossErr < 1e-3f && maxBackpropErr < 1e-3f) {
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