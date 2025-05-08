#include <sys/time.h>

#include <dirent.h>
#include <math.h>
#include <iostream>
#include <string>
#include <algorithm>
#include <iosfwd>
#include <vector>
#include <fstream>
#include <sstream>

#include "common_inc/infer.h"

DEFINE_string(mindir_path, "", "mindir path");
DEFINE_string(dataset_path, ".", "dataset path");
DEFINE_int32(device_id, 0, "device id");
DEFINE_string(device_type, "CPU", "device type");
DEFINE_int32(image_height, 640, "image height");
DEFINE_int32(image_width, 640, "image width");

int main(int argc, char **argv) {
  if (!ParseCommandLineFlags(argc, argv)) {
    std::cout << "Failed to parse args" << std::endl;
    return 1;
  }
  if (RealPath(FLAGS_mindir_path).empty()) {
    std::cout << "Invalid mindir" << std::endl;
    return 1;
  }

  auto ascend310 = std::make_shared<mindspore::Ascend310DeviceInfo>();
  ascend310->SetDeviceID(FLAGS_device_id);
  ascend310->SetPrecisionMode("preferred_fp32");
  ascend310->SetOpSelectImplMode("high_precision");
  ascend310->SetBufferOptimizeMode("off_optimize");
  mindspore::Model model;
  if (!LoadModel(FLAGS_mindir_path, FLAGS_device_type, FLAGS_device_id, ascend310, &model)) {
    std::cout << "Failed to load model " << FLAGS_mindir_path << ", device id: " << FLAGS_device_id
              << ", device type: " << FLAGS_device_type;
    return 1;
  }
  Status ret;

  auto all_files = GetAllFiles(FLAGS_dataset_path);
  std::map<double, double> costTime_map;
  size_t size = all_files.size();
  std::shared_ptr<TensorTransform> decode(new Decode());
  auto resize = Resize({FLAGS_image_height, FLAGS_image_width});
  Execute composeDecode({decode});

  for (size_t i = 0; i < size; ++i) {
    struct timeval start = {0};
    struct timeval end = {0};
    double startTimeMs;
    double endTimeMs;
    std::vector<MSTensor> inputs;
    std::vector<MSTensor> outputs;
    auto imgDecode = MSTensor();
    auto img = MSTensor();
    composeDecode(ReadFileToTensor(all_files[i]), &imgDecode);
    std::vector<int64_t> shape = imgDecode.Shape();

    if ((static_cast<int>(shape[0]) < static_cast<int>(FLAGS_image_height)) &&
        (static_cast<int>(shape[1]) < static_cast<int>(FLAGS_image_width))) {
      resize = Resize({FLAGS_image_height, FLAGS_image_width}, InterpolationMode::kCubic);
    } else if ((static_cast<int>(shape[0]) > static_cast<int>(FLAGS_image_height)) &&
               (static_cast<int>(shape[1]) > static_cast<int>(FLAGS_image_width))) {
      resize = Resize({FLAGS_image_height, FLAGS_image_width}, InterpolationMode::kNearestNeighbour);
    } else {
      resize = Resize({FLAGS_image_height, FLAGS_image_width}, InterpolationMode::kLinear);
    }
    if ((sizeof(shape) / sizeof(shape[0])) <= 2) {
      std::cout << "image channels is not 3." << std::endl;
      return 1;
    }
    Execute transform(resize);
    transform(imgDecode, &img);

    std::vector<MSTensor> model_inputs = model.GetInputs();
    inputs.emplace_back(model_inputs[0].Name(), model_inputs[0].DataType(), model_inputs[0].Shape(),
                        img.Data().get(), img.DataSize());
    gettimeofday(&start, nullptr);
    ret = model.Predict(inputs, &outputs);
    gettimeofday(&end, nullptr);
    if (ret != kSuccess) {
      std::cout << "Predict " << all_files[i] << " failed." << std::endl;
      return 1;
    }
    startTimeMs = (1.0 * start.tv_sec * 1000000 + start.tv_usec) / 1000;
    endTimeMs = (1.0 * end.tv_sec * 1000000 + end.tv_usec) / 1000;
    costTime_map.insert(std::pair<double, double>(startTimeMs, endTimeMs));
    WriteResult(all_files[i], outputs);
  }
  double average = 0.0;
  int inferCount = 0;

  for (auto iter = costTime_map.begin(); iter != costTime_map.end(); iter++) {
    double diff = iter->second - iter->first;
    average += diff;
    inferCount++;
  }
  average = average / inferCount;
  std::stringstream timeCost;
  timeCost << "NN inference cost average time: " << average << " ms of infer_count " << inferCount << std::endl;
  std::cout << "NN inference cost average time: " << average << "ms of infer_count " << inferCount << std::endl;
  std::string fileName = "./time_Result" + std::string("/test_perform_static.txt");
  std::ofstream fileStream(fileName.c_str(), std::ios::trunc);
  fileStream << timeCost.str();
  fileStream.close();
  costTime_map.clear();
  return 0;
}
