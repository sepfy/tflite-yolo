#ifndef OBJECT_DETECTOR_H_
#define OBJECT_DETECTOR_H_

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#include <vector>

typedef struct ObjectBox {

  //float GetWidth() { return rx_ - lx_; }
  //float GetHeight() { return ry_ - ly_; }
  //float GetArea() { return GetWidth()*GetHeight(); }

  int w;
  int h;
  int cx;
  int cy;
  int category;
  float score;

} ObjectBox;


class ObjectDetector {

 public:

  ObjectDetector();
  ~ObjectDetector();
  int LoadModel(const char *model_path);
  std::vector<ObjectBox> Inference(float *inputs);
  std::vector<ObjectBox> Inference();
  float *GetInputTensor();
  std::vector<ObjectBox> Postprocess(float *detections, int stride, std::vector<float> anchors);


  int BoxConfidenceArgmax(std::vector<ObjectBox> boxes);
  std::vector<ObjectBox> NonMaxSupression(std::vector<ObjectBox> candidate_boxes);

  void SetNumThreads(uint32_t num);

  inline void threshold(float threshold) { threshold_ = threshold; }

 private:

  int width_;
  int height_;
  int category_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
  std::unique_ptr<tflite::FlatBufferModel> model_;

  int num_of_category_;
  std::vector<float> anchors_;
  int num_of_anchor_;
  float iou_threshold_;
  std::vector<int> strides_;

  float threshold_;
};

#endif // OBJECT_DETECTOR_H_
