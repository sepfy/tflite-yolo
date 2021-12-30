#ifndef OBJECT_DETECTOR_H_
#define OBJECT_DETECTOR_H_

#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"

#include <vector>

class ObjectBox {

 public:

  ObjectBox(int cx, int cy, int w, int h, int category, float score) : cx_(cx), cy_(cy), w_(w), h_(h), category_(category), score_(score) {}

  //float GetWidth() { return rx_ - lx_; }
  //float GetHeight() { return ry_ - ly_; }
  //float GetArea() { return GetWidth()*GetHeight(); }

 private:
  int w_;
  int h_;
  int cx_;
  int cy_;
  int category_;
  float score_;

};


class ObjectDetector {

 public:
  ObjectDetector(int width, int height);
  ~ObjectDetector();
  int LoadModel(const char *model_path);
  int Inference(float *inputs);
  std::vector<ObjectBox> Postprocess(float *detections, int stride);


  int BoxConfidenceArgmax(std::vector<ObjectBox> boxes);
  std::vector<ObjectBox> NonMaxSupression(std::vector<ObjectBox> candidate_boxes);

 private:
  int width_;
  int height_;
  int category_;
  std::unique_ptr<tflite::Interpreter> interpreter_;
  std::unique_ptr<tflite::FlatBufferModel> model_;

  int num_of_category_;
  std::vector<float> anchors_;
  int num_of_anchor_;

  std::vector<int> strides_;
};

#endif // OBJECT_DETECTOR_H_
