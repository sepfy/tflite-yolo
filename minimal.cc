#include <cstdio>
#include <opencv2/opencv.hpp>

#include "object_detector.h"

using namespace cv;

int main(int argc, char* argv[]) {

  Mat image = imread(argv[2]);
  Mat src;
  image.copyTo(src);
  resize(image, image, Size(352, 352), 0, 0, INTER_NEAREST);
  cvtColor(image, image, COLOR_BGR2RGB);
  image.convertTo(image, CV_32FC3);
  image = (image - 127.5)/127.5;

  ObjectDetector object_detector(352, 352);
  object_detector.LoadModel(argv[1]);
  object_detector.Inference((float*)image.data);
#if 0
  std::unique_ptr<tflite::Interpreter> interpreter;
  builder(&interpreter);
  TFLITE_MINIMAL_CHECK(interpreter != nullptr);

  // Allocate tensor buffers.
  TFLITE_MINIMAL_CHECK(interpreter->AllocateTensors() == kTfLiteOk);
  printf("=== Pre-invoke Interpreter State ===\n");
  tflite::PrintInterpreterState(interpreter.get());

  // Fill input buffers
  // TODO(user): Insert code to fill input tensors.
  // Note: The buffer of the input tensor with index `i` of type T can
  // be accessed with `T* input = interpreter->typed_input_tensor<T>(i);`

  // Run inference
  TFLITE_MINIMAL_CHECK(interpreter->Invoke() == kTfLiteOk);
  printf("\n\n=== Post-invoke Interpreter State ===\n");
  tflite::PrintInterpreterState(interpreter.get());

  // Read output buffers
  // TODO(user): Insert getting data out code.
  // Note: The buffer of the output tensor with index `i` of type T can
  // be accessed with `T* output = interpreter->typed_output_tensor<T>(i);`
#endif
  return 0;
}
