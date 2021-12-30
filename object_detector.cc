#include <sys/time.h>
#include "object_detector.h"

long long getms() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec*1.0e+3 + tv.tv_usec/1000;
}

ObjectDetector::ObjectDetector(int width, int height) {

  width_ = width;
  height_ = height;

  num_of_category_ = 80;

  anchors_ = std::vector<float>({12.64, 19.39, 37.88, 51.48, 55.71, 138.31, 126.91, 78.23, 131.57, 214.55, 279.92, 258.87});
  num_of_anchor_ = anchors_.size()/2;

  strides_ = std::vector<int>({16, 32});

}

ObjectDetector::~ObjectDetector() {
}

int ObjectDetector::LoadModel(const char *model_path) {

  model_ = tflite::FlatBufferModel::BuildFromFile(model_path);

  if(model_ == nullptr) {
    printf("Load model failed\n");
    return -1;
  }

  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder builder(*model_, resolver);
  builder(&interpreter_);

  if(interpreter_ == nullptr) {
    printf("Build interpreter failed\n");
    return -1;
  }

#if 0
  std::vector<int> strides;
  strides.push_back(8);
  strides.push_back(16);
  strides.push_back(16);
  strides.push_back(16);

  anchor_boxes_ = GenerateAnchors(width_, height_, strides);
#endif

  if(interpreter_->AllocateTensors() != kTfLiteOk) {
    printf("Allocate Tensors failed\n");
    return -1;
  }

  tflite::PrintInterpreterState(interpreter_.get());
  return 0;
}


int ObjectDetector::Inference(float *inputs) {

  TfLiteTensor *input_tensor = interpreter_->tensor(interpreter_->inputs()[0]);
  memcpy(interpreter_->typed_input_tensor<float>(0), inputs, width_*height_*3*sizeof(float));
printf("Involke!\n");

  long long start_time, end_time;

  start_time = getms();
  if(!interpreter_->Invoke() == kTfLiteOk)
    return -1;

  std::vector<ObjectBox> candidate_boxes1;
  candidate_boxes1 = Postprocess(interpreter_->typed_output_tensor<float>(0), strides_[0]);
  std::vector<ObjectBox> candidate_boxes2;
  candidate_boxes2 = Postprocess(interpreter_->typed_output_tensor<float>(1), strides_[1]);

  end_time = getms();
  printf("Involke End! %lld\n", end_time - start_time);

  return 0;
}

std::vector<ObjectBox> ObjectDetector::Postprocess(float *detections, int stride) {

  std::vector<ObjectBox> object_boxes;

  int cx, cy, w, h;

  float score;
  float confidence;
  float max_confidence = 0;
  int category = 0;

  int feature_map_width = width_/stride;
  int feature_map_height = height_/stride;
  int feature_map_start_index = 0;

  for(int w = 0; w < feature_map_width; w++) {


    for(int h = 0; h < feature_map_height; h++) {


      for(int c = 0; c < num_of_category_; c++) {

        confidence = detections[feature_map_start_index + 4*num_of_anchor_ + num_of_anchor_ + c];

        if(confidence > max_confidence) {
          max_confidence = confidence;
          category = c;
        }

      }

  
      for(size_t i = 0; i < num_of_anchor_; i++) {

        score = max_confidence*detections[feature_map_start_index + 4*num_of_anchor_ + i];
        cx = (detections[feature_map_start_index + 4*i + 0]*2.0 - 0.5 + w)*stride;
        cy = (detections[feature_map_start_index + 4*i + 1]*2.0 - 0.5 + h)*stride;
        w = pow(detections[feature_map_start_index + 4*i + 1]*2.0, 2)*anchors_[i*2 + 0];
        h = pow(detections[feature_map_start_index + 4*i + 1]*2.0, 2)*anchors_[i*2 + 1];
                        
        ObjectBox object_box(cx, cy, w, h, category, score);
        object_boxes.push_back(object_box);
      }

      feature_map_start_index += 95;

    }

  }

  return object_boxes;
}


int ObjectDetector::BoxConfidenceArgmax(std::vector<ObjectBox> boxes) {

  float max_confidence = 0;
  int max_index = 0;
  for(size_t i = 0; i < boxes.size(); i++) {
    if(boxes[i].confidence > max_confidence) {
      max_confidence = boxes[i].confidence;
      max_index = i;
    }
  }
  return max_index;
}

std::vector<ObjectBox> ObjectDetector::NonMaxSupression(std::vector<ObjectBox> candidate_boxes) {

  std::vector<Box> detection_boxes;
  while(candidate_boxes.size() > 0) {
    size_t max_index = BoxConfidenceArgmax(candidate_boxes);
    Box selected_box = candidate_boxes[max_index];
    candidate_boxes.erase(candidate_boxes.begin() + max_index);
    detection_boxes.push_back(selected_box);

    float selected_box_left = selected_box.cx - 0.5*selected_box.w;
    float selected_box_top = selected_box.cy - 0.5*selected_box.h;
    float selected_box_right = selected_box.cx + 0.5*selected_box.w;
    float selected_box_bottom = selected_box.cy + 0.5*selected_box.h;

    auto box = candidate_boxes.begin();

    while(box != candidate_boxes.end()) {

      float box_left = box->cx - 0.5*box->w;
      float box_top = box->cy - 0.5*box->h;
      float box_right = box->cx + 0.5*box->w;
      float box_bottom = box->cy + 0.5*box->h;

      float union_box_w = std::min(selected_box_right, box_right)
       - std::max(selected_box_left, box_left);
      float union_box_h = std::min(selected_box_bottom, box_bottom)
       - std::max(selected_box_top, box_top);
      float iou = (union_box_w*union_box_h)/(selected_box.w*selected_box.h + box->w*box->h
       - union_box_w*union_box_h);

      if(iou > iou_threshold_ && union_box_w > 0 && union_box_h > 0)
        box = candidate_boxes.erase(box);
      else
        ++box;
    }
  }

  return detection_boxes;
}
