#include <cstdio>
#include <opencv2/opencv.hpp>

#include "object_detector.h"

using namespace cv;

const int kWidth = 352;
const int kHeight = 352;

const char *kClassNames[] = {
 "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
 "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
 "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
 "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
 "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
 "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
 "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
 "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
 "scissors", "teddy bear", "hair drier", "toothbrush"
};


int main(int argc, char* argv[]) {

  Mat image = imread(argv[2]);
  Mat src;
  image.copyTo(src);
  resize(image, image, Size(352, 352), 0, 0, INTER_NEAREST);
  cvtColor(image, image, COLOR_BGR2RGB);
  image.convertTo(image, CV_32FC3, 1.0/255.0);

  std::vector<cv::Mat> input_channels(3);
  cv::split(image, input_channels);

  std::vector<float> result(352*352*3);
  auto data = result.data();
  int channelLength = 352*352;
  for (int i = 0; i < 3; ++i) 
  {
      memcpy(data, input_channels[i].data, channelLength * sizeof(float));
      data += channelLength;
  }

  ObjectDetector object_detector(352, 352);
  object_detector.LoadModel(argv[1]);

  std::vector<ObjectBox> detection_boxes = object_detector.Inference(result.data());
  for(size_t i = 0; i < detection_boxes.size(); i++) {

    float lx = (detection_boxes[i].cx - detection_boxes[i].w*0.5)*(float)src.size().width/352.0;
    float ly = (detection_boxes[i].cy - detection_boxes[i].h*0.5)*(float)src.size().height/352.0;
    float rx = (detection_boxes[i].cx + detection_boxes[i].w*0.5)*(float)src.size().width/352.0;
    float ry = (detection_boxes[i].cy + detection_boxes[i].h*0.5)*(float)src.size().height/352.0;
    cv::Point pt1(lx, ly);
    cv::Point pt2(rx, ry);
    cv::rectangle(src, pt1, pt2, cv::Scalar(0, 255, 0));

    char text[256];
    sprintf(text, "%s %.1f%%", kClassNames[detection_boxes[i].category], detection_boxes[i].score * 100);

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    cv::rectangle(src, cv::Rect(pt1, cv::Size(label_size.width, label_size.height + baseLine)), cv::Scalar(0, 255, 0), -1);

    cv::putText(src, text, cv::Point(lx, ly + label_size.height),
     cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

  }

  imwrite("result.jpg", src);

  return 0;
}
