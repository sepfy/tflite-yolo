#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include <stdio.h>
#include <stdlib.h>
#include <gst/gst.h>
#include <gst/video/video-format.h>
#include <gst/base/gstbasetransform.h>
#include <opencv2/opencv.hpp>

#include "gstobjectdetect.h"

#include "object_detector.h"


GST_DEBUG_CATEGORY_STATIC (gst_objectdetect_debug_category);
#define GST_CAT_DEFAULT gst_objectdetect_debug_category

/* prototypes */

static void gst_objectdetect_set_property (GObject * object,
    guint property_id, const GValue * value, GParamSpec * pspec);
static void gst_objectdetect_get_property (GObject * object,
    guint property_id, GValue * value, GParamSpec * pspec);
static gboolean gst_objectdetect_set_caps (GstBaseTransform * trans,
    GstCaps * incaps, GstCaps * outcaps);
static GstFlowReturn gst_objectdetect_transform_ip (GstBaseTransform * trans,
    GstBuffer * buf);

#define DEFAULT_PROP_MODEL "yolo-fastestv2.tflite"
#define DEFAULT_PROP_NTHREADS 2
#define DEFAULT_PROP_THRESHOLD 0.4f 
#define DEFAULT_PROP_LABEL TRUE
enum
{
  PROP_0,
  PROP_MODEL,
  PROP_NTHREADS,
  PROP_THRESHOLD,
  PROP_LABEL,
};

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

/* pad templates */

#define FORMATS "{I420}"

static GstStaticPadTemplate gst_objectdetect_src_template =
GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE (FORMATS))
    );

static GstStaticPadTemplate gst_objectdetect_sink_template =
GST_STATIC_PAD_TEMPLATE ("sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS (GST_VIDEO_CAPS_MAKE (FORMATS))
    );


/* class initialization */

G_DEFINE_TYPE_WITH_CODE (GstObjectdetect, gst_objectdetect, GST_TYPE_BASE_TRANSFORM,
  GST_DEBUG_CATEGORY_INIT (gst_objectdetect_debug_category, "objectdetect", 0,
  "debug category for objectdetect element"));

static void
gst_objectdetect_class_init (GstObjectdetectClass * klass)
{
  GObjectClass *gobject_class = G_OBJECT_CLASS (klass);
  GstBaseTransformClass *base_transform_class = GST_BASE_TRANSFORM_CLASS (klass);

  gst_element_class_add_static_pad_template (GST_ELEMENT_CLASS(klass),
      &gst_objectdetect_src_template);
  gst_element_class_add_static_pad_template (GST_ELEMENT_CLASS(klass),
      &gst_objectdetect_sink_template);

  gst_element_class_set_static_metadata (GST_ELEMENT_CLASS(klass),
      "Object detection", "object detection", "Tensorflow-lite object detection by yolo fastest",
      "sepfy <sepfy95@gmail.com>");

  gobject_class->set_property = gst_objectdetect_set_property;
  gobject_class->get_property = gst_objectdetect_get_property;

  base_transform_class->set_caps = GST_DEBUG_FUNCPTR (gst_objectdetect_set_caps);
  base_transform_class->transform_ip = GST_DEBUG_FUNCPTR (gst_objectdetect_transform_ip);

  g_object_class_install_property (gobject_class, PROP_MODEL,
      g_param_spec_string ("model", "Model", "The path of model file", DEFAULT_PROP_MODEL,
      (GParamFlags) (G_PARAM_CONSTRUCT | G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_NTHREADS,
      g_param_spec_uint ("nthreads", "Threads", "The number of threads to use",
      0, G_MAXUINT, DEFAULT_PROP_NTHREADS,
      (GParamFlags) (G_PARAM_CONSTRUCT | G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_THRESHOLD,
      g_param_spec_float ("threshold", "Threshold", "The threshold of detection score",
      0.0, 1.0, DEFAULT_PROP_THRESHOLD,
      (GParamFlags) (G_PARAM_CONSTRUCT | G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

  g_object_class_install_property (gobject_class, PROP_LABEL,
      g_param_spec_boolean ("label", "Label", "Add label to image", DEFAULT_PROP_LABEL,
      (GParamFlags) (G_PARAM_CONSTRUCT | G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

}

static void
gst_objectdetect_init (GstObjectdetect *objectdetect)
{

  objectdetect->object_detector_ = std::unique_ptr<ObjectDetector>(new ObjectDetector());
}

void
gst_objectdetect_set_property (GObject * object, guint property_id,
    const GValue * value, GParamSpec * pspec)
{
  GstObjectdetect *objectdetect = GST_OBJECTDETECT (object);

  GST_DEBUG_OBJECT (objectdetect, "set_property");

  switch (property_id) {

    case PROP_MODEL:
      objectdetect->model_ = strdup(g_value_get_string (value));
      objectdetect->object_detector_->LoadModel(objectdetect->model_);
      break;
    case PROP_NTHREADS:
      objectdetect->nthread_ = g_value_get_uint (value);
      objectdetect->object_detector_->SetNumThreads(objectdetect->nthread_);
      break;
    case PROP_THRESHOLD:
      objectdetect->threshold_ = g_value_get_float (value);
      objectdetect->object_detector_->threshold(objectdetect->threshold_);
      break;
    case PROP_LABEL:
      objectdetect->label_ = g_value_get_boolean (value);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
      break;
  }

}

void
gst_objectdetect_get_property (GObject * object, guint property_id,
    GValue * value, GParamSpec * pspec)
{
  GstObjectdetect *objectdetect = GST_OBJECTDETECT (object);

  GST_DEBUG_OBJECT (objectdetect, "get_property");

  switch (property_id) {
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, property_id, pspec);
      break;
  }
}

static gboolean
gst_objectdetect_set_caps (GstBaseTransform * trans, GstCaps * incaps,
    GstCaps * outcaps)
{
  GstObjectdetect *objectdetect = GST_OBJECTDETECT (trans);

  GST_DEBUG_OBJECT (objectdetect, "set_caps");

  GstStructure *structure;

  structure = gst_caps_get_structure (incaps, 0);

  gst_structure_get_int(structure, "width", &objectdetect->width_);
  gst_structure_get_int(structure, "height", &objectdetect->height_);

  return TRUE;
}

static void gst_objectdetect_preprocess(uint32_t w, uint32_t h, uint8_t *buf, float *dest)
{

  cv::Mat image(h*3/2, w, CV_8UC1);

  memcpy(image.data, buf, w*h*3/2);

  cv::cvtColor(image, image, cv::COLOR_YUV2RGB_I420);

  cv::resize(image, image, cv::Size(kWidth, kHeight), cv::INTER_NEAREST);

  image.convertTo(image, CV_32FC3, 1.0/255.0);

  std::vector<cv::Mat> input_channels(3);

  cv::split(image, input_channels);

  int channel_length = kWidth*kHeight;

  float *data = dest;
  // HWC -> CWH
  for(int i = 0; i < 3; ++i)  {
    memcpy(data, input_channels[i].data, channel_length * sizeof(float));
    data += channel_length;
  }
  
}

static void gst_objectdetect_do_label(std::vector<ObjectBox> detection_boxes, cv::Mat image) {

  char text[256];
  int base_line = 0;

  for(size_t i = 0; i < detection_boxes.size(); i++) {

    float lx = (detection_boxes[i].cx - detection_boxes[i].w*0.5)*(float)image.size().width/kWidth;
    float ly = (detection_boxes[i].cy - detection_boxes[i].h*0.5)*(float)image.size().height/kHeight;
    float rx = (detection_boxes[i].cx + detection_boxes[i].w*0.5)*(float)image.size().width/kWidth;
    float ry = (detection_boxes[i].cy + detection_boxes[i].h*0.5)*(float)image.size().height/kHeight;

    cv::Point pt1(lx, ly);
    cv::Point pt2(rx, ry);

    cv::rectangle(image, pt1, pt2, cv::Scalar(255, 255, 255), 2);

    sprintf(text, "%s %.1f%%", kClassNames[detection_boxes[i].category], detection_boxes[i].score*100);

    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &base_line);

    cv::rectangle(image, cv::Rect(pt1, cv::Size(label_size.width, label_size.height + base_line)),
     cv::Scalar(255, 255, 255), -1);

    cv::putText(image, text, cv::Point(lx, ly + label_size.height),
     cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

  }

}

static GstFlowReturn
gst_objectdetect_transform_ip (GstBaseTransform * trans, GstBuffer * buf)
{
  GstObjectdetect *objectdetect = GST_OBJECTDETECT (trans);

  GST_DEBUG_OBJECT (objectdetect, "transform_ip");

  GstMapInfo info;
  gst_buffer_map(buf, &info, GST_MAP_WRITE);

  int h = objectdetect->height_;
  int w = objectdetect->width_;


  float *input_buffer = objectdetect->object_detector_->GetInputTensor();

  gst_objectdetect_preprocess(w, h, info.data, input_buffer);

  std::vector<ObjectBox> detection_boxes = objectdetect->object_detector_->Inference();

  if(objectdetect->label_) {
    cv::Mat draw(h, w, CV_8UC1, info.data);
    gst_objectdetect_do_label(detection_boxes, draw);
  }

  gst_buffer_unmap(buf,&info);

  return GST_FLOW_OK;
}

static gboolean
plugin_init (GstPlugin * plugin)
{

  return gst_element_register (plugin, "objectdetect", GST_RANK_NONE,
      GST_TYPE_OBJECTDETECT);
}

#ifndef VERSION
#define VERSION "0.0.1"
#endif
#ifndef PACKAGE
#define PACKAGE "gstobjectdetect"
#endif
#ifndef PACKAGE_NAME
#define PACKAGE_NAME "gstobjectdetect"
#endif
#ifndef GST_PACKAGE_ORIGIN
#define GST_PACKAGE_ORIGIN "https://github.com/sepfy"
#endif

GST_PLUGIN_DEFINE (GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    objectdetect,
    "FIXME plugin description",
    plugin_init, VERSION, "LGPL", PACKAGE_NAME, GST_PACKAGE_ORIGIN)

