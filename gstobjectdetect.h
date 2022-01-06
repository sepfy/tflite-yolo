/* GStreamer
 * Copyright (C) 2022 FIXME <fixme@example.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 51 Franklin St, Fifth Floor,
 * Boston, MA 02110-1301, USA.
 */

#ifndef _GST_OBJECTDETECT_H_
#define _GST_OBJECTDETECT_H_

#include <gst/base/gstbasetransform.h>

#include <memory>
#include "object_detector.h"

G_BEGIN_DECLS

#define GST_TYPE_OBJECTDETECT   (gst_objectdetect_get_type())
#define GST_OBJECTDETECT(obj)   (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_OBJECTDETECT,GstObjectdetect))
#define GST_OBJECTDETECT_CLASS(klass)   (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_OBJECTDETECT,GstObjectdetectClass))
#define GST_IS_OBJECTDETECT(obj)   (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_OBJECTDETECT))
#define GST_IS_OBJECTDETECT_CLASS(obj)   (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_OBJECTDETECT))

typedef struct _GstObjectdetect GstObjectdetect;
typedef struct _GstObjectdetectClass GstObjectdetectClass;

struct _GstObjectdetect
{
  GstBaseTransform base_objectdetect;

  int width_;

  int height_;

  char *model_;

  guint nthread_;

  float threshold_;

  gboolean label_;

  std::unique_ptr<ObjectDetector> object_detector_;

};

struct _GstObjectdetectClass
{
  GstBaseTransformClass base_objectdetect_class;
};

GType gst_objectdetect_get_type (void);

G_END_DECLS

#endif
