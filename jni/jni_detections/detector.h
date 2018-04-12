/*
 * detector.h using google-style
 *
 *  Created on: May 24, 2016
 *      Author: Tzutalin
 *
 *  Copyright (c) 2016 Tzutalin. All rights reserved.
 */

#pragma once

#include <jni_common/jni_fileutils.h>
#include <dlib/image_loader/load_image.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/opencv/cv_image.h>
#include <dlib/image_loader/load_image.h>
#include <glog/logging.h>
#include <jni.h>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <string>
#include <vector>
#include <unordered_map>
#define SKIP_FRAMES 2

class OpencvHOGDetctor {
 public:
  OpencvHOGDetctor() {}

  inline int det(const cv::Mat& src_img) {
    if (src_img.empty())
      return 0;

    cv::HOGDescriptor hog;
    hog.setSVMDetector(cv::HOGDescriptor::getDefaultPeopleDetector());
    std::vector<cv::Rect> found, found_filtered;
    hog.detectMultiScale(src_img, found, 0, cv::Size(8, 8), cv::Size(32, 32),
                         1.05, 2);
    size_t i, j;
    for (i = 0; i < found.size(); i++) {
      cv::Rect r = found[i];
      for (j = 0; j < found.size(); j++)
        if (j != i && (r & found[j]) == r)
          break;
      if (j == found.size())
        found_filtered.push_back(r);
    }

    for (i = 0; i < found_filtered.size(); i++) {
      cv::Rect r = found_filtered[i];
      r.x += cvRound(r.width * 0.1);
      r.width = cvRound(r.width * 0.8);
      r.y += cvRound(r.height * 0.06);
      r.height = cvRound(r.height * 0.9);
      cv::rectangle(src_img, r.tl(), r.br(), cv::Scalar(0, 255, 0), 2);
    }
    mResultMat = src_img;
    // cv::imwrite(path, mResultMat);
    LOG(INFO) << "det ends";
    mRets = found_filtered;
    return found_filtered.size();
  }

  inline cv::Mat& getResultMat() { return mResultMat; }

  inline std::vector<cv::Rect>& getResult() { return mRets; }

 private:
  cv::Mat mResultMat;
  std::vector<cv::Rect> mRets;
};

class DLibHOGDetector {
 private:
  typedef dlib::scan_fhog_pyramid<dlib::pyramid_down<6>> image_scanner_type;
  dlib::object_detector<image_scanner_type> mObjectDetector;

  inline void init() {
    LOG(INFO) << "Model Path: " << mModelPath;
    if (jniutils::fileExists(mModelPath)) {
      dlib::deserialize(mModelPath) >> mObjectDetector;
    } else {
      LOG(INFO) << "Not exist " << mModelPath;
    }
  }

 public:
  DLibHOGDetector(const std::string& modelPath = "/sdcard/person.svm")
      : mModelPath(modelPath) {
    init();
  }

  virtual inline int det(const std::string& path) {
    using namespace jniutils;
    if (!fileExists(mModelPath) || !fileExists(path)) {
      LOG(WARNING) << "No modle path or input file path";
      return 0;
    }
    cv::Mat src_img = cv::imread(path, CV_LOAD_IMAGE_COLOR);
    if (src_img.empty())
      return 0;
    int img_width = src_img.cols;
    int img_height = src_img.rows;
    int im_size_min = MIN(img_width, img_height);
    int im_size_max = MAX(img_width, img_height);

    float scale = float(INPUT_IMG_MIN_SIZE) / float(im_size_min);
    if (scale * im_size_max > INPUT_IMG_MAX_SIZE) {
      scale = (float)INPUT_IMG_MAX_SIZE / (float)im_size_max;
    }

    if (scale != 1.0) {
      cv::Mat outputMat;
      cv::resize(src_img, outputMat,
                 cv::Size(img_width * scale, img_height * scale));
      src_img = outputMat;
    }

    // cv::resize(src_img, src_img, cv::Size(320, 240));
    dlib::cv_image<dlib::bgr_pixel> cimg(src_img);

    double thresh = 0.5;
    mRets = mObjectDetector(cimg, thresh);
    return mRets.size();
  }

  inline std::vector<dlib::rectangle> getResult() { return mRets; }

  virtual ~DLibHOGDetector() {}

 protected:
  std::vector<dlib::rectangle> mRets;
  std::string mModelPath;
  const int INPUT_IMG_MAX_SIZE = 800;
  const int INPUT_IMG_MIN_SIZE = 600;
};

/*
 * DLib face detect and face feature extractor
 */
class DLibHOGFaceDetector : public DLibHOGDetector {
 private:
  std::string mLandMarkModel;
  dlib::shape_predictor shapePredictor;
  std::unordered_map<int, dlib::full_object_detection> mFaceShapeMap;
  dlib::frontal_face_detector mFaceDetector;
  cv::Rect lastFace;
  cv::Rect returnRect;

  inline void init() {
    LOG(INFO) << "Init mFaceDetector";
    mFaceDetector = dlib::get_frontal_face_detector();
  }

 public:
  DLibHOGFaceDetector() { init(); }

  DLibHOGFaceDetector(const std::string& landmarkmodel)
      : mLandMarkModel(landmarkmodel) {
    init();
    if (!mLandMarkModel.empty() && jniutils::fileExists(mLandMarkModel)) {
      dlib::deserialize(mLandMarkModel) >> shapePredictor;
      LOG(INFO) << "Load landmarkmodel from " << mLandMarkModel;
    }
  }

  virtual inline int det(const std::string& path) {
    LOG(INFO) << "Read path from " << path;
    cv::Mat src_img = cv::imread(path, CV_LOAD_IMAGE_COLOR);
    return det(src_img);
  }

  // The format of mat should be BGR or Gray
  // If converting 4 channels to 3 channls because the format could be BGRA or
  // ARGB
  virtual inline int detRaw(const cv::Mat& image) {
    cv::Mat cropped;
    int decreasedX = 0;
    int decreasedY = 0;
    int increasedWidth = 0;
    int increasedHeight = 0;
    int padding = 20;


    if (image.empty()) return 0;

    if (image.channels() > 1) {
        cv::cvtColor(image, image, CV_BGR2GRAY);
    }


    if (lastFace.area() > 0 && 0 <= lastFace.x
        && 0 <= lastFace.width
        && lastFace.x + lastFace.width <= image.cols
        && 0 <= lastFace.y
        && 0 <= lastFace.height
        && lastFace.y + lastFace.height <= image.rows) {
        LOG(INFO) << "DETECTING FROM LAST FACE";

        returnRect = cv::Rect(lastFace.x - padding, lastFace.y - padding, lastFace.width + (padding * 2), lastFace.height + (padding * 2));
        if (returnRect.x < 0) {
            decreasedX = returnRect.x + padding;
            returnRect.x = 0;
        }
        if (returnRect.y < 0) {
            decreasedY = returnRect.y + padding;
            returnRect.y = 0;
        }
        if (returnRect.x + returnRect.width >= image.cols) {
            returnRect.width = image.cols - returnRect.x;
            increasedWidth = returnRect.width - lastFace.width;
        }
        if (returnRect.y+returnRect.height >= image.rows) {
            returnRect.height = image.rows-returnRect.y;
            increasedHeight = returnRect.height - lastFace.height;
        }

        cropped = image(returnRect);
    } else {
        LOG(WARNING) << "LAST FACE IS EMPTY";
        returnRect = cv::Rect(0, 0, 0, 0);
        cropped = image;
    }
    dlib::cv_image<unsigned char> croppedImg(cropped);
    dlib::cv_image<unsigned char> img(image);
    // Resize
    // cv::Mat image_resize;
    // unsigned min_face_size = 200; //px
    // double k = 80.0 / min_face_size;
    // cv::resize(image_gray, image_resize, cv::Size(), k, k);



    // rect from full image left71 right167 top114 bottom210



    // HEIGHT DIFF: 223 WIDTHDIFF: 143

    // rect from cropped left-14 right92 top7 bottom103



    mRets = mFaceDetector(croppedImg);
    mFaceShapeMap.clear();
     // Process shape
     if (mRets.size() != 0 && mLandMarkModel.empty() == false) {
        LOG(INFO) << "DETECTED FACE";
       for (unsigned long j = 0; j < mRets.size(); ++j) {
         LOG(INFO) << "ORIGINAL FACE RECT" << "left" << mRets[j].left() << "right" << mRets[j].right() << "top" << mRets[j].top() << "bottom" << mRets[j].bottom();

         dlib::rectangle face(mRets[j].left() + returnRect.tl().x, mRets[j].top() + returnRect.tl().y, mRets[j].right() + returnRect.tl().x, mRets[j].bottom() + returnRect.tl().y);

         LOG(INFO) << "RESIZED FACE RECT" << "left" << face.left() << "right" << face.right() << "top" << face.top() << "bottom" << face.bottom();

         dlib::full_object_detection shape = shapePredictor(img, face);

         mFaceShapeMap[j] = shape;

         lastFace = cv::Rect(cv::Point2i(face.left(), face.top()), cv::Point2i(face.right(), face.bottom()));
       }
     } else {
        LOG(WARNING) << "DIDNT DETECT FACE";
        lastFace = cv::Rect(0, 0, 0, 0);
        returnRect = cv::Rect(0, 0, 0, 0);
     }

    return mRets.size();
  }



   virtual inline int det(const cv::Mat& image) {
       if (image.empty())
             return 0;
           LOG(INFO) << "com_tzutalin_dlib_PeopleDet go to det(mat)";
           if (image.channels() == 1) {
             cv::cvtColor(image, image, CV_GRAY2BGR);
           }
           CHECK(image.channels() == 3);
           // TODO : Convert to gray image to speed up detection
           // It's unnecessary to use color image for face/landmark detection
           dlib::cv_image<dlib::bgr_pixel> img(image);
           mRets = mFaceDetector(img);
           LOG(INFO) << "Dlib HOG face det size : " << mRets.size();
           mFaceShapeMap.clear();
           // Process shape
           if (mRets.size() != 0 && mLandMarkModel.empty() == false) {
             for (unsigned long j = 0; j < mRets.size(); ++j) {
               dlib::full_object_detection shape = shapePredictor(img, mRets[j]);
               LOG(INFO) << "face index:" << j
                         << "number of parts: " << shape.num_parts();
               mFaceShapeMap[j] = shape;
             }
           }
           return mRets.size();
   }

  std::unordered_map<int, dlib::full_object_detection>& getFaceShapeMap() {
    return mFaceShapeMap;
  }
};
