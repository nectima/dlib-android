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
#define FRAME_PADDING 20

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
  std::vector<cv::Point3d> modelPoints;
  std::vector<cv::Point2d> imagePoints;
  std::vector<cv::Point2d> headPose;
  std::vector<cv::Point3d> reprojectsrc;
  std::vector<std::vector<cv::Point2f>> delaunay;
  cv::Mat poseMatrix;
  cv::Mat rotationMatrix;
  cv::Mat eulerAngle;
  cv::Mat out_intrinsics;
  cv::Mat out_rotation;
  cv::Mat out_translation;

  inline void init() {
    LOG(INFO) << "Init mFaceDetector";
    mFaceDetector = dlib::get_frontal_face_detector();

    modelPoints.push_back(cv::Point3d(6.825897, 6.760612, 4.402142));     //#33 left brow left corner
    modelPoints.push_back(cv::Point3d(1.330353, 7.122144, 6.903745));     //#29 left brow right corner
    modelPoints.push_back(cv::Point3d(-1.330353, 7.122144, 6.903745));    //#34 right brow left corner
    modelPoints.push_back(cv::Point3d(-6.825897, 6.760612, 4.402142));    //#38 right brow right corner
    modelPoints.push_back(cv::Point3d(5.311432, 5.485328, 3.987654));     //#13 left eye left corner
    modelPoints.push_back(cv::Point3d(1.789930, 5.393625, 4.413414));     //#17 left eye right corner
    modelPoints.push_back(cv::Point3d(-1.789930, 5.393625, 4.413414));    //#25 right eye left corner
    modelPoints.push_back(cv::Point3d(-5.311432, 5.485328, 3.987654));    //#21 right eye right corner
    modelPoints.push_back(cv::Point3d(2.005628, 1.409845, 6.165652));     //#55 nose left corner
    modelPoints.push_back(cv::Point3d(-2.005628, 1.409845, 6.165652));    //#49 nose right corner
    modelPoints.push_back(cv::Point3d(2.774015, -2.080775, 5.048531));    //#43 mouth left corner
    modelPoints.push_back(cv::Point3d(-2.774015, -2.080775, 5.048531));   //#39 mouth right corner
    modelPoints.push_back(cv::Point3d(0.000000, -3.116408, 6.097667));    //#45 mouth central bottom corner
    modelPoints.push_back(cv::Point3d(0.000000, -7.415691, 4.070434));    //#6 chin corner

    reprojectsrc.push_back(cv::Point3d(10.0, 10.0, 10.0));
    reprojectsrc.push_back(cv::Point3d(10.0, 10.0, -10.0));
    reprojectsrc.push_back(cv::Point3d(10.0, -10.0, -10.0));
    reprojectsrc.push_back(cv::Point3d(10.0, -10.0, 10.0));
    reprojectsrc.push_back(cv::Point3d(-10.0, 10.0, 10.0));
    reprojectsrc.push_back(cv::Point3d(-10.0, 10.0, -10.0));
    reprojectsrc.push_back(cv::Point3d(-10.0, -10.0, -10.0));
    reprojectsrc.push_back(cv::Point3d(-10.0, -10.0, 10.0));

    eulerAngle = cv::Mat(3, 1, CV_64FC1);
    out_intrinsics = cv::Mat(3, 3, CV_64FC1);
    out_rotation = cv::Mat(3, 3, CV_64FC1);
    out_translation = cv::Mat(3, 1, CV_64FC1);
    headPose.resize(8);
    poseMatrix = cv::Mat(3, 4, CV_64FC1);
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

   std::vector<cv::Point2d> get_2d_image_points(dlib::full_object_detection &shape)
   {
        std::vector<cv::Point2d> image_points;

        image_points.push_back(cv::Point2d(shape.part(17).x(), shape.part(17).y())); //#17 left brow left corner
        image_points.push_back(cv::Point2d(shape.part(21).x(), shape.part(21).y())); //#21 left brow right corner
        image_points.push_back(cv::Point2d(shape.part(22).x(), shape.part(22).y())); //#22 right brow left corner
        image_points.push_back(cv::Point2d(shape.part(26).x(), shape.part(26).y())); //#26 right brow right corner
        image_points.push_back(cv::Point2d(shape.part(36).x(), shape.part(36).y())); //#36 left eye left corner
        image_points.push_back(cv::Point2d(shape.part(39).x(), shape.part(39).y())); //#39 left eye right corner
        image_points.push_back(cv::Point2d(shape.part(42).x(), shape.part(42).y())); //#42 right eye left corner
        image_points.push_back(cv::Point2d(shape.part(45).x(), shape.part(45).y())); //#45 right eye right corner
        image_points.push_back(cv::Point2d(shape.part(31).x(), shape.part(31).y())); //#31 nose left corner
        image_points.push_back(cv::Point2d(shape.part(35).x(), shape.part(35).y())); //#35 nose right corner
        image_points.push_back(cv::Point2d(shape.part(48).x(), shape.part(48).y())); //#48 mouth left corner
        image_points.push_back(cv::Point2d(shape.part(54).x(), shape.part(54).y())); //#54 mouth right corner
        image_points.push_back(cv::Point2d(shape.part(57).x(), shape.part(57).y())); //#57 mouth central bottom corner
        image_points.push_back(cv::Point2d(shape.part(8).x(), shape.part(8).y()));   //#8 chin corner
        return image_points;

   }

   cv::Mat get_camera_matrix(float focal_length, cv::Point2d center)
   {
        cv::Mat camera_matrix = (cv::Mat_<double>(3,3) << focal_length, 0, center.x, 0 , focal_length, center.y, 0, 0, 1);
        return camera_matrix;
   }

   void calculatePose(dlib::full_object_detection shape, cv::Mat image)
   {
         imagePoints = get_2d_image_points(shape);
         double focal_length = image.cols;
         cv::Mat camera_matrix = get_camera_matrix(focal_length, cv::Point2d(image.cols/2,image.rows/2));
         cv::Mat rotation_vector;
         cv::Mat rotation_matrix;
         cv::Mat translation_vector;

         cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, cv::DataType<double>::type);


         //bool solvePnP(InputArray objectPoints, InputArray imagePoints, InputArray cameraMatrix, InputArray distCoeffs, OutputArray rvec, OutputArray tvec, bool useExtrinsicGuess=false, int flags=ITERATIVE )
         cv::solvePnP(modelPoints, imagePoints, camera_matrix, dist_coeffs, rotation_vector, translation_vector);

         //void projectPoints(InputArray objectPoints, InputArray rvec, InputArray tvec, InputArray cameraMatrix, InputArray distCoeffs, OutputArray imagePoints, OutputArray jacobian=noArray(), double aspectRatio=0 )
         cv::projectPoints(reprojectsrc, rotation_vector, translation_vector, camera_matrix, dist_coeffs, headPose);

         // Calculate euler angle
         cv::Rodrigues(rotation_vector, rotationMatrix);
         cv::hconcat(rotationMatrix, translation_vector, poseMatrix);
         cv::decomposeProjectionMatrix(poseMatrix, out_intrinsics, out_rotation, out_translation, cv::noArray(), cv::noArray(), cv::noArray(), eulerAngle);
   }

   void calculateDelaunay(dlib::full_object_detection shape, cv::Rect rect)
   {
        cv::Subdiv2D subdiv(rect);

        for (unsigned long j = 0; j < shape.num_parts(); j++) {
            cv::Point2f point(shape.part(j).x(), shape.part(j).y());
            subdiv.insert(point);
        }

        std::vector<cv::Vec6f> triangleList;
        subdiv.getTriangleList(triangleList);
        std::vector<cv::Point2f> pt(3);

        for( size_t i = 0; i < triangleList.size(); i++ )
        {
            std::vector<cv::Point2f> pt(3);
            cv::Vec6f t = triangleList[i];
            pt[0] = cv::Point2f(t[0], t[1]);
            pt[1] = cv::Point2f(t[2], t[3]);
            pt[2] = cv::Point2f(t[4], t[5]);

            delaunay.push_back(pt);
        }
   }


  // The format of mat should be BGR or Gray
  // If converting 4 channels to 3 channls because the format could be BGRA or
  // ARGB
  virtual inline int detRaw(const cv::Mat& image) {
    cv::Mat cropped;



    if (image.empty()) return 0;

    // Make sure image is grey
    if (image.channels() > 1) {
        cv::cvtColor(image, image, CV_BGR2GRAY);
    }


    // Check if we got a previous face, if we do use the rectangle from that face to search the next
    if (lastFace.area() > 0 && 0 <= lastFace.x
        && 0 <= lastFace.width
        && lastFace.x + lastFace.width <= image.cols
        && 0 <= lastFace.y
        && 0 <= lastFace.height
        && lastFace.y + lastFace.height <= image.rows) {

        // Create a new rectangle from the last face, and add the padding to search a slightly bigger area since the face has probably moved a few pixels
        returnRect = cv::Rect(lastFace.x - FRAME_PADDING, lastFace.y - FRAME_PADDING, lastFace.width + (FRAME_PADDING * 2), lastFace.height + (FRAME_PADDING * 2));

        // Make sure the rectangle fits within the original image
        if (returnRect.x < 0) {
            returnRect.x = 0;
        }
        if (returnRect.y < 0) {
            returnRect.y = 0;
        }
        if (returnRect.x + returnRect.width >= image.cols) {
            returnRect.width = image.cols - returnRect.x;
        }
        if (returnRect.y+returnRect.height >= image.rows) {
            returnRect.height = image.rows-returnRect.y;
        }

        // crop the image to the new rectangle
        cropped = image(returnRect);
    } else {
        // Reset the enlarged search since we dont have any previous face, this is equal of setting it to null.
        returnRect = cv::Rect(0, 0, 0, 0);
        cropped = image;
    }

    // Create dlib images, required by the detector
    dlib::cv_image<unsigned char> croppedImg(cropped);
    dlib::cv_image<unsigned char> img(image);

    // Try to find the face
    mRets = mFaceDetector(croppedImg);
    // Reset the landmarks
    mFaceShapeMap.clear();
    imagePoints.clear();
    headPose.clear();

     // Process shape, make sure a face was found,
     if (mRets.size() != 0 && mLandMarkModel.empty() == false) {
       for (unsigned long j = 0; j < mRets.size(); ++j) {

         // Since we cropped the image we need to resize it again so that the landmarks gets correct coordinates
         dlib::rectangle face(mRets[j].left() + returnRect.tl().x, mRets[j].top() + returnRect.tl().y, mRets[j].right() + returnRect.tl().x, mRets[j].bottom() + returnRect.tl().y);

         // Find landmarks
         dlib::full_object_detection shape = shapePredictor(img, face);

         mFaceShapeMap[j] = shape;

         // Set the face bounding box to use for the next frame
         lastFace = cv::Rect(cv::Point2i(face.left(), face.top()), cv::Point2i(face.right(), face.bottom()));

         // Head pose
         calculatePose(shape, image);

         //delaunay
         calculateDelaunay(shape, lastFace);


       }
     } else {
        // If no face was found we reset this to have the next frame detect from a clean sheet.
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

  std::vector<cv::Point2d>& getHeadPose() {
    return headPose;
  }

  cv::Mat& getEulerAngle() {
      return eulerAngle;
  }

  std::vector<std::vector<cv::Point2f>>& getDelaunay() {
      return delaunay;
    }

};
