/*
 * jni_pedestrian_det.cpp using google-style
 *
 *  Created on: Oct 20, 2015
 *      Author: Tzutalin
 *
 *  Copyright (c) 2015 Tzutalin. All rights reserved.
 */
#include <android/bitmap.h>
#include <jni_common/jni_bitmap2mat.h>
#include <jni_common/jni_primitives.h>
#include <jni_common/jni_fileutils.h>
#include <jni_common/jni_utils.h>
#include <detector.h>
#include <jni.h>


using namespace cv;

extern JNI_VisionDetRet* g_pJNI_VisionDetRet;

namespace {

#define JAVA_NULL 0
using DetectorPtr = DLibHOGFaceDetector*;

class JNI_FaceDet {
 public:
  JNI_FaceDet(JNIEnv* env) {
    jclass clazz = env->FindClass(CLASSNAME_FACE_DET);
    mNativeContext = env->GetFieldID(clazz, "mNativeFaceDetContext", "J");
    env->DeleteLocalRef(clazz);
  }

  DetectorPtr getDetectorPtrFromJava(JNIEnv* env, jobject thiz) {
    DetectorPtr const p = (DetectorPtr)env->GetLongField(thiz, mNativeContext);
    return p;
  }

  void setDetectorPtrToJava(JNIEnv* env, jobject thiz, jlong ptr) {
    env->SetLongField(thiz, mNativeContext, ptr);
  }

  jfieldID mNativeContext;
};

// Protect getting/setting and creating/deleting pointer between java/native
std::mutex gLock;

std::shared_ptr<JNI_FaceDet> getJNI_FaceDet(JNIEnv* env) {
  static std::once_flag sOnceInitflag;
  static std::shared_ptr<JNI_FaceDet> sJNI_FaceDet;
  std::call_once(sOnceInitflag, [env]() {
    sJNI_FaceDet = std::make_shared<JNI_FaceDet>(env);
  });
  return sJNI_FaceDet;
}

DetectorPtr const getDetectorPtr(JNIEnv* env, jobject thiz) {
  std::lock_guard<std::mutex> lock(gLock);
  return getJNI_FaceDet(env)->getDetectorPtrFromJava(env, thiz);
}

// The function to set a pointer to java and delete it if newPtr is empty
void setDetectorPtr(JNIEnv* env, jobject thiz, DetectorPtr newPtr) {
  std::lock_guard<std::mutex> lock(gLock);
  DetectorPtr oldPtr = getJNI_FaceDet(env)->getDetectorPtrFromJava(env, thiz);
  if (oldPtr != JAVA_NULL) {
    DLOG(INFO) << "setMapManager delete old ptr : " << oldPtr;
    delete oldPtr;
  }

  if (newPtr != JAVA_NULL) {
    DLOG(INFO) << "setMapManager set new ptr : " << newPtr;
  }

  getJNI_FaceDet(env)->setDetectorPtrToJava(env, thiz, (jlong)newPtr);
}

}  // end unnamespace

#ifdef __cplusplus
extern "C" {
#endif


#define DLIB_FACE_JNI_METHOD(METHOD_NAME) \
  Java_com_tzutalin_dlib_FaceDet_##METHOD_NAME

void JNIEXPORT
    DLIB_FACE_JNI_METHOD(jniNativeClassInit)(JNIEnv* env, jclass _this) {}

jobjectArray getDetectResult(JNIEnv* env, DetectorPtr faceDetector,
                             const int& size) {
  jobjectArray jDetRetArray = JNI_VisionDetRet::createJObjectArray(env, size);
  for (int i = 0; i < size; i++) {
    jobject jDetRet = JNI_VisionDetRet::createJObject(env);
    env->SetObjectArrayElement(jDetRetArray, i, jDetRet);
    dlib::rectangle rect = faceDetector->getResult()[i];
    g_pJNI_VisionDetRet->setRect(env, jDetRet, rect.left(), rect.top(),
                                 rect.right(), rect.bottom());
    g_pJNI_VisionDetRet->setLabel(env, jDetRet, "face");
    std::unordered_map<int, std::vector<cv::Point2f>>& faceShapeMap =
        faceDetector->getFaceShapeMap();
    if (faceShapeMap.find(i) != faceShapeMap.end()) {
      std::vector<cv::Point2f> shape = faceShapeMap[i];
      for (unsigned long j = 0; j < shape.size(); j++) {
        float floatX = shape[j].x;
        int x = (floatX >= 0) ? (int)(floatX + 0.5) : (int)(floatX - 0.5);
        float floatY = shape[j].y;
        int y = (floatY >= 0) ? (int)(floatY + 0.5) : (int)(floatY - 0.5);
        // Call addLandmark
        g_pJNI_VisionDetRet->addLandmark(env, jDetRet, x, y);
      }
    }
  }
  return jDetRetArray;
}

void rotateMat(cv::Mat &matImage, int rotFlag) {
    //1=ClockWise
    //2=CounterClockWise
    //3=180degree
    if(rotFlag == 1) {cv::transpose(matImage, matImage);cv::flip(matImage, matImage, 1);}
    else if(rotFlag == 2) {cv::transpose(matImage, matImage);cv::flip(matImage, matImage, -1);}
    else if(rotFlag == 3) {cv::flip(matImage, matImage, -1);}
}

JNIEXPORT jobjectArray JNICALL
    DLIB_FACE_JNI_METHOD(jniRawDetect)(JNIEnv* env, jobject thiz, jbyteArray rawBytes, jint rotation, jint width, jint height) {
  jbyte* b_data = env->GetByteArrayElements(rawBytes, 0);
  cv::Mat bgrMat;
  cv::Mat yuvMat = cv::Mat(height+height/2, width, CV_8UC1, (unsigned char*)b_data);

  cv::cvtColor(yuvMat, bgrMat, CV_YUV2GRAY_NV21);

  if(rotation == 90) {rotateMat(bgrMat, 1);}
  else if(rotation == 180) {rotateMat(bgrMat, 3);}
  else if(rotation == 270 || rotation == -90) {rotateMat(bgrMat, 2);}

  DetectorPtr detPtr = getDetectorPtr(env, thiz);
  jint size = detPtr->detRaw(bgrMat);
  return getDetectResult(env, detPtr, size);
}

JNIEXPORT jobjectArray JNICALL
    DLIB_FACE_JNI_METHOD(jniBitmapDetect)(JNIEnv* env, jobject thiz,
                                          jobject bitmap) {
  LOG(INFO) << "jniBitmapFaceDet";
  cv::Mat rgbaMat;
  cv::Mat bgrMat;
  jniutils::ConvertBitmapToRGBAMat(env, bitmap, rgbaMat, true);
  cv::cvtColor(rgbaMat, bgrMat, cv::COLOR_RGBA2BGR);
  DetectorPtr detPtr = getDetectorPtr(env, thiz);
  jint size = detPtr->det(bgrMat);
#if 0
  cv::Mat rgbMat;
  cv::cvtColor(bgrMat, rgbMat, cv::COLOR_BGR2RGB);
  cv::imwrite("/sdcard/ret.jpg", rgbaMat);
#endif
  return getDetectResult(env, detPtr, size);
}

jint JNIEXPORT JNICALL DLIB_FACE_JNI_METHOD(jniInit)(JNIEnv* env, jobject thiz,
                                                     jstring jLandmarkPath) {
  LOG(INFO) << "jniInit";
  std::string landmarkPath = jniutils::convertJStrToString(env, jLandmarkPath);
  DetectorPtr detPtr = new DLibHOGFaceDetector(landmarkPath);
  setDetectorPtr(env, thiz, detPtr);
  ;
  return JNI_OK;
}

jint JNIEXPORT JNICALL
    DLIB_FACE_JNI_METHOD(jniDeInit)(JNIEnv* env, jobject thiz) {
  LOG(INFO) << "jniDeInit";
  setDetectorPtr(env, thiz, JAVA_NULL);
  return JNI_OK;
}

#ifdef __cplusplus
}
#endif
