#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <Eigen/Dense>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
#include <opencv2/core/core.hpp>
#include <math.h>
#include <std_msgs/Header.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <iostream>
#include <chrono>
#include <sensor_msgs/PointCloud2.h>
#include <opencv2/highgui/highgui.hpp>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include "yolo_visual_servoing/selectImg.h"



using namespace Eigen;
using namespace sensor_msgs;
using namespace message_filters;
using namespace std;

/// input topics 
std::string depth1_Topic = "/depthImage";
std::string depth2_Topic = "/camera2/aligned_depth_to_color/image_raw";


//Publisher
ros::Publisher pub_img_out;

// switch topic
bool camBool = true;

void callback_select_img(const yolo_visual_servoing::selectImg::ConstPtr &bool_cam){
  camBool = bool_cam->switch_CamTopicName.data;
}


//////////////////////////////////////callback

void callback(const ImageConstPtr& image_1, const ImageConstPtr& image_2)//, const ImageConstPtr& image_3)
{
  cv_bridge::CvImagePtr cv_ptr_1, cv_ptr_2;
  try
  {
    cv_ptr_1 = cv_bridge::toCvCopy(image_1, sensor_msgs::image_encodings::MONO16);   
    cv_ptr_2 = cv_bridge::toCvCopy(image_2, sensor_msgs::image_encodings::TYPE_16UC1);   
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  cv::Mat img_cam1  = cv_ptr_1->image;  
  cv::Mat img_cam2  = cv_ptr_2->image; 


  sensor_msgs::ImagePtr image_msg;
  //image_msg = cv_bridge::CvImage(std_msgs::Header(), "mono16", img_cam1).toImageMsg();
  if (camBool)
    image_msg = cv_bridge::CvImage(std_msgs::Header(), "mono16", img_cam1).toImageMsg();
  else
    image_msg = cv_bridge::CvImage(std_msgs::Header(), "mono16", img_cam2).toImageMsg();

  image_msg->header.stamp     = image_1->header.stamp;

  pub_img_out.publish(image_msg);

}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "switch_input_depth");
  ros::NodeHandle nh;  
  ros::Subscriber sub = nh.subscribe<yolo_visual_servoing::selectImg>("/bool_cam_control", 10, callback_select_img); 

  //image_transport::ImageTransport depth2_sub(nh);
  //image_transport::Subscriber sub = depth2_sub.subscribe("camera/image", 1, imageCallback);

  nh.getParam("/depth1_Topic", depth1_Topic);
  nh.getParam("/depth2_Topic", depth2_Topic);
  message_filters::Subscriber<Image>  depth1_sub(nh, depth1_Topic , 10);
  message_filters::Subscriber<Image>  depth2_sub(nh, depth2_Topic, 10);

  //message_filters::Subscriber<CompressedImage>  depth2_sub(nh, depth2_Topic, 1);

  typedef sync_policies::ApproximateTime<Image, Image> MySyncPolicy;
  Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), depth1_sub, depth2_sub);
  //TimeSynchronizer<detection_msgs::BoundingBoxes,Image> sync(yoloBbox_sub, depthImg_sub, 10);
  sync.registerCallback(boost::bind(&callback, _1, _2 ));
  
  pub_img_out = nh.advertise<sensor_msgs::Image>("/depthVS_input", 10);


  ros::spin();
}
