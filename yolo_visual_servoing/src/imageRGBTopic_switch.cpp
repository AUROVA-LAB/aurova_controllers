
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <Eigen/Dense>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
#include <opencv2/core/core.hpp>
#include <math.h>
#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>
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
std::string cam1_Topic = "/camera2/color/image_raw/compressed";
std::string cam2_Topic = "/camera2/color/image_raw/compressed";


//Publisher
ros::Publisher pub_img_out;

// switch topic
float camBool = 0;

// BLUE
//std_msgs::Float32 back_rec; 
//std_msgs::Float32 ford_rec;

//ros::Publisher ford_rec_publisher;
//ros::Publisher back_rec_publisher;

void callback_select_img(const yolo_visual_servoing::selectImg::ConstPtr &bool_cam){
  camBool = bool_cam->switch_CamTopicName.data;
}


//////////////////////////////////////callback

void callback(const CompressedImageConstPtr& image_1, const CompressedImageConstPtr& image_2)//, const ImageConstPtr& image_3)
{
   ros::Rate rate(12.5);
  cv_bridge::CvImagePtr cv_ptr_1, cv_ptr_2;
  try
  {
    cv_ptr_1 = cv_bridge::toCvCopy(image_1, sensor_msgs::image_encodings::RGB8);   
    cv_ptr_2 = cv_bridge::toCvCopy(image_2, sensor_msgs::image_encodings::RGB8);   
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  cv::Mat img_cam1  = cv_ptr_1->image;  
  cv::Mat img_cam2  = cv_ptr_2->image; 


  sensor_msgs::ImagePtr image_msg;
  
  if (camBool==1)
    image_msg = cv_bridge::CvImage(std_msgs::Header(), "rgb8", img_cam1).toImageMsg();
  else
  if (camBool ==0 )
    image_msg = cv_bridge::CvImage(std_msgs::Header(), "rgb8", img_cam2).toImageMsg();
  else{
  image_msg =  cv_bridge::CvImage(std_msgs::Header(), "rgb8", cv::Mat::zeros(720, 1280, CV_64FC1)).toImageMsg(); 
  }

  image_msg->header.stamp     = image_1->header.stamp;

  pub_img_out.publish(image_msg);

  //ford_rec.data = 1000;
 // back_rec.data = 1000;

  //ford_rec_publisher.publish(ford_rec);
  //back_rec_publisher.publish(back_rec);
  rate.sleep();

}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "switch_input_img");
  ros::NodeHandle nh;  
  ros::Subscriber sub = nh.subscribe<yolo_visual_servoing::selectImg>("/bool_cam_control", 10, callback_select_img); 

  nh.getParam("/cam1_Topic", cam1_Topic);
  nh.getParam("/cam2_Topic", cam2_Topic);
  message_filters::Subscriber<CompressedImage>  cam1_sub(nh, cam1_Topic , 10);
  message_filters::Subscriber<CompressedImage>  cam2_sub(nh, cam2_Topic, 10);

  typedef sync_policies::ApproximateTime<CompressedImage, CompressedImage> MySyncPolicy;
  Synchronizer<MySyncPolicy> sync(MySyncPolicy(50), cam1_sub, cam2_sub);
  //TimeSynchronizer<detection_msgs::BoundingBoxes,Image> sync(yoloBbox_sub, depthImg_sub, 10);
  sync.registerCallback(boost::bind(&callback, _1, _2 ));
  
  pub_img_out = nh.advertise<sensor_msgs::Image>("/yoloImg_input", 10);
 // ford_rec_publisher = nh.advertise <std_msgs::Float32>("/forward_recommended_velocity" , 10);
  //back_rec_publisher = nh.advertise <std_msgs::Float32>("/backward_recommended_velocity", 10);

  ros::spin();
}
