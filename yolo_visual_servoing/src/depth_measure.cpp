#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <Eigen/Dense>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
#include <opencv2/core/core.hpp>
#include <std_msgs/Header.h>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include <iostream>
#include <chrono>
#include <string>
#include <sensor_msgs/PointCloud2.h>
#include <opencv2/highgui/highgui.hpp>
#include <std_msgs/Float32.h>

#include <math.h>
#include <detection_msgs/BoundingBox.h>
#include <detection_msgs/BoundingBoxes.h>

#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include "ackermann_msgs/AckermannDriveStamped.h"
#include <nav_msgs/Odometry.h>

#include "geometry_msgs/Vector3.h"
#include "geometry_msgs/Quaternion.h"
#include "tf/transform_datatypes.h"

#include <visualization_msgs/Marker.h>

#include <iostream>
#include <fstream>

//#include <mutex>

//std::mutex mutex_lock;

#define PI 3.14159265359

using namespace Eigen;
using namespace sensor_msgs;
using namespace message_filters;
using namespace std;

/// input topics 
std::string objYoloTopic  =  "/yolov5/detections";
std::string objDepthTopic = "/depthVS_input";
std::string yoloImgTopic = "/camera/color/image_raw";

typedef std::chrono::high_resolution_clock Clock;

//================================================== Caracteristicas del BLUE ==========================================

float phi = 0;// orientacion actual del robot
float phi_est1 = 0; // angulo para restar al entrar en el estado 1
float phi_est4 = 0; // angulo para restar al entrar en el estado 4
float phi_cine = 0; // angulo para restar cuando entra en el cinematico.
float d = 1.1; // distancia entre los ejes de las llantas del BLue
float dist_a = 0.5; // distancia de la base del robot al punto de control

//float w_ant   = 0;// velocidad anterior angular de blue;
float u_ant   = 0;// velocidad anterior lineal de blue;
float psi_ant = 0;// steering blue;
//float phi_ant = 0;// orientacion anterior del robot blue

float u_max = 0.5; //velocidad maxima en metros/segundos que peude ir el blue 
float psi_max = 0.4363;//*PI/180.0 ; // angulo de steering maximo que puede llegar el blue

bool mod_blue = true; // modo de cinematica del robot blue, true es para cinematica hacia adelante, 0 es hacia atras
double psi = 0.0; // steering blue;
double u_vel =0.0; // velocidad lineal blue;
float camBool = 1; //

// profunfidades
float zr = 0;
float zd = 0;

ros::Publisher vis_pub;
ros::Publisher pub_rgb_yolo;

visualization_msgs::Marker marker;
sensor_msgs::ImagePtr image_msg;


// =====================================================================================================================

// los valores anteriores son respecto a la camra 1 o la camara 2, para convertir con respecto al robot se aplixa la sigueitne transformacion:
Eigen::MatrixXf bb_perce(1,1) ; 
Eigen::MatrixXf Mcam(3,4);   // matrix de calibracionde la camara que utilizo.

int Ox = 0 ; //punto medio en x del objeto a servo visual
int Oy = 0 ; //punto medio en y del objeto a servo visual
std::ofstream myfile;



//////////////////////////////////////callback

void callback(const boost::shared_ptr<const detection_msgs::BoundingBoxes> &bb_data, const ImageConstPtr& depthImg_1, const ImageConstPtr& yoloImg)

{ 

  auto t1 = Clock::now(); // lectura de tiempo para estimar el retardo del proceso
  //////////////// get dpeth image and color image and convert to eigen matrix ////////////////
  cv_bridge::CvImagePtr cv_ptr, cv_yolo;
  try
  { // lectura de las imagenes    
    cv_ptr = cv_bridge::toCvCopy(depthImg_1, sensor_msgs::image_encodings::MONO16);     // produndidad lidar+camara delantera
    cv_yolo = cv_bridge::toCvCopy(yoloImg, sensor_msgs::image_encodings::RGB8);  // yolo image 
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }
  
  detection_msgs::BoundingBoxes data = *bb_data;
  int num_obj_detect = data.bounding_boxes.size();
  Eigen::MatrixXf xyz_object_camera(3,num_obj_detect); // lugar dodne guardo las coordenadas de los objetos 

  cv::Mat img_servo_visual  = cv_yolo->image;  // imagen servovisual que entra en el control
  Eigen::Matrix<float,Dynamic,Dynamic> depth_data; 
  Eigen::Matrix<float,Dynamic,Dynamic> depth_sc; // para mediana, elimino 0


  // Lazo que recorre los objetos detectados y calcula los puntos X,Y y Z 

  for(int i=0 ;i<num_obj_detect; i++)
	{
    cv::Mat depth_img;
    depth_img  = cv_ptr->image; 
    cv2eigen(depth_img,depth_data); 

    std::cout<<"----------------->Obj_detect: "<<i<<std::endl;  


    ////////////////////////////////////////////////////////////////////////////

    uint xmin = data.bounding_boxes[i].xmin;
    uint ymin = data.bounding_boxes[i].ymin;
    uint xmax = data.bounding_boxes[i].xmax;
    uint ymax = data.bounding_boxes[i].ymax;

    std::string class_name = data.bounding_boxes[i].Class;

    float depth_ave = 0;   //  average distance of object
    float mas_cercano = 10000.0;
    int cont_pix=0;        // number of pixels 

    float bb_per = bb_perce(0,0);   // bounding box reduction percentage 
    bb_per = bb_per/2;

    uint start_x = (1-bb_per) * xmin + (bb_per * xmax);
    uint end_x   = (1-bb_per) * xmax + (bb_per * xmin);
    uint start_y = (1-bb_per) * ymin + (bb_per * ymax);
    uint end_y   = (1-bb_per) * ymax + (bb_per * ymin);

    std::vector<int> vec_std_depth; // vector para medianas
   
    for (uint iy = start_y;iy<end_y; iy++)
      for (uint ix = start_x;ix<end_x; ix++){        
        // sumatoria de elemetnos
        if(depth_data(iy,ix)>0){
          depth_ave += depth_data(iy,ix);
          cont_pix++;
          vec_std_depth.push_back(depth_data(iy,ix));
          if (depth_data(iy,ix)<mas_cercano)
            mas_cercano = depth_data(iy,ix);
        }       
      }

    if(depth_ave == 0 && cont_pix==0){
      cont_pix = 1;
      vec_std_depth.push_back(0);
    }


    //std::vector<int> vec_std_depth(depth_data.data(), depth_data.data() + depth_data.rows() * depth_data.cols()); // vector de EIGEN a std para luego ordenar   
    // ordenar datos para sacar la media
    int n = sizeof(vec_std_depth) / sizeof(vec_std_depth[0]);  
    sort(vec_std_depth.begin(), vec_std_depth.begin() + n, greater<int>());

    int tam = vec_std_depth.size();
    float median = 0;
    if (tam % 2 == 0) {  
        median = (vec_std_depth[((tam)/2) -1] + vec_std_depth[(tam)/2])/2.0; 
    }      
    else { 
       if(tam==1)
        median = vec_std_depth[tam];
      else
        median = vec_std_depth[tam/2];
    }  
       
    vec_std_depth.clear();  
       

     
    //orden de los puntos
     /* 2----3
        |    |
        0----1
     */

    Ox = (xmax +xmin) /2;  // puntos medios del objeto 
    Oy = (ymax +ymin) /2;

     ///Estadisticas meidana, promedio , el punto mas cercano y el centroide del BB
    median = median*200.0/pow(2,16);
    depth_ave = ((depth_ave/cont_pix)*200.0)/pow(2,16); //// Promedio de los valores
    mas_cercano = mas_cercano*200.0/pow(2,16); 
    std :: cout << "Centro: "<<Ox<<", "<<Oy<<std::endl; 
    float centro =depth_data(Oy,Ox)*200.0/pow(2,16);  

    float x_obj = Ox; // calculo el x deseado con el punto medio del recuadro que da yolo con menor distancia xmin+xmax/2 
    float y_obj = Oy; // calculo el y deseado con el punto medio del recuadro que da yolo con menor distancia xmin+xmax/2 ;
    x_obj = (x_obj-Mcam(0,2))/Mcam(0,0) * depth_ave;
    y_obj = (y_obj-Mcam(1,2))/Mcam(1,1) * depth_ave;
    float z_obj = sqrt(pow(depth_ave,2)-pow(x_obj,2)-pow(y_obj,2)); // convierto la profundidad que era en linea recta desde la camara hasta el objeto , en una distancia en solo en el eje z
    
    Eigen::MatrixXf cTr(4,4); // matriz homogenea para convertir los datos xyz de la camara1 o camara 2 con respecto al robot
    Eigen::MatrixXf xyz(4,1); // distancias calculadas a la cmara
    Eigen::MatrixXf xyz_rot(4,1); // distancias calculadas a la camara y rotadas

    xyz <<  x_obj, y_obj, z_obj, 1.0;
    

    cTr << -0.0099 ,-0.2325 ,0.9725   ,0.0 
          ,-0.9999 ,0.0061  ,-0.0087  ,0.0
          ,-0.0039 ,-0.9726 ,-0.23268 ,0.0
          ,0       ,0       ,0        ,1.0000;
    
      
    xyz_rot = cTr * xyz;

    xyz_object_camera(0,i) = xyz_rot(0,0);
    xyz_object_camera(1,i) = xyz_rot(1,0);
    xyz_object_camera(2,i) = xyz_rot(2,0);   
    
    std::cout<<"XYZ_camara:"<<xyz_object_camera(0,i)<<", "<<xyz_object_camera(1,i)<<", "<<xyz_object_camera(2,i)<<std::endl;
    std :: cout << "Promedio:"<<depth_ave<<std::endl;
    std :: cout << "Mediana:"<<median<<std::endl;     
    std :: cout << "Mas cercano:"<<mas_cercano<<std::endl;   
    std :: cout << "Centro: "<<centro<<std::endl;   

        
      myfile << 1 <<','<< i <<','<< depth_ave<<','<<median<<','<<mas_cercano<<','<<centro<<"\n";     

    cv::Point pt1(xmin, ymin);
    cv::Point pt2(xmax, ymax);
    cv::rectangle(img_servo_visual, pt1, pt2,cv::Scalar(0, 255, 0),2);    // objeto detectado con profundidades
    cv::Point pt_bb1(start_x, start_y);
    cv::Point pt_bb2(end_x, end_y);
    cv::rectangle(img_servo_visual, pt_bb1, pt_bb2,cv::Scalar(0, 0, 255),2);    // bb_proundidades

    int xx = 0 ;
    int med = (xmax-xmin)/2 + xmin;
    if (med>600)
      xx = xmin-50;
    else
     xx = xmin-20;
    if(med<50)
      xx = xmin+20;

    cv::putText(img_servo_visual,  cv::format("%.2f", depth_ave), cv::Point(xx, ymin-10), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(0, 0, 0), 2); 
    //cv::putText(img_servo_visual,  cv::format("D: %.2f", depth_ave), cv::Point(xmin-20, ymin-20), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(0, 0, 0), 2);   
     
         
  }
 
     //==========================================================================================================================================
  
 
  // Viusalizacion del objeto en RVIZ
 
  marker.header.frame_id = "velodyne";
  marker.header.stamp = ros::Time();
  marker.ns = "my_namespace";

  for (int j=0;j<num_obj_detect;j++)
  {
    marker.id = j;
    marker.type = visualization_msgs::Marker::SPHERE;
    marker.action = visualization_msgs::Marker::ADD;
    marker.pose.position.x = xyz_object_camera(0,j)-0.3;
    marker.pose.position.y = xyz_object_camera(1,j)+0.03;
    marker.pose.position.z = xyz_object_camera(2,j)-0.05;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;
    marker.scale.x = 0.1;
    marker.scale.y = 0.1;
    marker.scale.z = 0.1;
    marker.color.a = 1.0; // Don't forget to set the alpha!
    marker.color.r = 0.0;
    marker.color.g = 1.0;
    marker.color.b = 0.0;
    vis_pub.publish( marker );
  }

  image_msg->header.stamp     = yoloImg->header.stamp;
  image_msg = cv_bridge::CvImage(std_msgs::Header(), "rgb8", img_servo_visual).toImageMsg();
  
  

  /////////////////////////// imprimo tiempos
  auto t2= Clock::now();
  float time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count()/1000000000.0;
  std::cout<<"Tiempo: "<<time<<std::endl;  
 
}


int main(int argc, char** argv)
{

  ros::init(argc, argv, "VisualServoYolo");
  ros::NodeHandle nh;  
  
  /// Load Parameters
  nh.getParam("/detection_BoundingBoxes", objYoloTopic);
  nh.getParam("/depth_lidar_image", objDepthTopic);
  nh.getParam("/yoloImg_input" , yoloImgTopic);
  
  XmlRpc::XmlRpcValue param;
  nh.getParam("/control_params/camera1_640_480", param);
  // matriz de calibracion de parametros intrinsecos de la camara 1
  Mcam   <<  (double)param[0] ,(double)param[1] ,(double)param[2] ,(double)param[3]
           ,(double)param[4] ,(double)param[5] ,(double)param[6] ,(double)param[7]
           ,(double)param[8] ,(double)param[9] ,(double)param[10],(double)param[11];
   
  nh.getParam("/control_params/bb_perce", param);
  bb_perce <<  (double)param[0];

  std::cout<<"Parametros cargados"<<std::endl;
         

  message_filters::Subscriber<detection_msgs::BoundingBoxes> iaBbox_sub(nh, objYoloTopic , 10);
  message_filters::Subscriber<Image>  depthImg_sub(nh, objDepthTopic, 10);
  message_filters::Subscriber<Image>  iaInimg_sub(nh, yoloImgTopic, 10);

  typedef sync_policies::ApproximateTime<detection_msgs::BoundingBoxes, Image, Image> MySyncPolicy;
  Synchronizer<MySyncPolicy> sync(MySyncPolicy(100), iaBbox_sub, depthImg_sub, iaInimg_sub);
  sync.registerCallback(boost::bind(&callback, _1, _2, _3));
  ros::Rate loopRate(10.4);

  std::cout<<"Sincronizacion cargada"<<std::endl;

  //mutex_lock.unlock();
  pub_rgb_yolo = nh.advertise<sensor_msgs::Image>("/Img_distances", 10);
  vis_pub = nh.advertise<visualization_msgs::Marker>( "visualization_marker", 1 );
  image_msg = cv_bridge::CvImage(std_msgs::Header(), "rgb8", cv::Mat::zeros(480, 640, CV_64FC1)).toImageMsg();

  
  myfile.open ("/home/epvs/Escritorio/resultados_depth_estimation/resultados.csv");
  myfile << "Muestra,Objeto,Promedio,Mediana,Mas cercano,Centro,X,Y,Z\n";

  
  while(ros::ok()){    
    ros::spinOnce();
    pub_rgb_yolo.publish(image_msg); // publish image of yolo whit distance 
    loopRate.sleep();    
  } 
  myfile.close();
  
  return 0;
}
