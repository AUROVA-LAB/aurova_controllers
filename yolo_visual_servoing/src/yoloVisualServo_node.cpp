#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <Eigen/Dense>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <image_transport/image_transport.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <std_msgs/Header.h>

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
#include "yolo_visual_servoing/selectImg.h"
#include <nav_msgs/Odometry.h>

#include "geometry_msgs/Vector3.h"
#include "geometry_msgs/Quaternion.h"
#include "tf/transform_datatypes.h"

#include <visualization_msgs/Marker.h>

//#include <mutex>

//std::mutex mutex_lock;

#define PI 3.14159265359

using namespace Eigen;
using namespace sensor_msgs;
using namespace message_filters;
using namespace std;

/// input topics 
std::string objYoloTopic  =  "/yolov5/detections";
std::string objDepthTopic1 = "/depthVS_input";
std::string objDepthTopic2 = "/camera2/aligned_depth_to_color/image_raw";
std::string yoloImgTopic = "/yoloImg_input";
std:: string odometryTopic = "/odom";

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

ackermann_msgs::AckermannDriveStamped ackermann_state_rad;
ros::Publisher ford_rec_publisher;
ros::Publisher back_rec_publisher;
ros::Publisher vis_pub;
std_msgs::Float32 back_rec; 
std_msgs::Float32 ford_rec;

nav_msgs::Odometry odom;
visualization_msgs::Marker marker;
sensor_msgs::ImagePtr image_msg;


// =====================================================================================================================

//matrix calibration lidar and camera

/*Eigen::MatrixXf Tlc_1(3,1); // translation matrix lidar-camera1
Eigen::MatrixXf Rlc_1(3,3); // rotation matrix lidar-camera1
Eigen::MatrixXf Tlc_2(3,1); // translation matrix lidar-camera1
Eigen::MatrixXf Rlc_2(3,3); // rotation matrix lidar-camera1
Eigen::MatrixXf Tlc(3,1); // translation matrix lidar-camera1
Eigen::MatrixXf Rlc(3,3); // rotation matrix lidar-camera1*/

Eigen::MatrixXf tcb_1(3,1); // elemtnos de traslacion de la camara 1  a la base
Eigen::MatrixXf rcb_1(3,3); // elementos de rotacion de la camara 1 a la base
Eigen::MatrixXf tcb_2(3,1); // elemtnos de traslacion de la camara 2 a la base
Eigen::MatrixXf rcb_2(3,3); // elementos de rotacion de la camara 2 a la base
Eigen::MatrixXf tcb(3,1);   // elemtnos de traslacion de la camara selecionada a la base
Eigen::MatrixXf rcb(3,3);   // elementos de rotacion de la camara selecionada a la base

Eigen::MatrixXf Mc(3,4);   // camera1 calibration matrix
Eigen::MatrixXf Mc2(3,4);  // camera2 calibration matrix

//Eigen::MatrixXf bTv(4,4);  // matrix base to velodyne 

Eigen::MatrixXf Mcam(3,4);   // matrix de calibracionde la camara que utilizo, esta varia entre la camra frontal y la trasera

Eigen::MatrixXf Spx(8,1);   // 4 puntos x,y en pixeles que da yolo puntos acutales
Eigen::MatrixXf Spxd(8,1);   // 4 puntos x,y en pixeles deseados
//Eigen::MatrixXf Spm(8,1);   // 4 puntos x,y en metros que da yolo puntos acutales
//Eigen::MatrixXf Spmd(8,1);  // 4 puntos x,y en metros deseados

// los valores anteriores son respecto a la camra 1 o la camara 2, para convertir con respecto al robot se aplixa la sigueitne transformacion:
Eigen::MatrixXf xyz_object_camera(4,1);
Eigen::MatrixXf xyz_object_robot(4,1);
Eigen::MatrixXf posXYZ_robot_save(4,1); 
Eigen::MatrixXf posXYZ_robot(4,1); // posicion xyz robot

Eigen::MatrixXf error_xyz(4,1);
Eigen::MatrixXf error_xy(2,1);

// matriz de ganancias para el servovisual / esta se multiplixca con las velocidades de salida del control servovisual pero elemento por elemento teneindo 3 velocidades de salida lineal x lineal y y anfgular
// lambda parametro que se debe escoger desde afuera en un lauch
Eigen::MatrixXf lambda_1(6,1); // ganancias del control servovisual 1
Eigen::MatrixXf lambda_2(6,1); // ganancias del control servovisual 2
Eigen::MatrixXf lambda_c(8,1); // ganancias del control cineamtico 2 (lineal y steering) por cada posible estado (Estado 1: cineamtico y sv1,estado 2: giro adelante estado 3: giro atras,estado 4: cinamtico y sv2)
Eigen::MatrixXf lambda(6,1);  // ganancias seleccionadas en el control

Eigen::MatrixXf k_tanh1(1,2);  // ganancias de las leyes de control de la tangente hiperbolica del servovisual 1 hacia adelante
Eigen::MatrixXf k_tanh2(1,2);  // ganancias de las leyes de control de la tangente hiperbolica del servovisual 2 hacia adelante
Eigen::MatrixXf k_tanh(1,2);   // ganancias de las leyes de control de la tangente hiperbolicaseleccionadas en el control

Eigen::MatrixXf err_X(2,4);   // errores permitidos en X para cambiar estado (columna el estado, fila error minimo, error maximo)
Eigen::MatrixXf err_Y(2,4);   // errores permitidos en X para cambiar estado (columna el estado, fila error minimo, error maximo)
float error_control_x = 1000.0;
float error_control_y = 1000.0;
Eigen::MatrixXf X_desi_estado(4,1); // puntos deseados en X para el control cinematico en cada uno de los estados
Eigen::MatrixXf Y_desi_estado(4,1); // puntos deseados en Y para el control cinematico en cada uno de los estados
 
Eigen::MatrixXf k_1(3,1); // ganancias del control cinematico 1
Eigen::MatrixXf k_2(3,1); // ganancias del control cinematico 2

Eigen::MatrixXf oxy(1,2); // punto deseado del objeto en pixeles 
Eigen::MatrixXf oxy_1(1,2); // punto deseado del objeto en pixeles para la camara frontal configurable desde fichero cfg
Eigen::MatrixXf oxy_2(1,2); // punto deseado del objeto en pixeles para la camara trasera configurable desde fichero cfg
Eigen::MatrixXf select_cam(1,1); //entero decalrado para seleccionar "solo en las pruebas" que camara escoger 0-> forntal 1-> trasera
Eigen::MatrixXf q_vel(6,1) ; // velocidades del control servovisual
Eigen::MatrixXf q_vel_cine(2,1) ; // velocidades del control cinematico
Eigen::MatrixXf q_vel_cam(6,1) ; // velocidades del control servovisual



////// Solo para comparar con el otro metodo 
Eigen::MatrixXf spxd_data(8,1) ; 
Eigen::MatrixXf metodo_data(1,1) ; 

Eigen::MatrixXf bb_perce(1,1) ; 



// switch cam topic
yolo_visual_servoing::selectImg sel_cam;   // topic para selecionar que iamgen debe inmgresar al control servovisual 

//Publisher
ros::Publisher pub_switchCam;
ros::Publisher pub_rgb_yolo;
ros::Publisher odom_pub;
ros::Publisher ackermann_rad_publisher;

// goal points
float x_goal = 0 ,y_goal = 0 ,z_goal = 0; // paara cinematico

//double x1_s = 0.0, y1_s = 0.0; // x1 y y1  current de los puntos para servovisual
//double x2_s = 0.0, y2_s = 0.0; // x1 y y1  current de los puntos para servovisual
//double x3_s = 0.0, y3_s = 0.0; // x1 y y1  current de los puntos para servovisual
//double x4_s = 0.0, y4_s = 0.0; //  x1 y y1  current de los puntos para servovisual
//double z1_4_s = 0.0 ; // ocupo los mismos z para los 4 puntos 

// select control (esta variable decide que controlador se utiliza cunado el blue empieza a navegar)
uint var_control = 0; // (0 cinematico , 1 servovisual, 2 servovisual, 3 sin controlador)
float min_depth_object = 1000.0; //( primer objeto con menor distnacia en el escenario)
bool init_control = false; // boolean var to start control
bool goal_points_start = true; // boolean to get goal points
bool starting = 0;
bool ini_cine=0; // variable para inicializar la posicion del cinematico
bool inicial_phi_est1=1; // variable para inciaizar el angulo en el estado 1
bool inicial_phi_est2=1; // variable para inciaizar el angulo en el estado 1
bool inicial_phi_est4=1; // variable para inciaizar el angulo en el estado 4
bool cinebool =0;

uint width_real = 0;
uint height_real = 0;
int Ox = 0 ; //punto medio en x del objeto a servo visual
int Oy = 0 ; //punto medio en y del objeto a servo visual

float X_cine_d = 0;//valor deseado X para cinematico ;
float Y_cine_d = 0;//valor deseado X para cinematico ;

//bool act_cine = false; // activiar cinematica

//////////////////////////////////////callback
float cont_bool = 0; // cambiar a 0 para camara frontal, a 1 para camara trasera

// contador de estados
uint estado = 1; // los estados son 1-> control servo visual hacia adelante. 2-> cinematico para giro hacia adelante. 3-> cinematico para giro hacia adelante. 4-> Servovivsual hacia atras

void callback(const boost::shared_ptr<const detection_msgs::BoundingBoxes> &bb_data, const ImageConstPtr& depthImg_1, 
    const ImageConstPtr& depthImg_2, const ImageConstPtr& yoloImg, const nav_msgs::Odometry::ConstPtr& odom_msg)

//void callback(const boost::shared_ptr<const detection_msgs::BoundingBoxes> &bb_data, const ImageConstPtr& depthImg_1, 
 //   const ImageConstPtr& depthImg_2, const nav_msgs::Odometry::ConstPtr& odom_msg)
{
 // mutex_lock.lock();
 // ros::Rate rate(10.5); // freceuncia de meustreo de los datos para el control
  // lectura de todos los topics Bounding box de Yolo, Iamgen 1 de profundidad, imagen 2 de profundidad imagen RGB que entro a Yolo y odometria del robot

  cont_bool = select_cam(0,0);// cambiar a 0 para camara frontal, a 1 para camara trasera
  auto t1 = Clock::now(); // lectura de tiempo para estimar el retardo del proceso
  //////////////// get dpeth image and color image and convert to eigen matrix ////////////////
  cv_bridge::CvImagePtr cv_ptr, cv_ptr_2,  cv_yolo;
  try
  { // lectura de las imagenes    
    cv_ptr = cv_bridge::toCvCopy(depthImg_1, sensor_msgs::image_encodings::MONO16);     // produndidad lidar+camara delantera
    cv_ptr_2   = cv_bridge::toCvCopy (depthImg_2, sensor_msgs::image_encodings::TYPE_16UC1);  // profundidad camara realsense trasera
    cv_yolo = cv_bridge::toCvCopy(yoloImg, sensor_msgs::image_encodings::RGB8);  // yolo image 
  }
  catch (cv_bridge::Exception& e)
  {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  
  detection_msgs::BoundingBoxes data = *bb_data;
  uint num_obj_yolo = data.bounding_boxes.size();
  cv::Mat img_servo_visual  = cv_yolo->image;  // imagen servovisual que entra en el control
  Eigen::Matrix<float,Dynamic,Dynamic> depth_data; 

  // Lazo que recorre los objetos detectados y calcula los puntos X,Y y Z 
  for(uint i=0 ;i<num_obj_yolo; i++)
	{

    if (starting == 0)
    {
      starting = 1;
      estado = 1;
      posXYZ_robot << 0.0, 0.0, 0.0, 0.0;
      posXYZ_robot_save << 0.0, 0.0, 0.0, 0.0;
      inicial_phi_est1 = 1;
      inicial_phi_est4 = 1;
    }

    cv::Mat depth_img;

  ////////////// Seleccion de imagen de profundidad   /////////////////
    switch (estado){
      case 1:
      depth_img  = cv_ptr->image; 
      break; 
      case 4:
      depth_img  = cv_ptr_2->image; 
      break;
      default:
      depth_img  = cv_ptr->image; 
      break;
    }

    cv2eigen(depth_img,depth_data); 

    ////////////////////////////////////////////////////////////////////////////

    //std::string clase = data.bounding_boxes[i].Class.c_str();
    //uint clase_int = std::stoi(clase); 
  /*  if (clase_int == 0)
      continue;*/
    //std::cout<<"Clase: "<<clase_int<<std::endl;
    uint xmin = data.bounding_boxes[i].xmin;
    uint ymin = data.bounding_boxes[i].ymin;
    uint xmax = data.bounding_boxes[i].xmax;
    uint ymax = data.bounding_boxes[i].ymax;

    float depth_ave = 0;   //  average distance of object
    int cont_pix=0;        // number of pixels 

    float bb_per = bb_perce(0,0);   // bounding box reduction percentage 
    bb_per = bb_per/2;

    uint start_x = (1-bb_per) * xmin + (bb_per * xmax);
    uint end_x   = (1-bb_per) * xmax + (bb_per * xmin);
    uint start_y = (1-bb_per) * ymin + (bb_per * ymax);
    uint end_y   = (1-bb_per) * ymax + (bb_per * ymin);
 
    for (uint iy = start_y;iy<end_y; iy++)
      for (uint ix = start_x;ix<end_x; ix++){

        if(depth_data(iy,ix)>0){
          depth_ave += depth_data(iy,ix);
          cont_pix++;
        }
      }

    switch (estado){
      case 1: //para visual servo frontal
        depth_ave = ((depth_ave/cont_pix)*200.0)/pow(2,16);
      break;            
      case 4: //para visual servo reversa
        depth_ave = ((depth_ave/cont_pix))/1000;  
      break;
    }
    
    // texto en la iamgen de salida, solo para visualizacion de datos, objeto con profundidad // TAMBIE NMUESTRO EL REACUADOR DESEADO
    cv::Point pt1(xmin, ymin);
    cv::Point pt2(xmax, ymax);
    //string textBox = "Dist:" +std::to_string(depth_ave);
    cv::rectangle(img_servo_visual, pt1, pt2,cv::Scalar(0, 255, 0),2);    // objeto detectado con profundidades

    cv::Point pt_bb1(start_x, start_y);
    cv::Point pt_bb2(end_x, end_y);
    cv::rectangle(img_servo_visual, pt_bb1, pt_bb2,cv::Scalar(0, 0, 255),2);    // bb_proundidades

    //cv::putText(img_servo_visual, textBox, cv::Point(xmin, ymin), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(0, 0, 0), 2);
    cv::putText(img_servo_visual,  cv::format("Dist: %.2f", depth_ave), cv::Point(xmin-20, ymin-20), cv::FONT_HERSHEY_DUPLEX, 1.0, CV_RGB(0, 0, 0), 2);
   
    std::cout<<"Profundidad: "<<depth_ave<<std::endl;


    if(depth_ave < min_depth_object){ //encontrar el objeto con minima distancia <======================================== Solo selecciono el objeto a menor distancia
      min_depth_object = depth_ave;
      
      Spx(0,0) = xmin;
      Spx(1,0) = ymin;
      Spx(2,0) = xmax;
      Spx(3,0) = ymin;
      Spx(4,0) = xmin;
      Spx(5,0) = ymax;
      Spx(6,0) = xmax;
      Spx(7,0) = ymax;

                //orden de los puntos
      /* 2----3
         |    |
         0----1
      */

      Ox = (xmax +xmin) /2.0;  // puntos medios del objeto 
      Oy = (ymax +ymin) /2.0;
      width_real = xmax - xmin;
      height_real = ymax - ymin;

      if (estado == 1){
        Mcam = Mc;
      }
      else
      if(estado == 4)
      {
        Mcam = Mc2;
      }


      x_goal = Ox; // calculo el x deseado con el punto medio del recuadro que da yolo con menor distancia xmin+xmax/2 
      y_goal = Oy; // calculo el y deseado con el punto medio del recuadro que da yolo con menor distancia xmin+xmax/2 ;
      x_goal = (x_goal-Mcam(0,2))/Mcam(0,0) * min_depth_object;
      y_goal = (y_goal-Mcam(1,2))/Mcam(1,1) * min_depth_object;
      z_goal = sqrt(pow(min_depth_object,2)-pow(x_goal,2)-pow(y_goal,2)); // convierto la profundidad que era en linea recta desde la camara hasta el objeto , en una distancia en solo en el eje z
      //posXYZ_robot_save <<  0.0,0.0,0.0,1.0; // guardo la odometria del robot con respecto al mundo y el isntante que detecto el objeto

      Eigen::MatrixXf cTr(4,4); // matriz homogenea para convertir los datos xyz de la camara1 o camara 2 con respecto al robot
      xyz_object_camera << x_goal, y_goal, z_goal, 1.0;

      if (estado == 1 || estado == 2)
        {
        cTr << -0.0099 ,-0.2325 ,0.9725   ,tcb_1(0,0)   
              ,-0.9999 ,0.0061  ,-0.0087  ,tcb_1(1,0)
              ,-0.0039 ,-0.9726 ,-0.23268 ,tcb_1(2,0)
              ,0       ,0       ,0        ,1.0000;

        /*cTr << rcb_1(0,0) ,rcb_1(0,1) ,rcb_1(0,2) ,tcb_1(0,0)
              ,rcb_1(1,0) ,rcb_1(1,1) ,rcb_1(1,2) ,tcb_1(1,0)
              ,rcb_1(2,0) ,rcb_1(2,1) ,rcb_1(2,2) ,tcb_1(2,0)
              ,0     ,0        ,0                 ,1.0000;*/
        }
      else
      if (estado == 4 || estado == 3)
        { 
        cTr << rcb_2(0,0) ,rcb_2(0,1) ,rcb_2(0,2) ,tcb_2(0,0)
              ,rcb_2(1,0) ,rcb_2(1,1) ,rcb_2(1,2) ,tcb_2(1,0)
              ,rcb_2(2,0) ,rcb_2(2,1) ,rcb_2(2,2) ,tcb_2(2,0)
              ,0     ,0        ,0                 ,1.0000;
        } 
        xyz_object_robot = cTr * xyz_object_camera;
      std::cout<<"XYZ_camara:"<<x_goal<<", "<<y_goal<<", "<<z_goal<<std::endl;
    }
    

  }
  std::cout<<"Objeto-robot:"<<xyz_object_robot(0,0)<<", "<<xyz_object_robot(1,0)<<", "<<xyz_object_robot(2,0)<<std::endl;

  //////// get LiDAR odometry data
  posXYZ_robot << odom_msg->pose.pose.position.x, odom_msg->pose.pose.position.y, odom_msg->pose.pose.position.z, 1;
  //posXYZ_robot = posXYZ_robot - posXYZ_robot_save;

  tf::Quaternion quat(
        odom_msg->pose.pose.orientation.x,
        odom_msg->pose.pose.orientation.y,
        odom_msg->pose.pose.orientation.z,
        odom_msg->pose.pose.orientation.w); 

  double roll, pitch, yaw;
  tf::Matrix3x3(quat).getRPY(roll, pitch, yaw);

  phi = yaw ;//- phi_cine; // orientacion del robot
  std::cout<<"yaw: "<<yaw*180/PI<<std::endl;
  std::cout<<"phi_cine: "<<phi_cine*180/PI<<std::endl;
  std::cout<<"phi: "<<phi*180/PI<<std::endl;
  std::cout<<"----------------->ESTADO: "<<estado<<std::endl;
  std::cout<<"Pos robot: "<<posXYZ_robot<<std::endl;

  
  if (num_obj_yolo>0 && (estado==1 || estado == 4))//empieza el control
  {
    std::cout<<"Visual: "<<std::endl;
    
    if (estado == 1){
      Mcam = Mc;
      lambda << lambda_1;
      oxy = oxy_1;
      tcb = tcb_1;
      rcb = rcb_1;
      k_tanh = k_tanh1;
    }
    else
    if(estado == 4)
    {
      Mcam = Mc2;
      lambda << lambda_2;
      oxy = oxy_2;
      tcb = tcb_2;
      rcb = rcb_2;
      k_tanh = k_tanh2;
    }

    std::cout<<"----------------->num_obj_yolo: "<<num_obj_yolo<<std::endl;
  
  //=================================================Calculo de puntos deseados en pixeles para control servovisual======================================
    // puntos medios donde quiero posicionar el objeto deseados
    // Odx Ody zd parametros que se debe escoger desde afuera en un lauch
    
    int Odx = oxy(0); //   punto x deseado en pixeles
    int Ody = oxy(1); //   punto Y deseado en pixeles
    zd = 0;
    int cont_pix_d=0; // contador para sacar el promedio de la profundidad deseada

    // calculo de Z deseada (este z se calcula dependiendo de la profundidad al objeto que se quiere navegar, asuminedo que el objeto esta sobre un plano )
    for (int iy = Ody-5;iy<Ody+5; iy++)
      for (int ix = Odx-5;ix<Odx+5; ix++){

        if(depth_data(iy,ix)>0){
          zd += depth_data(iy,ix);
          cont_pix_d++;
        }
      }
     if(estado == 1)
      //calculo de la profundidad de la camara delantera 
      zd = ((zd/cont_pix_d)*200.0)/pow(2,16);
    else
     if(estado == 4){
      //calculo de profundidad de la camara trasera
      zd = ((zd/cont_pix_d))/1000;      
    }
    
    // condicion del valor de z deseado, si este valor no es valido (0 , negativo o inf) el valor de profundidad se valida
    if (zd<=0){
      if(estado == 1)
        zd = 1.55; // valor medido de la camara fonrtal al piso y donde quiero ubicar el objeto () produndidad
      else
       if(estado == 4)
      {
        zd = 0.5; // valor medido de la camara trasera al piso y donde quiero ubicar el objeto () produndidad
      }
    }

    zr = min_depth_object; // produndidad real donde esta el objeto

    float k = zr/zd; // relacion entre distancia real y la distancia deseada 

    uint width_des  = k * width_real;  // ancho deseado calculado con la relacion entre produnidad acuatl y deseada
    uint height_des = k * height_real; // ancho deseado calculado con la relacion entre produnidad acuatl y deseada
    std::cout<<"profundidad deseada: "<<zd<<std::endl;
    // calculo de los puntos deseados en pixeles dependiendo de la profundidad y el lugar donde quiero ubicar el objeto 
    Spxd(0,0) = Odx - (width_des /2.0);
    Spxd(1,0) = Ody - (height_des/2.0);
    Spxd(2,0) = Odx + (width_des /2.0);
    Spxd(3,0) = Ody - (height_des/2.0);
    Spxd(4,0) = Odx - (width_des /2.0);
    Spxd(5,0) = Ody + (height_des/2.0);
    Spxd(6,0) = Odx + (width_des /2.0);
    Spxd(7,0) = Ody + (height_des/2.0);



    // SOLO PARA CMPARAR CON EL OTRO METODO/////////////////////////////////////////////////////////////////////////////
    if(metodo_data(0,0) == 1){
      
      Spxd << spxd_data(0,0),spxd_data(1,0),spxd_data(2,0),spxd_data(3,0),spxd_data(4,0),spxd_data(5,0),spxd_data(6,0),spxd_data(7,0);
      float area_r = abs((Spx(2,0) - Spx(0,0)) *(Spx(7,0)-Spx(1,0))); 
      float aread_d = abs((spxd_data(2,0) - spxd_data(0,0)) *(spxd_data(7,0)-spxd_data(1,0))); 
      zr = sqrt(area_r);
      zd = sqrt(aread_d);

    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // puntos deseados dibujados
    cv::Point pt1d(Spxd(0,0), Spxd(1,0));    
    cv::Point pt2d(Spxd(2,0), Spxd(3,0));
    cv::Point pt3d(Spxd(4,0), Spxd(5,0));
    cv::Point pt4d(Spxd(6,0), Spxd(7,0));

    // puntos reales dibujados

    cv::Point pt1(Spx(0,0), Spx(1,0));    
    cv::Point pt2(Spx(2,0), Spx(3,0));
    cv::Point pt3(Spx(4,0), Spx(5,0));
    cv::Point pt4(Spx(6,0), Spx(7,0));

    cv::rectangle(img_servo_visual, pt1d,pt4d,cv::Scalar(255, 0, 0),2);    // recuadro del objeto deseado 
    cv::line(img_servo_visual, pt1d, pt1, cv::Scalar(0, 0, 255),1, cv::LINE_AA);
    cv::line(img_servo_visual, pt2d, pt2, cv::Scalar(0, 0, 255),1, cv::LINE_AA);
    cv::line(img_servo_visual, pt3d, pt3, cv::Scalar(0, 0, 255),1, cv::LINE_AA);
    cv::line(img_servo_visual, pt4d, pt4, cv::Scalar(0, 0, 255),1, cv::LINE_AA);
    //std::cout<<"plot image points : "<<pt1d<<std::endl;
  //==========================================================================================================================================


  // //===================================================== jacobiana imagen ================================================================

    float lx = Mcam(0,0);// focal de la camarax
    float ly = Mcam(1,1);// focal de la camaray

    Eigen::MatrixXf J_img(8,6); // jacobiana de imagen de 4 puntos de los objetos (2*4 x 6) matriz de medidas 8x6
    J_img << -(lx/zr) ,0.0    ,Spx(0,0)/zr  , (Spx(0,0)*Spx(1,0))/lx     , -((lx+Spx(0,0)*Spx(0,0))/lx),Spx(1,0)
            ,0.0    ,(-ly/zr) ,Spx(1,0)/zr  , (ly+ (Spx(1,0)*Spx(1,0)))/ly  ,-(Spx(0,0)*Spx(1,0))/ly   ,-Spx(0,0)

            ,-(lx/zr) ,0.0    ,Spx(2,0)/zr  , (Spx(2,0)*Spx(3,0))/lx     ,-((lx+Spx(2,0)*Spx(2,0))/lx) , Spx(3,0)
            ,0.0    ,(-ly/zr) ,Spx(3,0)/zr  , (ly+ Spx(3,0)*Spx(3,0))/ly  ,-(Spx(2,0)*Spx(3,0))/ly     ,-Spx(2,0)

            ,-(lx/zr) ,0.0    ,Spx(4,0)/zr  , (Spx(4,0)*Spx(5,0))/lx     ,-((lx+Spx(4,0)*Spx(4,0))/lx) , Spx(5,0)
            ,0.0    ,(-ly/zr) ,Spx(5,0)/zr  , (ly+ Spx(5,0)*Spx(5,0))/ly  ,-(Spx(4,0)*Spx(5,0))/ly     ,-Spx(4,0)

            ,-(lx/zr) ,0.0    ,Spx(6,0)/zr  , (Spx(6,0)*Spx(7,0))/lx     ,-((lx+Spx(6,0)*Spx(6,0))/lx) , Spx(7,0)
            ,0.0    ,(-ly/zr) ,Spx(7,0)/zr  , (ly+ Spx(7,0)*Spx(7,0))/ly  ,-(Spx(6,0)*Spx(7,0))/ly     ,-Spx(6,0);         

    //==========================================================================================================================================
     
    //======================================================jacobiana BLue para servovisual=====================================================
    Eigen::MatrixXf J_blue(6,6); // jacobiana de blue     
    if  (estado == 1)
    {
      if(inicial_phi_est1){
        phi_est1 = phi; 
        inicial_phi_est1 = 0;
      }
    J_blue <<cos(phi-phi_est1)  ,-d * sin(phi-phi_est1) ,0 ,0 ,0 ,0 
            ,sin(phi-phi_est1)  , d * cos(phi-phi_est1)  ,0 ,0 ,0 ,0  
            ,0                  ,0                      ,0 ,0 ,0 ,0
            ,0                  ,0                      ,0 ,0 ,0 ,0
            ,0                  ,0                      ,0 ,0 ,0 ,0
            ,0                  ,0                      ,0 ,0 ,0 ,0;            
    }
    else
     if(estado == 4)
    {

      if(inicial_phi_est4){
        phi_est4 = phi; 
        inicial_phi_est4 = 0;
      }

    J_blue <<cos(phi-phi_est4)  ,  d * sin(phi-phi_est4) ,0 ,0 ,0 ,0 
            ,sin(phi-phi_est4)  , -d * cos(phi-phi_est4) ,0 ,0 ,0 ,0  
            ,0                  ,0                       ,0 ,0 ,0 ,0
            ,0                  ,0                       ,0 ,0 ,0 ,0
            ,0                  ,0                       ,0 ,0 ,0 ,0
            ,0                  ,0                       ,0 ,0 ,0 ,0; 

    }
   // std::cout<<"Jacobiana blue: "<<J_blue<<std::endl;
    /////////////////////// Matriz de transforamciones para servovisual con BLUE
    Eigen::MatrixXf Tcr(3,1); // elemtnos de traslacion de la matriz RTcr
    Eigen::MatrixXf Rcr(3,3); // elementos de rotacion de la matriz RTcr
    Tcr << tcb(0,0) ,tcb(1,0) ,tcb(2,0);
    Rcr << rcb(0,0) ,rcb(0,1) ,rcb(0,2)
          ,rcb(1,0) ,rcb(1,1) ,rcb(1,2) 
          ,rcb(2,0) ,rcb(2,1) ,rcb(2,2);
            
    //std::cout<<"Tcb:"<<Tcr<<std::endl;
    //std::cout<<"Rcb:"<<Rcr<<std::endl;

    Eigen::MatrixXf ntc(3,3); //skew symmetric matrix traslation ntc = [0 -z y; z 0 -x; -y x 0] uso la traslacion de RTcb 

    ntc <<  0         ,-Tcr(2,0) , Tcr(1,0)
          , Tcr(2,0) , 0         ,-Tcr(0,0)
          ,-Tcr(1,0) , Tcr(0,0)  , 0      ;
          
    Eigen::MatrixXf nR_Tc(6,6);
    nR_Tc = Rcr.cwiseProduct(ntc);

    Eigen::MatrixXf T_n(6,6);
    T_n << Rcr(0,0) ,Rcr(0,1) ,Rcr(0,2) ,nR_Tc(0,0) ,nR_Tc(0,1) ,nR_Tc(0,2) 
          ,Rcr(1,0) ,Rcr(1,1) ,Rcr(1,2) ,nR_Tc(1,0) ,nR_Tc(1,1) ,nR_Tc(1,2)
          ,Rcr(2,0) ,Rcr(2,1) ,Rcr(2,2) ,nR_Tc(2,0) ,nR_Tc(2,1) ,nR_Tc(2,2)
          ,0.0      ,0.0      ,0.0      ,Rcr(0,0)   ,Rcr(0,1)   ,Rcr(0,2)
          ,0.0      ,0.0      ,0.0      ,Rcr(1,0)   ,Rcr(1,1)   ,Rcr(1,2) 
          ,0.0      ,0.0      ,0.0      ,Rcr(2,0)   ,Rcr(2,1)   ,Rcr(2,2);
    
    Eigen::MatrixXf Jcam_robot(6,6);
    Jcam_robot = J_img * T_n * J_blue;

    Eigen::MatrixXf pin_Jimg = J_img.completeOrthogonalDecomposition().pseudoInverse();
    Eigen::MatrixXf pin_Jc_r = Jcam_robot.completeOrthogonalDecomposition().pseudoInverse();  

    q_vel_cam = - pin_Jimg * (Spx - Spxd); // Velocidades de la camara 

    ///////////////////////////LEY DE CONTROL SERVO VISUAL + CINEAMTICO
    q_vel =  - pin_Jc_r * (Spx - Spxd);  //calculo de velocidades de salida que van al robot 
    q_vel << q_vel(0,0)* lambda(0), q_vel(1,0)* lambda(1) ,q_vel(2,0)* lambda(2),
             q_vel(3,0)* lambda(3), q_vel(4,0)* lambda(4) ,q_vel(5,0)* lambda(5); // ganancias apra velocidades de salida

    u_vel =q_vel(0,0); // asignacion del valor de velocidad de salida para el robot 
    psi = q_vel(1,0);

    error_control_x = ((Odx-Ox)-Mcam(0,2))/Mcam(0,0) * (zd-zr);
    error_control_y = ((Ody-Oy)-Mcam(1,2))/Mcam(1,1) * (zd-zr);
    ini_cine = 1;
    cinebool = 0;
    
  }

  else
  
  {if(1)
  {
    std::cout<<"Cinematico: "<<std::endl;
    if(ini_cine){
     ini_cine=0;
     posXYZ_robot_save <<  odom_msg->pose.pose.position.x, odom_msg->pose.pose.position.y, odom_msg->pose.pose.position.z, 1; // guardo la odometria del robot con respecto al mundo y el isntante que detecto el objeto
     phi_cine = yaw;
    }
    cinebool = 1;
    /*if (inicial_phi_est2 && estado == 2){
      inicial_phi_est2 = 0;
      phi_cine = phi;
    }*/
    
    phi_cine = phi - phi_cine;
    Eigen::MatrixXf J_blue_cine(2,2); // jacobiana de blue solo para el control cinematico
    Eigen::MatrixXf gain_c(2,1);
   
    //float phi_dif = phi - phi_cine; // angulo inicial que entra al control cinematico
    switch (estado){
      case 1: //para visual servo frontal
        J_blue_cine << cos(phi_cine) ,-0.01 * sin(phi_cine)
                      ,sin(phi_cine) , 0.01 * cos(phi_cine);  
        gain_c << lambda_c (0), lambda_c (1); 
        X_cine_d = X_desi_estado(0);
        Y_cine_d = Y_desi_estado(0);
      break;            
      case 2: //para visual servo frontal
        J_blue_cine << cos(phi_cine) ,-0.01 * sin(phi_cine)
                      ,sin(phi_cine) , 0.01 * cos(phi_cine);  
        gain_c << lambda_c (2), lambda_c (3); 
        X_cine_d = X_desi_estado(1);
        Y_cine_d = Y_desi_estado(1);
      break;
      case 3: //para visual servo frontal
        J_blue_cine << cos(phi_cine) , 0.01 * sin(phi_cine)
                      ,sin(phi_cine) ,-0.01 * cos(phi_cine);
        gain_c << lambda_c (4), lambda_c (5);   
        X_cine_d = X_desi_estado(2);
        Y_cine_d = Y_desi_estado(2);
      break;
      case 4: //para visual servo frontal
        J_blue_cine << cos(phi_cine) , 0.01 * sin(phi_cine)
                      ,sin(phi_cine) ,-0.01 * cos(phi_cine);
        gain_c << lambda_c (6), lambda_c (7);   
        X_cine_d = X_desi_estado(3);
        Y_cine_d = Y_desi_estado(3);
      break;
    }       

   
    
    error_xyz << xyz_object_robot-(posXYZ_robot - posXYZ_robot_save);  
    //error_xyz << posXYZ_robot;  
    error_xy  << error_xyz(0,0)-X_cine_d ,error_xyz(1,0)-Y_cine_d;
    std::cout<<"Obj: "<<xyz_object_robot<<std::endl;   
    std::cout<<"ErroresXYZ: "<<error_xyz<<std::endl;   
    std::cout<<"Errores: "<<error_xy<<std::endl;   
    std::cout<<"Ganancias: "<<gain_c<<std::endl;  
    
    q_vel_cine = J_blue_cine.inverse() * (error_xy); 
    u_vel = q_vel_cine(0,0)*gain_c(0);
    psi   = q_vel_cine(1,0)*gain_c(1);

    error_control_x = error_xy(0,0);
    error_control_y = error_xy(1,0);
  }
  }
 // mutex_lock.unlock();

 std::cout<<"Vel Angular psi: "<<psi<<std::endl;   

  u_vel = u_max * tanh(k_tanh(0,0)*u_vel); // saturacion de la velocidad lineal del robot blue 

  if(!(u_vel<0.001 && u_vel>-0.001)){ // condicion para evitar que la division no tienda al infinito
    psi = psi_max* (tanh(k_tanh(0,1)*atan((psi*dist_a)/u_vel)));   // angulo de stering calculado para el blue
  }
  else
  {
    psi = psi_ant;
  }
  psi_ant =psi;

  psi = psi*180.0/PI;
  ////// condiciones para no entrar en el rango de velocidades donde el robot no se mueve -0.09m/s -> 0.09m/s

  
  float u_vel_SF = u_vel; // velocidad lineal sin el punot muerto, uso para las graficas

  if(u_vel < 0.09 && u_vel > 0.009)
    u_vel =0.09;
  if(u_vel > -0.09 && u_vel < -0.009)
    u_vel =-0.09;
   ///////////////////////////////////////////////////////////////////////////////////////////////////////////

  
  if (estado==1)
    camBool = 1;
  else
  if (estado==4)
  {
    camBool = 0;    
  }
  else
    camBool = 2;

  sel_cam.switch_CamTopicName.data= camBool;
  image_msg = cv_bridge::CvImage(std_msgs::Header(), "rgb8", img_servo_visual).toImageMsg();
  image_msg->header.stamp     = yoloImg->header.stamp;
  
  min_depth_object=1000.0; // reset distance objetcs

  
  //if (init_control)
  //{
  u_vel = (1-(u_vel-u_ant))*u_vel; // calculo de velocidades en rampa para evitar cambios bruscos
  u_ant = u_vel; // guardo la velocidad actual que tenia en el control en esta variable
   // ackermann_state_rad.drive.steering_angle = psi;
   // ackermann_state_rad.drive.speed = u_vel;
 // }
 // else
 // {
    //ackermann_state_rad.drive.steering_angle = 0.0;
  //  ackermann_state_rad.drive.speed = 0.0;
 //   u_vel = u_ant* (1-0.30); // frenado de velocidad, en cada itreracion se disminuye un 30% de la velocidad anterior
   
 // }
  float err_min_x=0, err_max_x=0, err_min_y=0, err_max_y=0;

   if(starting && estado>0) // verifico si se ha inicializado el proceso y tambien si el estado es mayor de 0
  {
    switch (estado){
    case 1: //errores permitidos para terminar el primer estado
      err_min_x = err_X(0,0);
      err_max_x = err_X(1,0);
      err_min_y = err_Y(0,0);
      err_max_y = err_Y(1,0);
    break;            
    case 2: //errores permitidos para terminar el primer estado
    
      err_min_x = err_X(0,1);
      err_max_x = err_X(1,1);
      err_min_y = err_Y(0,1);
      err_max_y = err_Y(1,1);
    break;
    case 3: //errores permitidos para terminar el primer estado
    
      err_min_x = err_X(0,2);
      err_max_x = err_X(1,2);
      err_min_y = err_Y(0,2);
      err_max_y = err_Y(1,2); 
    break;
    case 4: //errores permitidos para terminar el primer estado
      err_min_x = err_X(0,3);
      err_max_x = err_X(1,3);
      err_min_y = err_Y(0,3);
      err_max_y = err_Y(1,3);   
    break;
    }     

    std::cout<<"Errores control: "<<error_control_x<<", "<<error_control_y<<std::endl;      

   /* if(error_control_x > err_min_x && error_control_x < err_max_x ) // verificamos si esta en el rango de X deseado
      if(error_control_y > err_min_y && error_control_y < err_max_y ) // verificamos si esta en el rango de Y deseado
        estado++;
    if (estado >4)
      estado = 4;*/
  } 

  //ackermann_state_rad.header.stamp = odom_msg->header.stamp;
  //ackermann_state_rad.header.stamp = ros::Time::now();
  //ackermann_state_rad.drive.steering_angle = psi;
  //ackermann_state_rad.drive.speed = u_vel;  
  //ackermann_state_rad.drive.steering_angle = 0.0;
  //ackermann_state_rad.drive.speed = 0.0;

  std::cout<<"Coordenadas (u_vel, steering, ang): "<<u_vel<<", "<<psi<<std::endl;
  
  
  //////////// Envio de resultados de pixeles origen- destino, velocidades al robot, 

  odom.header.stamp =  odom_msg->header.stamp;
  odom.header.frame_id = "VS_results";

  odom.twist.twist.linear.x = u_vel*100;
  odom.twist.twist.linear.y = psi;   

  odom.pose.pose.position.x =    q_vel(0,0);
  odom.pose.pose.position.y =    q_vel(1,0);
  odom.pose.pose.position.z =    q_vel(2,0);
  odom.pose.pose.orientation.x = q_vel(3,0);
  odom.pose.pose.orientation.y = q_vel(4,0);
  odom.pose.pose.orientation.z = q_vel(5,0);

  // NO ES LA COOVARIANZA QUE GUARDO SI NO LOS PIXELES DEL BOUNDING BOX DEL OBJETO 
  // CURRENT BOUNDIG BOX
  odom.pose.covariance[0]  = Spx (0,0);
  odom.pose.covariance[1]  = Spx (1,0);
  odom.pose.covariance[2]  = Spx (2,0);
  odom.pose.covariance[3]  = Spx (3,0);
  odom.pose.covariance[4]  = Spx (4,0);
  odom.pose.covariance[5]  = Spx (5,0);
  odom.pose.covariance[6]  = Spx (6,0);
  odom.pose.covariance[7]  = Spx (7,0);
  //DESIRED BOUNDIG BOX
  odom.pose.covariance[8]  = Spxd(0,0);
  odom.pose.covariance[9]  = Spxd(1,0);
  odom.pose.covariance[10] = Spxd(2,0);
  odom.pose.covariance[11] = Spxd(3,0);
  odom.pose.covariance[12] = Spxd(4,0);
  odom.pose.covariance[13] = Spxd(5,0);
  odom.pose.covariance[14] = Spxd(6,0);
  odom.pose.covariance[15] = Spxd(7,0);

  //Verificar si esta el control servo visual o no
  float control_float = cinebool?1.0f:0.0f;  // variable en flotante que dice si esta el control inicializado
  odom.pose.covariance[16] = control_float;
  odom.pose.covariance[17] = u_vel; // velocidad lineal deseada
  odom.pose.covariance[18] = psi; // angulo blue deseado
  odom.pose.covariance[19] = zd; // profundidad deseada
  odom.pose.covariance[20] = zr; // profundidad real

  odom.pose.covariance[21] = estado; // estado del proceso
  odom.pose.covariance[22] = err_min_x; // error minimo en x aceptado 
  odom.pose.covariance[23] = err_max_x; // error maximo en x aceptado 
  odom.pose.covariance[24] = err_min_y; // error minimo en y aceptado 
  odom.pose.covariance[25] = err_max_y; // error maximo en y aceptado 
  odom.pose.covariance[26] = error_control_x; // error de posicion real en x
  odom.pose.covariance[27] = error_control_y; // error de posicion real en y
  odom.pose.covariance[28] = posXYZ_robot(0,0); // error de posicion real en y
  odom.pose.covariance[29] = posXYZ_robot(1,0); // error de posicion real en y
  odom.pose.covariance[30] = posXYZ_robot_save(0,0); // error de posicion real en y
  odom.pose.covariance[31] = posXYZ_robot_save(1,0); // error de posicion real en y
  odom.pose.covariance[32] = xyz_object_robot(0,0); // error de posicion real en y
  odom.pose.covariance[33] = xyz_object_robot(1,0); // error de posicion real en y
  odom.pose.covariance[34] = yaw;
  odom.pose.covariance[35] = phi_cine;

  odom.twist.covariance[0]  = q_vel_cam (0,0);
  odom.twist.covariance[1]  = q_vel_cam (1,0);
  odom.twist.covariance[2]  = q_vel_cam (2,0);
  odom.twist.covariance[3]  = q_vel_cam (3,0);
  odom.twist.covariance[4]  = q_vel_cam (4,0);
  odom.twist.covariance[5]  = q_vel_cam (5,0);
  odom.twist.covariance[6]  = X_cine_d;
  odom.twist.covariance[7]  = Y_cine_d;
  odom.twist.covariance[8]  = error_xy(0,0);
  odom.twist.covariance[9]  = error_xy(1,0);
  odom.twist.covariance[10] = xyz_object_camera(0,0);
  odom.twist.covariance[11] = xyz_object_camera(1,0);
  odom.twist.covariance[12] = xyz_object_camera(2,0);
  odom.twist.covariance[13] = u_vel_SF;
  

  // Viusalizacion del objeto en RVIZ
 
  marker.header.frame_id = "base_link";
  marker.header.stamp = ros::Time();
  marker.ns = "my_namespace";
  marker.id = 0;
  marker.type = visualization_msgs::Marker::CYLINDER;
  marker.action = visualization_msgs::Marker::ADD;
  marker.pose.position.x = xyz_object_robot(0,0);
  marker.pose.position.y = xyz_object_robot(1,0);
  marker.pose.position.z = xyz_object_robot(2,0);
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;
  marker.scale.x = 0.2;
  marker.scale.y = 0.2;
  marker.scale.z = 0.2;
  marker.color.a = 1.0; // Don't forget to set the alpha!
  marker.color.r = 0.0;
  marker.color.g = 1.0;
  marker.color.b = 0.0;
  vis_pub.publish( marker );
  //only if using a MESH_RESOURCE marker type:
 // marker.mesh_resource = "package://pr2_description/meshes/base_v0/base.dae";
  

  // Publishers    

  //ackermann_rad_publisher.publish(ackermann_state_rad);

  /////////////////////////// imprimo tiempos
  auto t2= Clock::now();
  float time = std::chrono::duration_cast<std::chrono::nanoseconds>(t2-t1).count()/1000000000.0;
  std::cout<<"Tiempo: "<<time<<std::endl;

  init_control = true ; 
  std::cout<<"init_control callback: "<<init_control<<std::endl;
  
  //rate.sleep();

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

}


int main(int argc, char** argv)
{

  ros::init(argc, argv, "VisualServoYolo");
  ros::NodeHandle nh;  
  
  /// Load Parameters
  nh.getParam("/detection_BoundingBoxes", objYoloTopic);
  nh.getParam("/depth_lidar_image", objDepthTopic1);
  nh.getParam("/depth_cam2_image" , objDepthTopic2);
  nh.getParam("/yoloImg_input" , yoloImgTopic);
  nh.getParam("/odom" , odometryTopic);
  
  XmlRpc::XmlRpcValue param;

  nh.getParam("/control_params/lambda_1", param);
  // ganacias del control servo visual 1
  lambda_1 <<  (double)param[0], (double)param[1], (double)param[2], (double)param[3], (double)param[4], (double)param[5];
  nh.getParam("/control_params/lambda_2", param);
  // ganacias del control servo visual 2
  lambda_2 <<  (double)param[0], (double)param[1], (double)param[2], (double)param[3], (double)param[4], (double)param[5];
  nh.getParam("/control_params/lambda_c", param);
  // ganacias del control cineamtico
  lambda_c <<  (double)param[0], (double)param[1], (double)param[2], (double)param[3], (double)param[4], (double)param[5] ,(double)param[6] ,(double)param[7];
  nh.getParam("/control_params/oxy_1", param);
  // Ubicacion del centro del boundung box en pixeles en la imagen de la camara 1 en donde se desea posicionar el objeto 
  oxy_1 <<  (double)param[0], (double)param[1];
  nh.getParam("/control_params/oxy_2", param);
  // Ubicacion del centro del boundung box en pixeles en la imagen de la camara 2 en donde se desea posicionar el objeto 
  oxy_2 <<  (double)param[0], (double)param[1];
  nh.getParam("/control_params/select_cam", param);
  select_cam << (double)param[0]; // parametro usado unicamente para seleccionar el tipo de control en las pruebas
  nh.getParam("/control_params/k_tanh1", param);
  // ganancias de las velocidades de salida para el movimiento del robot con el control servo visual hacia adelante 
  k_tanh1 <<  (double)param[0], (double)param[1];
  nh.getParam("/control_params/k_tanh2", param);
  // ganancias de las velocidades de salida para el movimiento del robot con el control servo visual hacia adelante 
  k_tanh2 <<  (double)param[0], (double)param[1];
  nh.getParam("/control_params/err_X", param);
  // errores permitidos en X para cambiar estado (columna el estado, fila error minimo, error maximo)
  err_X <<  (double)param[0], (double)param[1], (double)param[2], (double)param[3],
            (double)param[4], (double)param[5], (double)param[6], (double)param[7];
  nh.getParam("/control_params/err_Y", param);
  // errores permitidos en Y para cambiar estado (columna el estado, fila error minimo, error maximo)
  err_Y <<  (double)param[0], (double)param[1], (double)param[2], (double)param[3],
            (double)param[4], (double)param[5], (double)param[6], (double)param[7];
   nh.getParam("/control_params/X_desi_estado", param);
  // errores permitidos en Y para cambiar estado (columna el estado, fila error minimo, error maximo)
  X_desi_estado <<  (double)param[0], (double)param[1], (double)param[2], (double)param[3];
  nh.getParam("/control_params/Y_desi_estado", param);
  // errores permitidos en Y para cambiar estado (columna el estado, fila error minimo, error maximo)
  Y_desi_estado <<  (double)param[0], (double)param[1], (double)param[2], (double)param[3];
  nh.getParam("/control_params/camera1_matrix", param);
  // matriz de calibracion de parametros intrinsecos de la camara 1
  Mc  <<  (double)param[0] ,(double)param[1] ,(double)param[2] ,(double)param[3]
         ,(double)param[4] ,(double)param[5] ,(double)param[6] ,(double)param[7]
         ,(double)param[8] ,(double)param[9] ,(double)param[10],(double)param[11];
  
  nh.getParam("/control_params/camera2_matrix", param);
  // matriz de calibracion de parametros intrinsecos de la camara 2
  Mc2 <<  (double)param[0] ,(double)param[1] ,(double)param[2] ,(double)param[3]
         ,(double)param[4] ,(double)param[5] ,(double)param[6] ,(double)param[7]
         ,(double)param[8] ,(double)param[9] ,(double)param[10],(double)param[11];



  // Solo para comparar con el segundo metodo
  nh.getParam("/control_params/spxd", param);
  // spxd data
  spxd_data <<  (double)param[0] ,(double)param[1] ,(double)param[2] ,(double)param[3]
               ,(double)param[4] ,(double)param[5] ,(double)param[6] ,(double)param[7];


  nh.getParam("/control_params/metodo_data", param);
  // metodo data
  metodo_data <<  (double)param[0];

  nh.getParam("/control_params/bb_perce", param);
  // metodo data
  bb_perce <<  (double)param[0];
         
         

  /*nh.getParam("/control_params/tlc_1", param);
  // traslacion de1 lidar con respecto a la camara 1
  Tlc_1 <<  (double)param[0]
           ,(double)param[1]
           ,(double)param[2];

  nh.getParam("/control_params/rlc_1", param);
  // rotacion de1 lidar con respecto a la camara 1
  Rlc_1 <<  (double)param[0] ,(double)param[1] ,(double)param[2]
           ,(double)param[3] ,(double)param[4] ,(double)param[5]
           ,(double)param[6] ,(double)param[7] ,(double)param[8];

  nh.getParam("/control_params/tlc_2", param);
  // rotacion de1 lidar con respecto a la camara 1
  Tlc_2 <<  (double)param[0]
           ,(double)param[1]
           ,(double)param[2];

  nh.getParam("/control_params/rlc_2", param);
  Rlc_2 <<  (double)param[0] ,(double)param[1] ,(double)param[2]
           ,(double)param[3] ,(double)param[4] ,(double)param[5]
           ,(double)param[6] ,(double)param[7] ,(double)param[8];

  nh.getParam("/control_params/bTv", param); // traslacion  base to lidar
  bTv <<  (double)param[0] ,(double)param[1] ,(double)param[2] ,(double)param[3]
         ,(double)param[4] ,(double)param[5] ,(double)param[6] ,(double)param[7]
         ,(double)param[8] ,(double)param[9] ,(double)param[10],(double)param[11]
         ,(double)param[12] ,(double)param[13] ,(double)param[14],(double)param[15];*/


  nh.getParam("/control_params/tcb_1", param);
  // traslacion de1 lidar con respecto a la camara 1
  tcb_1 <<  (double)param[0] ,(double)param[1] ,(double)param[2];

  nh.getParam("/control_params/rcb_1", param);
  // rotacion de1 lidar con respecto a la camara 1
  rcb_1 <<  (double)param[0] ,(double)param[1] ,(double)param[2]
           ,(double)param[3] ,(double)param[4] ,(double)param[5]
           ,(double)param[6] ,(double)param[7] ,(double)param[8];

  nh.getParam("/control_params/tcb_2", param);
  // rotacion de1 lidar con respecto a la camara 1
  tcb_2 <<  (double)param[0] ,(double)param[1] ,(double)param[2];

  nh.getParam("/control_params/rcb_2", param);
  rcb_2 <<  (double)param[0] ,(double)param[1] ,(double)param[2]
           ,(double)param[3] ,(double)param[4] ,(double)param[5]
           ,(double)param[6] ,(double)param[7] ,(double)param[8];


  //mutex_lock.lock();

  message_filters::Subscriber<detection_msgs::BoundingBoxes> yoloBbox_sub(nh, objYoloTopic , 10);
  message_filters::Subscriber<Image>  depthImg_sub_1(nh, objDepthTopic1, 10);
  message_filters::Subscriber<Image>  depthImg_sub_2(nh, objDepthTopic2, 10);
  message_filters::Subscriber<Image>  yoloInimg_sub(nh, yoloImgTopic, 10);
  message_filters::Subscriber<nav_msgs::Odometry>  odom_input(nh, odometryTopic, 10); // suscriptor odometria

  typedef sync_policies::ApproximateTime<detection_msgs::BoundingBoxes, Image, Image, Image, nav_msgs::Odometry> MySyncPolicy;
  Synchronizer<MySyncPolicy> sync(MySyncPolicy(500), yoloBbox_sub, depthImg_sub_1, depthImg_sub_2, yoloInimg_sub, odom_input);
  sync.registerCallback(boost::bind(&callback, _1, _2, _3, _4, _5));
  /*typedef sync_policies::ApproximateTime<detection_msgs::BoundingBoxes, Image, Image, nav_msgs::Odometry> MySyncPolicy;
  Synchronizer<MySyncPolicy> sync(MySyncPolicy(500), yoloBbox_sub, depthImg_sub_1, depthImg_sub_2, odom_input);
  sync.registerCallback(boost::bind(&callback, _1, _2, _3, _4));*/

  //mutex_lock.unlock();

  pub_switchCam = nh.advertise<yolo_visual_servoing::selectImg>("/bool_cam_control", 10);
  pub_rgb_yolo = nh.advertise<sensor_msgs::Image>("/yoloImg_distances", 10);
  odom_pub = nh.advertise<nav_msgs::Odometry>("/vs_results", 10);
    // topics para BLUE

  ackermann_rad_publisher = nh.advertise <ackermann_msgs::AckermannDriveStamped> ("/desired_ackermann_state", 1);
  ford_rec_publisher = nh.advertise <std_msgs::Float32>("/forward_recommended_velocity" , 1);
  back_rec_publisher = nh.advertise <std_msgs::Float32>("/backward_recommended_velocity", 1);
  vis_pub = nh.advertise<visualization_msgs::Marker>( "visualization_marker", 1 );
  ros::Rate loopRate(10.4);

  ford_rec.data = 1000;
  back_rec.data = 1000;
  image_msg = cv_bridge::CvImage(std_msgs::Header(), "rgb8", cv::Mat::zeros(720, 1280, CV_64FC1)).toImageMsg(); 
  sel_cam.switch_CamTopicName.data= camBool;
  while(ros::ok()){
    
    init_control = false;
    std::cout<<"init_control antes: "<<init_control<<std::endl;
    ros::spinOnce();
    std::cout<<"init_control despues: "<<init_control<<std::endl;
    //u_vel = (1-(u_vel-u_ant))*u_vel; // calculo de velocidades en rampa para evitar cambios bruscos
    ackermann_state_rad.header.stamp = ros::Time::now();
    if(!init_control)
    {
      u_vel = u_ant* (1-0.30);
      //ackermann_state_rad.drive.steering_angle = 0.0;
     // ackermann_state_rad.drive.speed = 0.0;
      if (u_vel > -0.01 && u_vel <0.01)
        u_vel = 0.0;
 
    }
    ackermann_state_rad.drive.steering_angle = psi;
    ackermann_state_rad.drive.speed = u_vel;

    u_ant = u_vel;
    ackermann_rad_publisher.publish(ackermann_state_rad);
    ford_rec_publisher.publish(ford_rec);
    back_rec_publisher.publish(back_rec);

    
    pub_switchCam.publish(sel_cam); // topic to select the yolo input image nas dpeth input image
    if(init_control)
      pub_rgb_yolo.publish(image_msg); // publish image of yolo whit distance 
    odom_pub.publish(odom); // publico velocidades para revisar en graficaq si sirve o no 
    
    loopRate.sleep();
    
  } 
  
  return 0;
}
