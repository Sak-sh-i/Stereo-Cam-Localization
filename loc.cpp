#include <ros/ros.h>
#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <boost/foreach.hpp>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "ceres/ceres.h"
#include "glog/logging.h"
#include <sstream>

static const std::string OPENCV_WINDOW = "Depth Image";
cv_bridge::CvImagePtr cv_ptr;

pcl::PointCloud<pcl::PointXYZ> pcin;
pcl::PointCloud<pcl::PointXYZ> pcincam;
pcl::PointCloud<pcl::PointXYZ> pcfinal;

tf::TransformListener listener;

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

tf::Transform get_tf_from_stamped_tf(tf::StampedTransform sTf)
{
	tf::Transform tf(sTf.getBasis(), sTf.getOrigin());
	return tf;
}

bool multiply_stamped_tfs(tf::StampedTransform A_stf, tf::StampedTransform B_stf, tf::StampedTransform &C_stf)
{
	tf::Transform A, B, C;

	std::string str1(A_stf.child_frame_id_);
	std::string str2(B_stf.frame_id_);

	A = get_tf_from_stamped_tf(A_stf);
	B = get_tf_from_stamped_tf(B_stf);

	C = A * B;

	C_stf.frame_id_ = A_stf.frame_id_;
	C_stf.child_frame_id_ = B_stf.child_frame_id_;

	C_stf.setOrigin(C.getOrigin());
	C_stf.setBasis(C.getBasis());

	C_stf.stamp_ = ros::Time::now();

	return true;
}

struct ExponentialResidual
{
	ExponentialResidual(pcl::PointCloud<pcl::PointXYZ> pcin) : pcin_(pcin)
	{
	
	}

	template <typename T> bool operator()(const T* const Tksi, T* residual) const
	{
		residual[0] = (get_tf_from_stamped_tf(Tksi[0])*T(pcin_))[2]-cv_ptr->image(	get_tf_from_stamped_tf(Tksi[0])*T(pcin_)[0],get_tf_from_stamped_tf(Tksi[0])*T(pcin_)[1]		);
		return true;
	}

	private:
	// Observations for a sample.
	pcl::PointCloud<pcl::PointXYZ> pcin_;
};

class ImageConverter
{
	ros::NodeHandle n;
	
	image_transport::ImageTransport it_;
	image_transport::Subscriber image_sub;

		public:
		ImageConverter() : it_(n)
		{
			image_sub = it_.subscribe("/depth_image", 1, &ImageConverter::imageCb, this);

			cv::namedWindow(OPENCV_WINDOW);
		}

		~ImageConverter()
		{
			cv::destroyWindow(OPENCV_WINDOW);
		}

		void imageCb(const sensor_msgs::ImageConstPtr& input)
		{
			try
			{
				cv_ptr = cv_bridge::toCvCopy(input, sensor_msgs::image_encodings::MONO8);
			}
			catch (cv_bridge::Exception& e)
			{
				ROS_ERROR("cv_bridge exception: %s", e.what());
				return;
			}

			cv::imshow(OPENCV_WINDOW, cv_ptr->image);
			cv::waitKey(3);
		}
};

void velodyneCallback(const sensor_msgs::PointCloud2ConstPtr& input)
{
	//pcin=*input;
	pcl::fromROSMsg(*input,pcin); 	
	return;
}
int main(int argc, char **argv)
{
	google::InitGoogleLogging(argv[0]);
	
	ros::init(argc, argv, "loc");
		
	ros::NodeHandle n;
	
	ros::Publisher location_pub=n.advertise<tf::StampedTransform>("/location",1800);
	ros::Subscriber velodyne_sub=n.subscribe<sensor_msgs::PointCloud2>("/velodyne_points",1000,velodyneCallback);
	
	//tf::TransformListener listener;
	
	ImageConverter ic;
	
	//use cv_ptr->image to access the grayscale image now
	
	ros::Rate loop_rate(10);
	
	while(ros::ok())
	{
		//static tf::TransformBroadcaster br;
		tf::StampedTransform transform;										//tracking transform
	
		try
		{
			listener.lookupTransform("/tracking", "/tracking", ros::Time(0), transform);
		}
		
		catch (tf::TransformException &ex)
		{
			ROS_ERROR("%s",ex.what());
			ros::Duration(1.0).sleep();
			continue;
		}
		
		tf::Transform input;
		input.setIdentity();
		
		tf::StampedTransform tcm;
		
		tcm.frame_id_="camera";
		tcm.child_frame_id_="pointcloud";
		tcm.stamp_=ros::Time::now();
		tcm.setData(input);
		
		pcl_ros::transformPointCloud(pcin,pcincam,get_tf_from_stamped_tf(tcm));
		pcl_ros::transformPointCloud(pcincam,pcfinal,get_tf_from_stamped_tf(transform));

		//br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "lidar"));
		
		/*ss<<x<<" "<<y<<" "<<z;
		q.data=ss.str();
		ROS_INFO("%s",msg.data.c_str());
		
		location_pub.publish(q);*/
		
		ros::spinOnce();
		
		loop_rate.sleep();
	}
	
	return 0;
}
