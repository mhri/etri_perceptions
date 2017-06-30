#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/time_synchronizer.h>

#include <cv_bridge/cv_bridge.h>

//pcl
#include <pcl/conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>

#include <Eigen/Core>
#include <pcl_conversions/pcl_conversions.h>

#include <tf/transform_broadcaster.h>
//#include <face_detector_pointcloud/FaceDetectResult.h>
#include <geometry_msgs/Point32.h>
#include <std_srvs/Empty.h>

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <boost/filesystem.hpp>

#include <perception_msgs/PersonPerceptArray.h>
#include <perception_msgs/PersonPercept.h>


using namespace std;

class Face3DPositionEstimator
{
public:
    Face3DPositionEstimator()
    {
        percepts_sub_ =  new message_filters::Subscriber<perception_msgs::PersonPerceptArray>(nh_, "Tracked_People", 2);
        pointcloud_sub_ = new message_filters::Subscriber<sensor_msgs::PointCloud2>(nh_, "PointCloud2", 2);

        sync_ = new message_filters::Synchronizer<MySyncPolicy>(MySyncPolicy(10), *percepts_sub_, *pointcloud_sub_);
        sync_->registerCallback(boost::bind(&Face3DPositionEstimator::callback, this, _1, _2));

        // Message Publisher
        publisher = nh_.advertise<perception_msgs::PersonPerceptArray>("/mhri/perception_core/3d_pos_estimation/persons", 2);

				// 필요시 아래 코멘트를 열어서 PointCloud를 배포하여 영상으로 정보를 확인
				// pcl_pub = nh_.advertise<sensor_msgs::PointCloud2> ("/mhri/points2", 1);

        ROS_INFO("Initialized");
    }

    ~Face3DPositionEstimator()
    {
        delete sync_;
        delete pointcloud_sub_;
        delete percepts_sub_;
    }

    void callback(const perception_msgs::PersonPerceptArrayConstPtr& percepts1, const sensor_msgs::PointCloud2ConstPtr& pointcloud)
    {
        perception_msgs::PersonPerceptArray percepts = *percepts1;

        ROS_DEBUG("[point_cloud] Callback Called.");

        // Get PointCloudXYZRGB from sensor_msgs::PointCloud2
        pcl::PointCloud<pcl::PointXYZRGB> pcl_cloud;
        pcl::fromROSMsg(*pointcloud, pcl_cloud);

        //ROS_INFO("No of faces: %d", percepts.person_percepts.size());
				uint32_t width = pcl_cloud.width;
				uint32_t height = pcl_cloud.height;

        // 2. Calculate 3D Point at Depth Camera
        std::vector<Eigen::Vector4f> pcl_centroid;
        pcl_centroid.resize(percepts.person_percepts.size());
        for(size_t i = 0; i < percepts.person_percepts.size(); i++)
        {
            perception_msgs::PersonPercept percept = percepts.person_percepts[i];

						if (percept.face_detected == 0)
						{
							percepts.person_percepts[i].face_pos3d.x = prevPos[percept.session_face_id][0];
							percepts.person_percepts[i].face_pos3d.y = prevPos[percept.session_face_id][1];
							percepts.person_percepts[i].face_pos3d.z = prevPos[percept.session_face_id][2];
							percepts.person_percepts[i].frame_id = "kinect2_head_rgb_optical_frame";
							continue;
						}

            std::vector<cv::Point> centroid_points;
            pcl::PointCloud<pcl::PointXYZRGB> pcl_centroid_points;

            int vx = percept.face_roi.width / 3.0;
            int vy = percept.face_roi.height / 3.0;
            int w = vx;
            int h = vy;

            percepts.person_percepts[i].face_pos3droi.x_offset = percept.face_roi.x_offset + vx;
            percepts.person_percepts[i].face_pos3droi.y_offset = percept.face_roi.y_offset + vy;
            percepts.person_percepts[i].face_pos3droi.width = w;
            percepts.person_percepts[i].face_pos3droi.height = h;

						cv::Point pt;
            for(int y = 0; y < h; y++)
            {
                for(int x = 0; x < w; x++)
                {
										// 얼굴의 위치를 좌우 Flip된 영상에서 찾기 때문에 아래와 같이 얼굴 중심부의 X 좌표를 변경해야 함
										pt.x = percept.face_roi.x_offset + vx + x; //(width - percept.face_roi.x_offset - 1) - vx - x;
                    pt.y = percept.face_roi.y_offset + vy + y;
                    if (!isnan(pcl_cloud(pt.x,pt.y).x) && !isnan(pcl_cloud(pt.x,pt.y).y))
                    {
                        pcl_centroid_points.push_back(pcl_cloud(pt.x, pt.y));
                    }
										// 아래와 같이 필요할 때 PointCloud에 표식을 그려넣을 수 있음.
										//uint32_t rgb = (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
										//pcl_cloud(pt.x,pt.y).rgb = *reinterpret_cast<float*>(&rgb);
                }
            }

            Eigen::Vector4f centroid;
            pcl::compute3DCentroid(pcl_centroid_points, centroid);

            if(!isnan(centroid[0]))
            {
                percepts.person_percepts[i].face_pos3d.x = /*-1.0 * */centroid[0];
                percepts.person_percepts[i].face_pos3d.y = centroid[1];
                percepts.person_percepts[i].face_pos3d.z = centroid[2];
                percepts.person_percepts[i].frame_id = "kinect2_head_rgb_optical_frame";

                prevPos[percepts.person_percepts[i].session_face_id] = centroid;
								//cout << "DIST: " << centroid[2] << endl;
            }
            else
            {
                string session_face_id = percepts.person_percepts[i].session_face_id;
                if (prevPos.find(session_face_id) != prevPos.end())
                {
                    percepts.person_percepts[i].face_pos3d.x = prevPos[session_face_id][0];
                    percepts.person_percepts[i].face_pos3d.y = prevPos[session_face_id][1];
                    percepts.person_percepts[i].face_pos3d.z = prevPos[session_face_id][2];
                    percepts.person_percepts[i].frame_id = "kinect2_head_rgb_optical_frame";
                }
                else
                {
                    percepts.person_percepts[i].face_pos3d.x = 0;
                    percepts.person_percepts[i].face_pos3d.y = 0;
                    percepts.person_percepts[i].face_pos3d.z = 0;
                    percepts.person_percepts[i].frame_id = "kinect2_head_rgb_optical_frame";
                }
            }
       }

       ROS_DEBUG("POS3D Publishing: %d", int(percepts.person_percepts.size()));
        publisher.publish(percepts);

				// 디버깅 목적으로 아래 코멘트를 열어서 PointCloud를 배포할 수 있음
				/*
				sensor_msgs::PointCloud2 msg;
				pcl::toROSMsg(pcl_cloud, msg);
				pcl_pub.publish(msg);
				*/
    }

private:
    ros::NodeHandle nh_;
    //pcl
    message_filters::Subscriber<perception_msgs::PersonPerceptArray> *percepts_sub_;
    message_filters::Subscriber<sensor_msgs::PointCloud2> *pointcloud_sub_;
    typedef message_filters::sync_policies::ApproximateTime<perception_msgs::PersonPerceptArray, sensor_msgs::PointCloud2> MySyncPolicy;
    message_filters::Synchronizer<MySyncPolicy> *sync_;

    tf::TransformBroadcaster br_;

    ros::Publisher publisher;

    std::map<string,Eigen::Vector4f> prevPos;

		// 디버깅 목적으로 아래 코멘트를 열어서 PointCloud를 배포할 수 있음
		// ros::Publisher pcl_pub;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "face_position_detector_using_pointcloud");
    Face3DPositionEstimator hd;
    ros::spin();
}
