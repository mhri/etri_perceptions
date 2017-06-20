#include <iostream>
#include <string>
#include <vector>
#include <ros/package.h>
#include <ros/ros.h>
#include <perception_msgs/PersonPercept.h>
#include <perception_msgs/PersonPerceptArray.h>
#include <MindStateRecog.h>

using namespace std;
using namespace perception_msgs;

extern bool DEBUG;

/**
 * subscribed topics: camera/rgb/image_rect (type: sensor_msgs::Image)
 * published topics: /mhri/perception_core/perception_core_node/persons (type: mhri_common::PersonPerceptArray)
 */

class MindStateRecognition
{
  public:
    ros::Publisher publisher;

    CMindStateRecog m_MindStateRecog;

    MindStateRecognition()
    {
        string packagePath = ros::package::getPath("perception_mindstate");
        string dataPath = packagePath + "/data";

        initializeMindReader(dataPath, 4);

        // Message Publisher
        publisher = nh_.advertise<PersonPerceptArray>(
            "/mhri/perception_mindstate/percepts", 2);

        ROS_INFO("perception_mindstate initialized.");
    }

    ~MindStateRecognition()
    {
    }

    void initializeMindReader(const string dataPath, int num_frames)
    {
        if (this->m_MindStateRecog.LoadMindStateModel(dataPath + "/smodel.dat", num_frames) > 0)
            ROS_INFO("perception_mindstate Load MindStateModel succeed.");
        else
            ROS_INFO("perception_mindstate Load MindStateModel failed.");
    }


    string recognizeMindState(string id, vector<int> &landmarks)
    {
        vector <vector<Position> > positionVec;

        int num_points = (int)(landmarks.size()/2);
        vector<Position> newPtVec;
        for (int i=0; i<num_points; i++)
        {
            Position newPt;
            newPt.x = landmarks[2*i];
            newPt.y = landmarks[2*i + 1];

            newPtVec.push_back(newPt);
        } // for(it1)

        positionVec.push_back(newPtVec);

        string result = m_MindStateRecog.MindRecogRes(positionVec);

        // Return Values: '', Concentrating, Agree, Disagree

        //ROS_INFO("############ MIND STATE: PERSON=%d STATE=%s", id, result.c_str());
        
        return result;
    }


    void callback(const PersonPerceptArrayConstPtr &in_percepts)
    {
        //if (DEBUG)
            ROS_DEBUG("perception_mindstate callback() called!");

        PersonPerceptArray percepts = *in_percepts;

        for (size_t i = 0; i < percepts.person_percepts.size(); i++)
        {
            PersonPercept& person = percepts.person_percepts[i];
            if (person.face_detected == 0)
                continue;

            vector<int>& landmarks = person.stasm_landmarks;

            person.cognitive_status = this->recognizeMindState(person.trk_id, landmarks);
        }

        publisher.publish(percepts);
    }

    ros::NodeHandle nh_;

  private:
};


int main(int argc, char **argv)
{
    ros::init(argc, argv, "perception_mindstate_node");

    MindStateRecognition msr;

    ros::Subscriber sub =msr. nh_.subscribe("People_With_Face", 10, 
                                            &MindStateRecognition::callback, &msr);

    ros::spin();

    return 0;
}
