#include <iostream>
#include "ros/ros.h"
#include <tf/transform_broadcaster.h>
#include "sensor_msgs/PointCloud2.h"
#include <pcl/io/pcd_io.h>
#include <pcl_conversions/pcl_conversions.h>
#include <bits/stdc++.h>
#include <tf/transform_datatypes.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/console/parse.h>
#include <pcl/common/transforms.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>

#include <Eigen/StdVector>
#include <Eigen/Dense>
#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <eigen3/Eigen/Dense>

#include "ceres/loss_function.h"
#include "ceres/ceres.h"
#include "ceres/jet.h"
#include "ceres/rotation.h"

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::LossFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

using namespace std;
using namespace pcl;

#define weight 0.8

Eigen::Matrix<double, 4, 4> cumm_transform;
pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_frame(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr previous_frame(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr current_frame(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PCLPointCloud2 pcl_pc2;
//std::vector<Eigen::Matrix<float,4,1>, Eigen::aligned_allocator<Eigen::Matrix<float,4,1> > > prev_points;
template <typename T>
void QuaternionToRotation(const T q_x, const T q_y, const T q_z, const T q_w, T R[9])
{
    T xx, xy, xz, xw, yy, yz, yw, zz, zw;
    xx = q_x * q_x;
    xy = q_x * q_y;
    xz = q_x * q_z;
    xw = q_x * q_w;
    yy = q_y * q_y;
    yz = q_y * q_z;
    yw = q_y * q_w;
    zz = q_z * q_z;
    zw = q_z * q_w;

    /* R[0] R[1] R[2]
     R[3] R[4] R[5]
     R[6] R[7] R[8] */
    R[0] = 1.0 - 2.0 * (yy + zz);
    R[1] = 2.0 * (xy - zw);
    R[2] = 2.0 * (xz + yw);
    R[3] = 2.0 * (xy + zw);
    R[4] = 1.0 - 2.0 * (xx + zz);
    R[5] = 2.0 * (yz - xw);
    R[6] = 2.0 * (xz - yw);
    R[7] = 2.0 * (yz + xw);
    R[8] = 1.0 - 2.0 * (xx + yy);
};
struct ResidualError
{
    ResidualError(const std::vector<Eigen::Matrix<float, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 4, 1>>> prev_points) : prev_points(prev_points) {}
    template <typename T>
    bool operator()(const T *const q_x, const T *q_y, const T *q_z, const T *q_w, const T *t_x, const T *t_y, const T *t_z, T *residual) const
    {
        Eigen::Matrix<T, 4, 1> current_point;
        Eigen::Matrix<T, 4, 1> prev_point;
        Eigen::Matrix<T, 4, 4> variable;

        T R[9];
        QuaternionToRotation(q_x[0], q_y[0], q_z[0], q_w[0], R);
        /*Eigen::Quaterniond Q(q_x[0], q_y[0], q_z[0], q_w[0]);
        Eigen::Matrix<double, 3, 3> R = Q.toRotationMatrix();*/

        double error, current_intensity, prev_intensity;
        int found = -1;
        unsigned int i, j;
        double tot_error = 0.0;

        //make the 4*4 transformation matrix
        variable(0, 0) = R[0];
        variable(0, 1) = R[1];
        variable(0, 2) = R[2];
        variable(0, 3) = t_x[0];
        variable(1, 0) = R[3];
        variable(1, 1) = R[4];
        variable(1, 2) = R[5];
        variable(1, 3) = t_y[0];
        variable(2, 0) = R[6];
        variable(2, 1) = R[7];
        variable(2, 2) = R[8];
        variable(2, 3) = t_z[0];

        variable(3, 0) = T(0.0);
        variable(3, 1) = T(0.0);
        variable(3, 2) = T(0.0);
        variable(3, 3) = T(1.0);
        /*variable(seqN(0,3), seqN(0,3)) = R; 
        variable(seqN(0,3), seqN(3,1)) = t;*/

        for (i = 0; i < prev_points.size(); i++) //looping over the array of points
        {
            prev_point(0, 0) = T(prev_points[i](0, 0));
            prev_point(1, 0) = T(prev_points[i](1, 0));
            prev_point(2, 0) = T(prev_points[i](2, 0));
            prev_point(3, 0) = T(1.0);
            prev_intensity = prev_points[i](3, 0);
            current_point = variable * prev_point;
            found = -1;
            for (j = 0; j < current_frame->points.size(); j++)
            {
                //need to optimise this
                if (T(current_frame->points[j].x) == current_point(0, 0) && T(current_frame->points[j].y) == current_point(1, 0) && T(current_frame->points[j].z) == current_point(2, 0))
                {
                    found = j;
                    break;
                }
            }
            if (found == -1)
                continue;
            current_intensity = current_frame->points[found].intensity;
            error = prev_intensity - current_intensity;
            //residual[0] = sqrt(5.0) * (x3[0] - x4[0]);
            tot_error += error * error;
        }

        residual[0] = T(tot_error);
        return true;
    }

private:
    const std::vector<Eigen::Matrix<float, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 4, 1>>> prev_points;
};

void performTracking()
{
    static tf::TransformBroadcaster br;

    pcl::fromPCLPointCloud2(pcl_pc2, *current_frame);

    cout << endl
         << current_frame->size() << endl;
    cout << endl
         << previous_frame->size() << endl;

    //Eigen::Matrix variable = cumm_transform;
    Eigen::Matrix<double, 3, 3> R;
    Eigen::Matrix<double, 3, 1> t;
    /*R = cumm_transform(Eigen::seqN(0,3), Eigen::seqN(0,3)); 
    t = cumm_transform(Eigen::seqN(0,3), Eigen::seqN(3,1));*/
    R(0, 0) = cumm_transform(0, 0);
    R(0, 1) = cumm_transform(0, 1);
    R(0, 2) = cumm_transform(0, 2);
    R(1, 0) = cumm_transform(1, 0);
    R(1, 1) = cumm_transform(1, 1);
    R(1, 2) = cumm_transform(1, 2);
    R(2, 0) = cumm_transform(2, 0);
    R(2, 1) = cumm_transform(2, 1);
    R(2, 2) = cumm_transform(2, 2);

    Eigen::Quaterniond Q{R};

    t(0, 0) = cumm_transform(0, 3);
    t(1, 0) = cumm_transform(1, 3);
    t(2, 0) = cumm_transform(2, 3);

    Eigen::MatrixXf point(4, 1);
    std::vector<Eigen::Matrix<float, 4, 1>, Eigen::aligned_allocator<Eigen::Matrix<float, 4, 1>>> prev_points;
    prev_points.clear();

    unsigned int h = previous_frame->height;
    unsigned int w = previous_frame->width;
    unsigned int start_row, start_col, end_row, end_col, i, j;

    start_row = (1 - weight) / 2 * h;
    end_row = (weight + (1 - weight) / 2) * h;
    start_col = (1 - weight) / 2 * w;
    end_col = (weight + (1 - weight) / 2) * w;

    for (i = start_row; i < end_row; i++)
    {
        for (j = start_col; j < end_col; j++)
        {
            point(0, 0) = previous_frame->at(i, j).x;
            point(1, 0) = previous_frame->at(i, j).y;
            point(2, 0) = previous_frame->at(i, j).z;
            point(3, 0) = previous_frame->at(i, j).intensity;
            prev_points.push_back(point);
        }
    }

    //gauss newton minimisation of photometric error
    //============================ceres part====================================
    Problem problem;
    problem.AddResidualBlock(new AutoDiffCostFunction<ResidualError, 1, 1, 1, 1, 1, 1, 1, 1>(new ResidualError(prev_points)),
                             new ceres::HuberLoss(1.0),
                             &Q.x(), &Q.y(), &Q.z(), &Q.w(), &t(0, 0), &t(1, 0), &t(2, 0));

    Solver::Options solver_options;
    solver_options.minimizer_type = ceres::TRUST_REGION;
    solver_options.trust_region_strategy_type = ceres::DOGLEG; //or LEVENBERG_MARQUARDT
    solver_options.dogleg_type = ceres::SUBSPACE_DOGLEG;       //or TRADITIONAL_DOGLEG

    //Relax the requirement that the trust-region algorithm take strictly decreasing steps.
    solver_options.use_nonmonotonic_steps = true;

    //The window size used by the step selection algorithm to accept non-monotonic steps.
    //solver_options.maximum_consecutive_nonmonotonic_steps = 5; //default 5

    solver_options.linear_solver_type = ceres::DENSE_QR;

    //Maximum number of iterations for which the solver should run.
    solver_options.max_num_iterations = 50; //default = 50

    //Number of threads used by Ceres to evaluate the Jacobian.
    solver_options.num_threads = 1; //default = 1

    //output the logged data in STDOUT
    solver_options.minimizer_progress_to_stdout = true;

    //solver_options.ordering_type = ceres::SCHUR;

    Solver::Summary summary;
    Solve(solver_options, &problem, &summary); //after this step,minimisation occurs

    std::cout << summary.FullReport() << "\n";

    //================================================================================

    Eigen::Matrix<double, 3, 3> new_R;
    new_R = Q.toRotationMatrix();

    cumm_transform(0, 0) = new_R(0, 0);
    cumm_transform(0, 1) = new_R(0, 1);
    cumm_transform(0, 2) = new_R(0, 2);
    cumm_transform(1, 0) = new_R(1, 0);
    cumm_transform(1, 1) = new_R(1, 1);
    cumm_transform(1, 2) = new_R(1, 2);
    cumm_transform(2, 0) = new_R(2, 0);
    cumm_transform(2, 1) = new_R(2, 1);
    cumm_transform(2, 2) = new_R(2, 2);

    //==================================================================================
    tf::Vector3 origin;
    origin.setValue(static_cast<double>(cumm_transform(0, 3)), static_cast<double>(cumm_transform(1, 3)), 0); //static_cast<double>(cumm_transform(2,3)));

    cout << origin << endl;
    tf::Matrix3x3 tf3d;
    tf3d.setValue(static_cast<double>(cumm_transform(0, 0)), static_cast<double>(cumm_transform(0, 1)), static_cast<double>(cumm_transform(0, 2)),
                  static_cast<double>(cumm_transform(1, 0)), static_cast<double>(cumm_transform(1, 1)), static_cast<double>(cumm_transform(1, 2)),
                  static_cast<double>(cumm_transform(2, 0)), static_cast<double>(cumm_transform(2, 1)), static_cast<double>(cumm_transform(2, 2)));

    tf::Quaternion tfqt;
    tf3d.getRotation(tfqt);

    tf::Transform transform;
    transform.setOrigin(origin);
    transform.setRotation(tfqt);

    br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "odom", "base_link"));
    pcl::transformPointCloud(*previous_frame, *transformed_frame, cumm_transform);

    previous_frame = current_frame;
}

void ROS_to_PCL(const boost::shared_ptr<const sensor_msgs::PointCloud2> &input)
{
    pcl_conversions::toPCL(*input, pcl_pc2);
}
int main(int argc, char **argv)
{
    ros::init(argc, argv, "tracking_camera");
    ros::NodeHandle n;
    ros::Subscriber sub = n.subscribe("/pointcloud2_type_topic", 1000, ROS_to_PCL);
    ros::Publisher cloud_pub = n.advertise<sensor_msgs::PointCloud2>("/transformed_cloud", 1000);

    cumm_transform << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;

    tf::Transform transform;
    transform.setIdentity();

    while (ros::ok())
    {
        sensor_msgs::PointCloud2 object_msg;

        performTracking();

        pcl::toROSMsg(*transformed_frame.get(), object_msg);

        object_msg.header.frame_id = "odom";
        cloud_pub.publish(object_msg);

        ros::spinOnce();
    }
    return 0;
}