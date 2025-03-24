#include "rclcpp/rclcpp.hpp"
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <geometry_msgs/msg/polygon.hpp>
#include <geometry_msgs/msg/point32.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <custom_msgs_pkg/msg/polygon_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <casadi/casadi.hpp>
#include <chrono>
#include <limits>

using namespace std;
using namespace casadi;

class MPCPlannerCorridors: public rclcpp::Node{
public:
    MPCPlannerCorridors(): Node("mpc_controller"){
        odometry_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/dlio/odom_node/odom", 10, bind(&MPCPlannerCorridors::odometryCallback, this, placeholders::_1));
        mpc_path_sub_ = this->create_subscription<nav_msgs::msg::Path>(
            "/planners/mpc_path", 10, bind(&MPCPlannerCorridors::pathCallback, this, placeholders::_1));
        risk_map_sub_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            "/planners/risk_map", 10, bind(&MPCPlannerCorridors::riskMapCallback, this, placeholders::_1));

        control_pub_ = this->create_publisher<geometry_msgs::msg::Twist>("/planners/mpc_cmd_vel_unstamped", 10);
        RCLCPP_INFO(this->get_logger(), "MPC Controller Initialized.");
    }

private:
    void odometryCallback(const nav_msgs::msg::Odometry::SharedPtr msg){
        odometry_ = msg;

        if (!mpc_path_) {
            RCLCPP_WARN(this->get_logger(), "No MPC Path Received Yet.");
            return;
        }

        if (!risk_map_) {
            RCLCPP_WARN(this->get_logger(), "No Risk Map Received Yet.");
            return;
        }
        
        computeControl();
    }

    void riskMapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr msg){
        risk_map_ = msg;
        global_to_lower_left_ = reshape(DM({
                                        risk_map_->info.origin.position.x,
                                        risk_map_->info.origin.position.y,
                                        0.0}), 3, 1);
        lower_left_to_risk_center_ = reshape(DM({
                                            risk_map_->info.height*risk_map_->info.resolution/2,
                                            risk_map_->info.width*risk_map_->info.resolution/2,
                                            0.0}), 3, 1);
        global_risk_to_center_ = global_to_lower_left_ - lower_left_to_risk_center_;

        int height = risk_map_->info.height;
        int width = risk_map_->info.width;
        double resolution = risk_map_->info.resolution;

        vector<double> grid_x(height);
        vector<double> grid_y(width);

        for (int i = 0; i < height; i++) {
            grid_x[i] = risk_map_->info.origin.position.x + (i + 0.5) * resolution;
        }
        for (int j = 0; j < width; j++) {
            grid_y[j] = risk_map_->info.origin.position.y + (j + 0.5) * resolution;
        }

        vector<double> risk_values(width * height);
        for (int j = 0; j < height; j++) {
            for (int i = 0; i < width; i++) {
            int idx = j * width + i;
            int occ_value = risk_map_->data[idx];
            risk_values[idx] = static_cast<double>(occ_value);
            }
        }

        vector<vector<double>> grid = {grid_x, grid_y};
        continuous_risk_map_ = interpolant("interp", "bspline", grid, risk_values, {{"degree", vector<int>{3, 3}}});
        // continuous_risk_map_ = interpolant("interp", "linear", grid, risk_values);
    }

    void pathCallback(const nav_msgs::msg::Path::SharedPtr msg) {
        mpc_path_ = msg;
    }

    void computeControl(){
        const int N = 20;
        double dt = 0.1;

        MX X = MX::sym("state_variables", 3, N + 1);
        MX U = MX::sym("control_variables", 2, N);

        vector<MX> variables_list = {X, U};
        vector<string> variables_name = {"states", "inputs"};

        MX variables_flat = vertcat(reshape(X, X.size1()*X.size2(), 1), reshape(U, U.size1()*U.size2(), 1));

        Function pack_variables_fn = Function("pack_variables_fn", variables_list, {variables_flat}, variables_name, {"flat"});
        Function unpack_variables_fn = Function("unpack_variables_fn", {variables_flat}, variables_list, {"flat"}, variables_name);

        DMVector upper_bounds = unpack_variables_fn(DM::inf(variables_flat.rows(), 1));
        DMVector lower_bounds = unpack_variables_fn(-DM::inf(variables_flat.rows(), 1));
        RCLCPP_INFO(this->get_logger(), "Made Variables and Bound Arrays");

        // set initial and final state vectors
        DM initial_state = reshape(DM(vector<double>{
            odometry_->pose.pose.position.x, 
            odometry_->pose.pose.position.y, 
            yaw_from_quaternion(odometry_->pose.pose.orientation)}), 3, 1);
        RCLCPP_INFO(this->get_logger(), "Initial Pose: %s", initial_state.get_str().c_str());

        // input bounds
        lower_bounds[1] = repmat(DM(vector<double>{0.0, -pi/4}), 1, lower_bounds[1].size2());
        upper_bounds[1] = repmat(DM(vector<double>{1.5, pi/4}), 1, upper_bounds[1].size2());
        RCLCPP_INFO(this->get_logger(), "Set Input Bounds");

        // state bounds
        lower_bounds[0](0, Slice()) = risk_map_->info.origin.position.x * DM::ones(1, lower_bounds[0].size2());
        lower_bounds[0](1, Slice()) = risk_map_->info.origin.position.y * DM::ones(1, lower_bounds[0].size2());
        upper_bounds[0](0, Slice()) = (risk_map_->info.origin.position.x + 
                                        risk_map_->info.resolution * risk_map_->info.height
                                        ) * DM::ones(1, lower_bounds[0].size2());
        upper_bounds[0](1, Slice()) = (risk_map_->info.origin.position.y + 
                                        risk_map_->info.resolution * risk_map_->info.width
                                        ) * DM::ones(1, lower_bounds[0].size2());

        // determine the closest point index on the mpc path
        double x_odom = odometry_->pose.pose.position.x;
        double y_odom = odometry_->pose.pose.position.y;

        long unsigned int closest_idx = 0;
        double min_dist = numeric_limits<double>::infinity();
        for (long unsigned int i=0; i < mpc_path_->poses.size(); i++){
            double x_path = mpc_path_->poses[i].pose.position.x;
            double y_path = mpc_path_->poses[i].pose.position.y;
            double dist = sqrt(pow(x_odom - x_path, 2) + pow(y_odom - y_path, 2));
            if (dist < min_dist){
                min_dist = dist;
                closest_idx = i;
            }
        }
        RCLCPP_INFO(this->get_logger(), "Closest Index: %d", closest_idx);

        // running state cost, control cost, and risk cost
        vector<vector<double>> Q_vals = {{10, 0}, {0, 10}};
        DM Q = DM(Q_vals);
        vector<vector<double>> R_vals = {{1, 0}, {0, 1/pi}};
        DM R = DM(R_vals);
        MX objective = 0.0;
        for (int k=0; k < N; ++k){
            MX position = X(Slice(0, 2), k);
            int ref_index = min(closest_idx + static_cast<long unsigned int>(k), mpc_path_->poses.size() - 1);
            RCLCPP_INFO(this->get_logger(), "Reference Index: %d", ref_index);
            geometry_msgs::msg::PoseStamped ref_pose = mpc_path_->poses[ref_index];
            DM ref_position = reshape(DM({ref_pose.pose.position.x, ref_pose.pose.position.y}), 2, 1);

            MX state_penalty = position - ref_position;
            MX control_penalty = U(Slice(), k);
            objective = objective + mtimes(mtimes(state_penalty.T(), Q), state_penalty);
            objective = objective + mtimes(mtimes(control_penalty.T(), R), control_penalty);
        }   
        RCLCPP_INFO(this->get_logger(), "Set Running State and Control Cost");

        // initial state constraint
        MX initial_state_constraint = reshape(X(Slice(), 0) - initial_state, -1, 1);
        RCLCPP_INFO(this->get_logger(), "Set Initial State Constraint");

        // add acceleration constraint
        MX v_dot_constraint = reshape((1/dt)*(U(0, Slice(1, N)) - U(0, Slice(0, N-1))), -1, 1);
        MX r_dot_constraint = reshape((1/dt)*(U(1, Slice(1, N)) - U(1, Slice(0, N-1))), -1, 1);
        RCLCPP_INFO(this->get_logger(), "Set V_dot Constraint");
        RCLCPP_INFO(this->get_logger(), "Set R_dot Constraint");

        // dynamics constraints
        MX x_now = X(Slice(), Slice(0, N));
        MX delta_x = dt * vertcat(
            U(0, Slice()) * cos(X(2, Slice(0, N))),
            U(0, Slice()) * sin(X(2, Slice(0, N))),
            U(1, Slice()));
        MX x_next = x_now + delta_x;
        MX dynamics_constraint = reshape(x_next - X(Slice(), Slice(1, N+1)), -1, 1);
        RCLCPP_INFO(this->get_logger(), "Set Dynamics Constraint");

        // polyhedron constraints
        vector<MX> polyhedron_constraint_vector;
        // int last_k = 0;
        // for(size_t i=0; i < polyhedrons.size(); ++i){
        //     int next_k = static_cast<int>(path_proportions[i] * N);
        //     vec_E<Hyperplane2D> hyperplanes = polyhedrons[i].hyperplanes();
        //     for(int k=last_k; k < next_k; ++k){
        //         // ensure point is in the polyhedron
        //         for(Hyperplane2D hyperplane : hyperplanes){
        //             Vec2f normal = hyperplane.n_;
        //             Vec2f point = hyperplane.p_;
        //             MX value = normal[0]*(X(0, k) - point[0]) + normal[1]*(X(1, k) - point[1]);
        //             polyhedron_constraint_vector.push_back(value);
        //         }
        //     }
        //     last_k = next_k;
        // }
        MX polyhedron_constraint = vertcat(polyhedron_constraint_vector);
        RCLCPP_INFO(this->get_logger(), "Set Polyhedron Constraint");
        
        MX equality_constraints = vertcat(
            initial_state_constraint, 
            dynamics_constraint,
            v_dot_constraint);
        MX constraints = vertcat(equality_constraints, r_dot_constraint, polyhedron_constraint);

        // set up NLP solver and solve the program
        Function solver = nlpsol("solver", "ipopt", MXDict{
            {"x", variables_flat},
            {"f", objective},
            {"g", constraints}
        });

        // Set constraint bounds
        DM zero_bg_constraints = vertcat(
            DM::zeros(initial_state_constraint.size1(), 1), 
            DM::zeros(dynamics_constraint.size1(), 1));

        DM lbg = vertcat(
            zero_bg_constraints, 
            -DM::ones(v_dot_constraint.size1(), 1),
            -(pi/4)*DM::ones(r_dot_constraint.size1(), 1),
            -DM::inf(polyhedron_constraint.size1(), 1));
        DM ubg = vertcat(
            zero_bg_constraints,
            DM::ones(v_dot_constraint.size1(), 1),
            (pi/4)*DM::ones(r_dot_constraint.size1(), 1),
            DM::zeros(polyhedron_constraint.size1(), 1));

        // Flatten decision variable bounds
        DM lbx = pack_variables_fn(lower_bounds)[0];
        DM ubx = pack_variables_fn(upper_bounds)[0];
        
        // Initial guess for optimization
        // DM initial_guess = DM::zeros(variables_flat.size1(), 1);
        // for (int i = 0; i < N + 1; ++i) {
        //     double alpha = static_cast<double>(i) / N;
        //     initial_guess(3 * i) = (1 - alpha) * initial_state(0) + alpha * final_position(0);
        //     initial_guess(3 * i + 1) = (1 - alpha) * initial_state(1) + alpha * final_position(1);
        //     // initial_guess(3 * i + 2) = (1 - alpha) * initial_state(2) + alpha * final_state(2);
        // }

        // Solve NLP
        map<string, DM> solver_args = {
            // {"x0", initial_guess},   
            {"lbx", lbx}, 
            {"ubx", ubx}, 
            {"lbg", lbg},
            {"ubg", ubg}
            };

        RCLCPP_INFO(this->get_logger(), "Solving NLP");
        map<string, DM> solver_result = solver(solver_args);
        RCLCPP_INFO(this->get_logger(), "Optimization complete");

        DM solution = solver_result["x"];
        DMVector unpacked_solution = unpack_variables_fn(solution);

        // extract the first control input
        DM control_input = unpacked_solution[1](Slice(), 0);
        RCLCPP_INFO(this->get_logger(), "Control Input: %s", control_input.get_str().c_str());
        
        // publish the control input
        geometry_msgs::msg::Twist control_msg;
        control_msg.linear.x = static_cast<double>(unpacked_solution[1](0, 0));
        control_msg.angular.z = static_cast<double>(unpacked_solution[1](1, 0));
        control_pub_->publish(control_msg);

        // log the v, omega, x, y, and theta values for plotting

        // allow to read in bounds from a config file

        // RCLCPP_INFO(this->get_logger(), "X Optimal: %s", unpacked_solution[0].get_str().c_str());
        // RCLCPP_INFO(this->get_logger(), "U Optimal: %s", unpacked_solution[1].get_str().c_str());
    }

    double yaw_from_quaternion(geometry_msgs::msg::Quaternion& quaternion){
        return atan2(2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y), 
        1.0 - 2.0 * (pow(quaternion.y, 2) + pow(quaternion.z, 2)));
    }

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odometry_sub_;
    rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr mpc_path_sub_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr risk_map_sub_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr control_pub_;

    nav_msgs::msg::Odometry::SharedPtr odometry_;
    nav_msgs::msg::OccupancyGrid::SharedPtr risk_map_;
    nav_msgs::msg::Path::SharedPtr mpc_path_;

    DM global_to_lower_left_;
    DM lower_left_to_risk_center_;
    DM global_risk_to_center_;
    Function continuous_risk_map_;

    // casadi optimization relevant declarations
    Function pack_variables_fn_;
    Function unpack_variables_fn_;
    DMVector lower_bounds_;
    DMVector upper_bounds_;
    SX cost_;
    DM x_opt_;
    DM u_opt_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    shared_ptr<MPCPlannerCorridors> node = make_shared<MPCPlannerCorridors>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
