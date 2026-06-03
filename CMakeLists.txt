cmake_minimum_required(VERSION 3.8)
project(arm)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY urdf meshes launch config worlds
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/ball_detector.py
    scripts/trajectory_predictor.py
    scripts/ik_solver.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()