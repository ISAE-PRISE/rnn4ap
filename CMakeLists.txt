# ---------------------------------------------------------------------
# RNN4AP project
# Copyright (C) 2021 ISAE
# 
# Purpose:
# Evaluation of Recurrent Neural Networks for future Autopilot Systems
#
# Contact:
# jean-baptiste.chaudron@isae-supaero.fr
# goncalo.fontes-neves@student.isae-supaero.fr
# ---------------------------------------------------------------------

PROJECT(RNN4AP C CXX)


CMAKE_MINIMUM_REQUIRED(VERSION 2.8.11)

SET(PACKAGE_NAME SMARTIES)
SET(DESCRIPTION_SUMMARY "RNN4AP Project .")
SET(TARGET_NAME smarties)

SET(ABI_VERSION_MAJOR "0")
SET(ABI_VERSION_MINOR "0")
SET(ABI_VERSION_PATCH "1")
SET(ABI_VERSION "${ABI_VERSION_MAJOR}.${ABI_VERSION_MINOR}.${ABI_VERSION_PATCH}")

SET(PACKAGE_BUGREPORT "jean-baptiste.chaudron@isae-supaero.fr")

SET(CMAKE_CXX_FLAGS "-std=c++0x")

INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/src/main)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/src/px4)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/src/rnn)


# -------------------------------------------------------------------
# LSTM test executable
SET(EXE_NAME1 LSTM_PX4)
SET(SRC_LIST1 	${CMAKE_SOURCE_DIR}/src/main/lstm_main.cc
				${CMAKE_SOURCE_DIR}/src/rnn/lstm.cc
				${CMAKE_SOURCE_DIR}/src/px4/px4_log_handler.cc
				)
				
ADD_EXECUTABLE(${EXE_NAME1} ${SRC_LIST1})
TARGET_LINK_LIBRARIES (${EXE_NAME1})
INSTALL(TARGETS ${EXE_NAME1} RUNTIME DESTINATION bin)

# -------------------------------------------------------------------
# GRU test executable
SET(EXE_NAME2 GRU_PX4)
SET(SRC_LIST2 	${CMAKE_SOURCE_DIR}/src/main/gru_main.cc
				${CMAKE_SOURCE_DIR}/src/rnn/gru.cc
				${CMAKE_SOURCE_DIR}/src/px4/px4_log_handler.cc
				)

ADD_EXECUTABLE(${EXE_NAME2} ${SRC_LIST2})
TARGET_LINK_LIBRARIES (${EXE_NAME2})
INSTALL(TARGETS ${EXE_NAME2} RUNTIME DESTINATION bin)        


# -------------------------------------------------------------------
# GRU test executable
SET(EXE_NAME3 PERF_LSTM_GRU_PX4)
SET(SRC_LIST3 	${CMAKE_SOURCE_DIR}/src/main/perf_main.cc
				${CMAKE_SOURCE_DIR}/src/rnn/gru.cc
				${CMAKE_SOURCE_DIR}/src/rnn/lstm.cc
				${CMAKE_SOURCE_DIR}/src/px4/px4_log_handler.cc
				)

ADD_EXECUTABLE(${EXE_NAME3} ${SRC_LIST3})
TARGET_LINK_LIBRARIES (${EXE_NAME3})
INSTALL(TARGETS ${EXE_NAME3} RUNTIME DESTINATION bin)           

# -------------------------------------------------------------------
# Copy datasets
INSTALL(DIRECTORY ${CMAKE_SOURCE_DIR}/datasets/px4 DESTINATION datasets)
INSTALL(DIRECTORY ${CMAKE_SOURCE_DIR}/datasets/imu DESTINATION datasets)
INSTALL(DIRECTORY ${CMAKE_SOURCE_DIR}/src/pytorch DESTINATION .)
