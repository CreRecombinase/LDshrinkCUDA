#ifndef LDSHRINK_TYPES_H
#define LDSHRINK_TYPES_H

#include <RcppEigen.h>

typedef Eigen::Map<Eigen::ArrayXd> marrayd;
typedef Eigen::Map<Eigen::ArrayXi> marrayi;
typedef Eigen::Map<Eigen::MatrixXd> mmatd ;
typedef Eigen::Map<Eigen::VectorXd> mvecd;

typedef Eigen::Map<Eigen::ArrayXf> marrayf;
typedef Eigen::Map<Eigen::MatrixXf> mmatf ;
typedef Eigen::Map<Eigen::VectorXf> mvecf ;

typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> rowdmat;
typedef Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> rowfmat;
typedef Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor> coldmat;
typedef Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor> colfmat;

typedef Eigen::Map<rowdmat> mrowdmat;
typedef Eigen::Map<coldmat> mcoldmat;

typedef Eigen::Map<rowfmat> mrowfmat;
typedef Eigen::Map<colfmat> mcolfmat;

#endif
