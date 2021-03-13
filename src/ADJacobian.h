// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


// Updated version of AutoDiffJacobian.h by Gael Guennebaud (2009)
// Updated by Emil Fresk (2016)
//
// Changes:
// * Removed unnecessary types from the Functor by inferring from its types
// * Removed the inputs() function reference, replaced with .rows()
// * Updated the forward constructor to use Variadic templates
// * Added optional parameters to the Functor

#include <unsupported/Eigen/AutoDiff>

#ifndef EIGEN_AD_JACOBIAN_H
#define EIGEN_AD_JACOBIAN_H

namespace Eigen
{

template<typename Functor>
class ADJacobian : public Functor
{
public:
  // forward constructors
  template<typename... T>
  ADJacobian(const T& ...Values) : Functor(Values...) {}

  typedef typename Functor::InputType InputType;
  typedef typename Functor::ValueType ValueType;
  typedef typename ValueType::Scalar Scalar;

  enum {
    InputsAtCompileTime = InputType::RowsAtCompileTime,
    ValuesAtCompileTime = ValueType::RowsAtCompileTime
  };

  typedef Matrix<Scalar, ValuesAtCompileTime, InputsAtCompileTime> JacobianType;
  typedef typename JacobianType::Index Index;

  typedef Matrix<Scalar, InputsAtCompileTime, 1> DerivativeType;
  typedef AutoDiffScalar<DerivativeType> ActiveScalar;

  typedef Matrix<ActiveScalar, InputsAtCompileTime, 1> ActiveInput;
  typedef Matrix<ActiveScalar, ValuesAtCompileTime, 1> ActiveValue;

  template<typename... ParamsType>
  void operator() (const InputType& x, ValueType* v, JacobianType* _jac=0,
                   const ParamsType&... Params) const
  {
    eigen_assert(v!=0);

    if (!_jac)
    {
      Functor::operator()(x, v, Params...);
      return;
    }

    JacobianType& jac = *_jac;

    ActiveInput ax = x.template cast<ActiveScalar>();
    ActiveValue av(jac.rows());

    if(InputsAtCompileTime==Dynamic)
      for (Index j=0; j<jac.rows(); j++)
        av[j].derivatives().resize(x.rows());

    for (Index i=0; i<jac.cols(); i++)
      ax[i].derivatives() = DerivativeType::Unit(x.rows(),i);

    Functor::operator()(ax, &av, Params...);

    for (Index i=0; i<jac.rows(); i++)
    {
      (*v)[i] = av[i].value();
      jac.row(i) = av[i].derivatives();
    }
  }
};

}

#endif // EIGEN_AD_JACOBIAN_H
