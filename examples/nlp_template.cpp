//[[[cog import nlp_to_generate as nlp ]]]
//[[[end]]]


POLYMPC_FORWARD_NLP_DECLARATION(
    /*[[[cog 
    cog.outl(str(nlp.base_name) + ", // Name")
    cog.outl(str(nlp.nx) + ", // Number of primal variables")
    cog.outl(str(nlp.ne) + ", // Number of equalities")
    cog.outl(str(nlp.ni) + ", // Number of inequalities")
    cog.outl(str(0) + ", // Number of parameters (not used)")
    cog.outl(str(nlp.scalar) + " // Type);")
    ]]]
    [[[end]]]*/

//[[[cog cog.outl(f"class {nlp.base_name} : public ProblemBase<{nlp.base_name}>") ]]]
//[[[end]]]
{



    EIGEN_STRONG_INLINE void cost(const Eigen::Ref<const nlp_variable_t>& var, 
                                  scalar_t &cost) noexcept
    {
      //[[[cog nlp.gen_cost() ]]]
      //[[[end]]]
    }

    EIGEN_STRONG_INLINE void cost_gradient(const Eigen::Ref<const nlp_variable_t>& var, 
                                           scalar_t &_cost, 
                                           Eigen::Ref<nlp_variable_t> cost_gradient) noexcept
    {
      //[[[cog nlp.gen_cost_gradient() ]]]
      //[[[end]]]
    }

    EIGEN_STRONG_INLINE void cost_gradient_hessian(const Eigen::Ref<const nlp_variable_t>& var, 
                                                   const Eigen::Ref<const static_parameter_t>& p,
                                                   scalar_t &_cost, 
                                                   Eigen::Ref<nlp_variable_t> _cost_gradient, 
                                                   Eigen::Ref<nlp_hessian_t> hessian) noexcept
    {
      //[[[cog nlp.gen_cost_gradient_hessian() ]]]
      //[[[end]]]
    }

    EIGEN_STRONG_INLINE void equalities(const Eigen::Ref<const nlp_variable_t>& var, 
                                        Eigen::Ref<nlp_constraints_t> _equalities) const noexcept
    {
      //[[[cog nlp.gen_equalities() ]]]
      //[[[end]]]
    }

    EIGEN_STRONG_INLINE void equalities_linearised(const Eigen::Ref<const nlp_variable_t>& var,
                                                   Eigen::Ref<nlp_constraints_t> equalities,
                                                   Eigen::Ref<nlp_eq_jacobian_t> jacobian) noexcept
    {
      //[[[cog nlp.gen_equalities_linearised() ]]]
      //[[[end]]]
    }

    EIGEN_STRONG_INLINE void inequalities(const Eigen::Ref<const nlp_variable_t>& var, 
                                          Eigen::Ref<nlp_constraints_t> _equalities) const noexcept
    {
      //[[[cog nlp.gen_inequalities() ]]]
      //[[[end]]]
    }

    EIGEN_STRONG_INLINE void inequalities_linearised(const Eigen::Ref<const nlp_variable_t>& var,
                                                     Eigen::Ref<nlp_constraints_t> equalities,
                                                     Eigen::Ref<nlp_eq_jacobian_t> jacobian) noexcept
    {
      //[[[cog nlp.gen_inequalities_linearised() ]]]
      //[[[end]]]
    }
}
