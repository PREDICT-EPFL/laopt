#ifndef LAOPT_HPIPM_INTERFACE_HPP
#define LAOPT_HPIPM_INTERFACE_HPP

#include "laopt/utility.hpp"
#include "laopt/solvers/qp_base.hpp"
#include "laopt/solvers/sqp_solver.hpp"
#include "laopt/tools/multiple_shooting.hpp"

#include "blasfeo_d_aux.h"
#include "hpipm_d_ocp_qp.h"
#include "hpipm_d_ocp_qp_ipm.h"
#include "hpipm_d_ocp_qp_dim.h"
#include "hpipm_d_ocp_qp_sol.h"

namespace laopt
{

class HPIPMSolver : public QPBase<HPIPMSolver, double>
{
public:
    using Base = QPBase<HPIPMSolver, double>;
    using scalar_t = typename Base::scalar_t;

private:
    using constraint_t = typename Base::constraint_t;
    using constraint_changed_t = typename Base::constraint_changed_t;

    Eigen::VectorX<scalar_t> m_b;
    Eigen::VectorX<scalar_t> m_tmp_lam_lb; // temporary vector to extract dual lb solutions
    Eigen::VectorX<scalar_t> m_tmp_lam_ub; // temporary vector to extract dual ub solutions

    int m_N{0}, m_nx{0}, m_nu{0}, m_ng{0}, m_ng0{0}, m_ngf{0};
    d_ocp_qp_dim m_dim{};
    std::unique_ptr<char[]> m_dim_memory;
    d_ocp_qp m_qp{};
    std::unique_ptr<char[]> m_qp_memory;
    d_ocp_qp_sol m_qp_sol{};
    std::unique_ptr<char[]> m_qp_sol_memory;
    d_ocp_qp_ipm_arg m_qp_ipm_arg{};
    std::unique_ptr<char[]> m_qp_ipm_arg_memory;
    d_ocp_qp_ipm_ws m_qp_ipm_ws{};
    std::unique_ptr<char[]> m_qp_ipm_ws_memory;
    bool m_hpipm_initialized;

public:
    HPIPMSolver(int n, int m) :
        Base(n, m),
        m_hpipm_initialized(false)
    {
        this->m_settings.max_iter = 50;
    }

    void set_problem_dims(int N, int nx, int nu, int ng, int ng0, int ngf)
    {
        m_N = N;
        m_nx = nx;
        m_nu = nu;
        m_ng = ng;
        m_ng0 = ng0;
        m_ngf = ngf;

        m_b.resize(N * nx);
        m_tmp_lam_lb.resize(std::max({nx, nu, ng0, ng, ngf}));
        m_tmp_lam_ub.resize(std::max({nx, nu, ng0, ng, ngf}));

        hpipm_size_t dim_size = d_ocp_qp_dim_memsize(N);
        m_dim_memory = std::unique_ptr<char[]>(new char[dim_size]);
        d_ocp_qp_dim_create(N, &m_dim, m_dim_memory.get());

        for (int i = 0; i <= N; i++) {
            d_ocp_qp_dim_set_nx(i, nx, &m_dim);
            d_ocp_qp_dim_set_nu(i, nu, &m_dim);
        }
    }

    qp_solver_info_t solve_impl(const Eigen::BlockSparseMatrix<scalar_t>& H,
                                const Eigen::Ref<const Eigen::VectorX<scalar_t>>& f,
                                const Eigen::Ref<const Eigen::VectorX<scalar_t>>& xlb,
                                const Eigen::Ref<const Eigen::VectorX<scalar_t>>& xub,
                                const Eigen::BlockSparseMatrix<scalar_t>& A,
                                const Eigen::Ref<const Eigen::VectorX<scalar_t>>& Alb,
                                const Eigen::Ref<const Eigen::VectorX<scalar_t>>& Aub) noexcept
    {
        set_hpipm_data(H, f, xlb, xub, A, Alb, Aub);

        if (!this->settings().reuse_pattern || !m_hpipm_initialized)
        {
            hpipm_size_t qp_ipm_arg_size = d_ocp_qp_ipm_arg_memsize(&m_dim);
            m_qp_ipm_arg_memory = std::unique_ptr<char[]>(new char[qp_ipm_arg_size]);
            d_ocp_qp_ipm_arg_create(&m_dim, &m_qp_ipm_arg, m_qp_ipm_arg_memory.get());
            d_ocp_qp_ipm_arg_set_default(BALANCE, &m_qp_ipm_arg);
        }

        set_hpipm_settings();

        if (!this->settings().reuse_pattern ||
            !m_hpipm_initialized ||
            m_qp_ipm_arg.iter_max > m_qp_ipm_arg.stat_max)
        {
            hpipm_size_t qp_ipm_ws_size = d_ocp_qp_ipm_ws_memsize(&m_dim, &m_qp_ipm_arg);
            m_qp_ipm_ws_memory = std::unique_ptr<char[]>(new char[qp_ipm_ws_size]);
            d_ocp_qp_ipm_ws_create(&m_dim, &m_qp_ipm_arg, &m_qp_ipm_ws, m_qp_ipm_ws_memory.get());
        }
        m_hpipm_initialized = true;

        // zero primal solution
        int ii;
        int N = m_qp.dim->N;
        int *nx = m_qp.dim->nx;
        int *nu = m_qp.dim->nu;
        int *ns = m_qp.dim->ns;
        for(ii=0; ii <= N; ii++)
        {
            blasfeo_dvecse(nu[ii] + nx[ii] + 2 * ns[ii], 0.0, m_qp_sol.ux + ii, 0);
        }

        int hpipm_status;
        d_ocp_qp_ipm_solve(&m_qp, &m_qp_sol, &m_qp_ipm_arg, &m_qp_ipm_ws);

        d_ocp_qp_ipm_get_status(&m_qp_ipm_ws, &hpipm_status);
        d_ocp_qp_ipm_get_iter(&m_qp_ipm_ws, &this->m_info.iter);

        extract_hpipm_sol();

        // update status
        switch (hpipm_status)
        {
            case SUCCESS:
                this->m_info.status = qp_status_t::SOLVED;
                break;
            case MAX_ITER:
                this->m_info.status = qp_status_t::MAX_ITER_REACHED;
                break;
            case MIN_STEP:
                this->m_info.status = qp_status_t::MIN_STEP;
                break;
            default:
                this->m_info.status = qp_status_t::UNSOLVED;
                break;
        }

        return this->m_info;
    }

private:
    EIGEN_STRONG_INLINE void
    set_hpipm_data(const Eigen::BlockSparseMatrix<scalar_t>& H,
                   const Eigen::Ref<const Eigen::VectorX<scalar_t>>& f,
                   const Eigen::Ref<const Eigen::VectorX<scalar_t>>& xlb,
                   const Eigen::Ref<const Eigen::VectorX<scalar_t>>& xub,
                   const Eigen::BlockSparseMatrix<scalar_t>& A,
                   const Eigen::Ref<const Eigen::VectorX<scalar_t>>& Alb,
                   const Eigen::Ref<const Eigen::VectorX<scalar_t>>& Aub) noexcept
    {
        constraint_changed_t constraints_type_change = this->parse_constraints_bounds(xlb, xub, Alb, Aub);
        if (constraints_type_change != constraint_changed_t::NO_CHANGE)
        {
            // if the types of the constraints changed
            // we have to reinitialize solver because of sparsity pattern change
            m_hpipm_initialized = false;
        }

        if (!this->settings().reuse_pattern || !m_hpipm_initialized) {
            set_hpipm_dim();
            setup_hpipm_qp();
        }
        copy_hpipm_data(H, f, xlb, xub, A, Alb, Aub);
    }

    EIGEN_STRONG_INLINE void set_hpipm_dim() noexcept
    {
        int offset = 0;
        int i;
        for (i = 0; i < m_N; i++) {
            // nx box constraints
            int nbx = 0;
            int nbxe = 0;
            for (int j = 0; j < m_nx; j++) {
                if (m_box_constraint_type[offset + j] != constraint_t::UNBOUNDED_CONSTR) {
                    nbx++;
                    if (m_box_constraint_type[offset + j] == constraint_t::EQ_CONSTR) {
                        nbxe++;
                    }
                }
            }
            offset += m_nx;
            d_ocp_qp_dim_set_nbx(i, nbx, &m_dim);
            d_ocp_qp_dim_set_nbxe(i, nbxe, &m_dim);

            // nu box constraints
            int nbu = 0;
            int nbue = 0;
            for (int j = 0; j < m_nu; j++) {
                if (m_box_constraint_type[offset + j] != constraint_t::UNBOUNDED_CONSTR) {
                    nbu++;
                    if (m_box_constraint_type[offset + j] == constraint_t::EQ_CONSTR) {
                        nbue++;
                    }
                }
            }
            offset += m_nu;
            d_ocp_qp_dim_set_nbu(i, nbu, &m_dim);
            d_ocp_qp_dim_set_nbue(i, nbue, &m_dim);
        }

        // final nx box constraints
        i = m_N;
        int nbx = 0;
        int nbxe = 0;
        for (int j = 0; j < m_nx; j++) {
            if (m_box_constraint_type[offset + j] != constraint_t::UNBOUNDED_CONSTR) {
                nbx++;
                if (m_box_constraint_type[offset + j] == constraint_t::EQ_CONSTR) {
                    nbxe++;
                }
            }
        }
        offset += m_nx;
        d_ocp_qp_dim_set_nbx(i, nbx, &m_dim);
        d_ocp_qp_dim_set_nbxe(i, nbxe, &m_dim);

        // ng0 constraints
        offset = m_N * m_nx;
        i = 0;
        int ng = 0;
        int nge = 0;
        for (int j = 0; j < m_ng0; j++) {
            ng++;
            if (m_constraint_type[offset + j] == constraint_t::EQ_CONSTR) {
                nge++;
            }
        }
        offset += m_ng0;
        d_ocp_qp_dim_set_ng(i, ng, &m_dim);
        d_ocp_qp_dim_set_nge(i, nge, &m_dim);

        // ng constraints
        for (i = 1; i < m_N; i++) {
            ng = 0;
            nge = 0;
            for (int j = 0; j < m_ng; j++) {
                ng++;
                if (m_constraint_type[offset + j] == constraint_t::EQ_CONSTR) {
                    nge++;
                }
            }
            offset += m_ng;
            d_ocp_qp_dim_set_ng(i, ng, &m_dim);
            d_ocp_qp_dim_set_nge(i, nge, &m_dim);
        }

        // ngf constraints
        i = m_N;
        ng = 0;
        nge = 0;
        for (int j = 0; j < m_ngf; j++) {
            ng++;
            if (m_constraint_type[offset + j] == constraint_t::EQ_CONSTR) {
                nge++;
            }
        }
        offset += m_ngf;
        d_ocp_qp_dim_set_ng(i, ng, &m_dim);
        d_ocp_qp_dim_set_nge(i, nge, &m_dim);
    }

    EIGEN_STRONG_INLINE void setup_hpipm_qp() noexcept
    {
        hpipm_size_t qp_size = d_ocp_qp_memsize(&m_dim);
        m_qp_memory = std::unique_ptr<char[]>(new char[qp_size]);
        d_ocp_qp_create(&m_dim, &m_qp, m_qp_memory.get());

        hpipm_size_t qp_sol_size = d_ocp_qp_sol_memsize(&m_dim);
        m_qp_sol_memory = std::unique_ptr<char[]>(new char[qp_sol_size]);
        d_ocp_qp_sol_create(&m_dim, &m_qp_sol, m_qp_sol_memory.get());

        int* nu = m_qp.dim->nu;
        int* nb = m_qp.dim->nb;
        int* nbx = m_qp.dim->nbx;
        int* nbu = m_qp.dim->nbu;
        int* nbxe = m_qp.dim->nbxe;
        int* nbue = m_qp.dim->nbue;
        int* ng = m_qp.dim->ng;

        // see hpipm_d_ocp_qp.h for explanation of data format:
        // struct d_ocp_qp { ...
        //   // indices of box constrained variables within [u; x]
        //   int **idxb;
        //   // indices of constraints within [bu, bx, g] that are equalities, subset of [0, ..., nbu+nbx+ng-1]
        //   int **idxe;
        // ... }

        int offset = 0;
        int i;
        for (i = 0; i < m_N; i++) {
            // nx box constraints
            int ix = 0;
            int ixe = 0;
            for (int j = 0; j < m_nx; j++) {
                if (m_box_constraint_type[offset + j] != constraint_t::UNBOUNDED_CONSTR) {
                    // d_ocp_qp_set_idxbx()
                    m_qp.idxb[i][nbu[i] + ix] = nu[i] + j;
                    if (m_box_constraint_type[offset + j] == constraint_t::EQ_CONSTR) {
                        // d_ocp_qp_set_idxbxe()
                        m_qp.idxe[i][nbue[i] + ixe++] = nbu[i] + ix;
                        // d_ocp_qp_set_lbx_mask()
                        m_qp.d_mask[i].pa[nbu[i] + ix] = 1.0;
                        // d_ocp_qp_set_ubx_mask()
                        m_qp.d_mask[i].pa[nb[i] + ng[i] + nbu[i] + ix] = 1.0;
                    } else if (m_box_constraint_type[offset + j] == constraint_t::INEQ_CONSTR) {
                        // d_ocp_qp_set_lbx_mask()
                        m_qp.d_mask[i].pa[nbu[i] + ix] = 1.0;
                        // d_ocp_qp_set_ubx_mask()
                        m_qp.d_mask[i].pa[nb[i] + ng[i] + nbu[i] + ix] = 1.0;
                    } else if (m_box_constraint_type[offset + j] == constraint_t::INEQ_LB_ONLY_CONSTR) {
                        // d_ocp_qp_set_lbx_mask()
                        m_qp.d_mask[i].pa[nbu[i] + ix] = 1.0;
                        // d_ocp_qp_set_ubx_mask()
                        m_qp.d_mask[i].pa[nb[i] + ng[i] + nbu[i] + ix] = 0.0;
                    } else if (m_box_constraint_type[offset + j] == constraint_t::INEQ_UB_ONLY_CONSTR) {
                        // d_ocp_qp_set_lbx_mask()
                        m_qp.d_mask[i].pa[nbu[i] + ix] = 0.0;
                        // d_ocp_qp_set_ubx_mask()
                        m_qp.d_mask[i].pa[nb[i] + ng[i] + nbu[i] + ix] = 1.0;
                    }
                    ix++;
                }
            }
            offset += m_nx;

            // nu box constraints
            int iu = 0;
            int iue = 0;
            for (int j = 0; j < m_nu; j++) {
                if (m_box_constraint_type[offset + j] != constraint_t::UNBOUNDED_CONSTR) {
                    // d_ocp_qp_set_idxbu()
                    m_qp.idxb[i][iu] = j;
                    if (m_box_constraint_type[offset + j] == constraint_t::EQ_CONSTR) {
                        // d_ocp_qp_set_idxbue()
                        m_qp.idxe[i][iue++] = iu;
                        // d_ocp_qp_set_lbu_mask()
                        m_qp.d_mask[i].pa[iu] = 1.0;
                        // d_ocp_qp_set_ubu_mask()
                        m_qp.d_mask[i].pa[nb[i] + ng[i] + iu] = 1.0;
                    } else if (m_box_constraint_type[offset + j] == constraint_t::INEQ_CONSTR) {
                        // d_ocp_qp_set_lbu_mask()
                        m_qp.d_mask[i].pa[iu] = 1.0;
                        // d_ocp_qp_set_ubu_mask()
                        m_qp.d_mask[i].pa[nb[i] + ng[i] + iu] = 1.0;
                    } else if (m_box_constraint_type[offset + j] == constraint_t::INEQ_LB_ONLY_CONSTR) {
                        // d_ocp_qp_set_lbu_mask()
                        m_qp.d_mask[i].pa[iu] = 1.0;
                        // d_ocp_qp_set_ubu_mask()
                        m_qp.d_mask[i].pa[nb[i] + ng[i] + iu] = 0.0;
                    } else if (m_box_constraint_type[offset + j] == constraint_t::INEQ_UB_ONLY_CONSTR) {
                        // d_ocp_qp_set_lbu_mask()
                        m_qp.d_mask[i].pa[iu] = 0.0;
                        // d_ocp_qp_set_ubu_mask()
                        m_qp.d_mask[i].pa[nb[i] + ng[i] + iu] = 1.0;
                    }
                    iu++;
                }
            }
            offset += m_nu;
        }

        // final nx box constraints
        i = m_N;
        int ix = 0;
        int ixe = 0;
        for (int j = 0; j < m_nx; j++) {
            if (m_box_constraint_type[offset + j] != constraint_t::UNBOUNDED_CONSTR) {
                // d_ocp_qp_set_idxbx()
                m_qp.idxb[i][nbu[i] + ix] = nu[i] + j;
                if (m_box_constraint_type[offset + j] == constraint_t::EQ_CONSTR) {
                    // d_ocp_qp_set_idxbxe()
                    m_qp.idxe[i][nbue[i] + ixe++] = nbu[i] + ix;
                    // d_ocp_qp_set_lbx_mask()
                    m_qp.d_mask[i].pa[nbu[i] + ix] = 1.0;
                    // d_ocp_qp_set_ubx_mask()
                    m_qp.d_mask[i].pa[nb[i] + ng[i] + nbu[i] + ix] = 1.0;
                } else if (m_box_constraint_type[offset + j] == constraint_t::INEQ_CONSTR) {
                    // d_ocp_qp_set_lbx_mask()
                    m_qp.d_mask[i].pa[nbu[i] + ix] = 1.0;
                    // d_ocp_qp_set_ubx_mask()
                    m_qp.d_mask[i].pa[nb[i] + ng[i] + nbu[i] + ix] = 1.0;
                } else if (m_box_constraint_type[offset + j] == constraint_t::INEQ_LB_ONLY_CONSTR) {
                    // d_ocp_qp_set_lbx_mask()
                    m_qp.d_mask[i].pa[nbu[i] + ix] = 1.0;
                    // d_ocp_qp_set_ubx_mask()
                    m_qp.d_mask[i].pa[nb[i] + ng[i] + nbu[i] + ix] = 0.0;
                } else if (m_box_constraint_type[offset + j] == constraint_t::INEQ_UB_ONLY_CONSTR) {
                    // d_ocp_qp_set_lbx_mask()
                    m_qp.d_mask[i].pa[nbu[i] + ix] = 0.0;
                    // d_ocp_qp_set_ubx_mask()
                    m_qp.d_mask[i].pa[nb[i] + ng[i] + nbu[i] + ix] = 1.0;
                }
                ix++;
            }
        }
        offset += m_nx;

        // ng0 constraints
        offset = m_N * m_nx;
        i = 0;
        int ig0 = 0;
        int ig0e = 0;
        for (int j = 0; j < m_ng0; j++) {
            if (m_constraint_type[offset + j] == constraint_t::EQ_CONSTR) {
                // d_ocp_qp_set_idxbxe()
                m_qp.idxe[i][nbue[i] + nbxe[i] + ig0e++] = nbu[i] + nbx[i] + ig0;
                // d_ocp_qp_set_lg_mask()
                m_qp.d_mask[i].pa[nb[i] + ig0] = 1.0;
                // d_ocp_qp_set_ug_mask()
                m_qp.d_mask[i].pa[2 * nb[i] + ng[i] + ig0] = 1.0;
            } else if (m_constraint_type[offset + j] == constraint_t::INEQ_CONSTR) {
                // d_ocp_qp_set_lg_mask()
                m_qp.d_mask[i].pa[nb[i] + ig0] = 1.0;
                // d_ocp_qp_set_ug_mask()
                m_qp.d_mask[i].pa[2 * nb[i] + ng[i] + ig0] = 1.0;
            } else if (m_constraint_type[offset + j] == constraint_t::INEQ_LB_ONLY_CONSTR) {
                // d_ocp_qp_set_lg_mask()
                m_qp.d_mask[i].pa[nb[i] + ig0] = 1.0;
                // d_ocp_qp_set_ug_mask()
                m_qp.d_mask[i].pa[2 * nb[i] + ng[i] + ig0] = 0.0;
            } else if (m_constraint_type[offset + j] == constraint_t::INEQ_UB_ONLY_CONSTR) {
                // d_ocp_qp_set_lg_mask()
                m_qp.d_mask[i].pa[nb[i] + ig0] = 0.0;
                // d_ocp_qp_set_ug_mask()
                m_qp.d_mask[i].pa[2 * nb[i] + ng[i] + ig0] = 1.0;
            } else {
                // d_ocp_qp_set_lg_mask()
                m_qp.d_mask[i].pa[nb[i] + ig0] = 0.0;
                // d_ocp_qp_set_ug_mask()
                m_qp.d_mask[i].pa[2 * nb[i] + ng[i] + ig0] = 0.0;
            }
            ig0++;
        }
        offset += m_ng0;

        // ng constraints
        for (i = 1; i < m_N; i++) {
            int ig = 0;
            int ige = 0;
            for (int j = 0; j < m_ng; j++) {
                if (m_constraint_type[offset + j] == constraint_t::EQ_CONSTR) {
                    // d_ocp_qp_set_idxbxe()
                    m_qp.idxe[i][nbue[i] + nbxe[i] + ige++] = nbu[i] + nbx[i] + ig;
                    // d_ocp_qp_set_lg_mask()
                    m_qp.d_mask[i].pa[nb[i] + ig] = 1.0;
                    // d_ocp_qp_set_ug_mask()
                    m_qp.d_mask[i].pa[2 * nb[i] + ng[i] + ig] = 1.0;
                } else if (m_constraint_type[offset + j] == constraint_t::INEQ_CONSTR) {
                    // d_ocp_qp_set_lg_mask()
                    m_qp.d_mask[i].pa[nb[i] + ig] = 1.0;
                    // d_ocp_qp_set_ug_mask()
                    m_qp.d_mask[i].pa[2 * nb[i] + ng[i] + ig] = 1.0;
                } else if (m_constraint_type[offset + j] == constraint_t::INEQ_LB_ONLY_CONSTR) {
                    // d_ocp_qp_set_lg_mask()
                    m_qp.d_mask[i].pa[nb[i] + ig] = 1.0;
                    // d_ocp_qp_set_ug_mask()
                    m_qp.d_mask[i].pa[2 * nb[i] + ng[i] + ig] = 0.0;
                } else if (m_constraint_type[offset + j] == constraint_t::INEQ_UB_ONLY_CONSTR) {
                    // d_ocp_qp_set_lg_mask()
                    m_qp.d_mask[i].pa[nb[i] + ig] = 0.0;
                    // d_ocp_qp_set_ug_mask()
                    m_qp.d_mask[i].pa[2 * nb[i] + ng[i] + ig] = 1.0;
                } else {
                    // d_ocp_qp_set_lg_mask()
                    m_qp.d_mask[i].pa[nb[i] + ig] = 0.0;
                    // d_ocp_qp_set_ug_mask()
                    m_qp.d_mask[i].pa[2 * nb[i] + ng[i] + ig] = 0.0;
                }
                ig++;
            }
            offset += m_ng;
        }

        // ngf constraints
        i = m_N;
        int igf = 0;
        int igfe = 0;
        for (int j = 0; j < m_ngf; j++) {
            if (m_constraint_type[offset + j] == constraint_t::EQ_CONSTR) {
                // d_ocp_qp_set_idxbxe()
                m_qp.idxe[i][nbue[i] + nbxe[i] + igfe++] = nbu[i] + nbx[i] + igf;
                // d_ocp_qp_set_lg_mask()
                m_qp.d_mask[i].pa[nb[i] + igf] = 1.0;
                // d_ocp_qp_set_ug_mask()
                m_qp.d_mask[i].pa[2 * nb[i] + ng[i] + igf] = 1.0;
            } else if (m_constraint_type[offset + j] == constraint_t::INEQ_CONSTR) {
                // d_ocp_qp_set_lg_mask()
                m_qp.d_mask[i].pa[nb[i] + igf] = 1.0;
                // d_ocp_qp_set_ug_mask()
                m_qp.d_mask[i].pa[2 * nb[i] + ng[i] + igf] = 1.0;
            } else if (m_constraint_type[offset + j] == constraint_t::INEQ_LB_ONLY_CONSTR) {
                // d_ocp_qp_set_lg_mask()
                m_qp.d_mask[i].pa[nb[i] + igf] = 1.0;
                // d_ocp_qp_set_ug_mask()
                m_qp.d_mask[i].pa[2 * nb[i] + ng[i] + igf] = 0.0;
            } else if (m_constraint_type[offset + j] == constraint_t::INEQ_UB_ONLY_CONSTR) {
                // d_ocp_qp_set_lg_mask()
                m_qp.d_mask[i].pa[nb[i] + igf] = 0.0;
                // d_ocp_qp_set_ug_mask()
                m_qp.d_mask[i].pa[2 * nb[i] + ng[i] + igf] = 1.0;
            } else {
                // d_ocp_qp_set_lg_mask()
                m_qp.d_mask[i].pa[nb[i] + igf] = 0.0;
                // d_ocp_qp_set_ug_mask()
                m_qp.d_mask[i].pa[2 * nb[i] + ng[i] + igf] = 0.0;
            }
            igf++;
        }
        offset += m_ngf;
    }

    EIGEN_STRONG_INLINE void copy_hpipm_data(const Eigen::BlockSparseMatrix<scalar_t>& H,
                                             const Eigen::Ref<const Eigen::VectorX<scalar_t>>& f,
                                             const Eigen::Ref<const Eigen::VectorX<scalar_t>>& xlb,
                                             const Eigen::Ref<const Eigen::VectorX<scalar_t>>& xub,
                                             const Eigen::BlockSparseMatrix<scalar_t>& A,
                                             const Eigen::Ref<const Eigen::VectorX<scalar_t>>& Alb,
                                             const Eigen::Ref<const Eigen::VectorX<scalar_t>>& Aub) noexcept
    {
        m_b = -Alb.head(m_N * m_nx);

        int* nu = m_qp.dim->nu;
        int* nb = m_qp.dim->nb;
        int* nbx = m_qp.dim->nbx;
        int* nbu = m_qp.dim->nbu;
        int* ng = m_qp.dim->ng;

        int offset = 0;
        int i;
        for (i = 0; i < m_N; i++) {
            // dynamics
            double* Ai = const_cast<double*>(A.coeff(i, 2 * i).data());
            double* Bi = const_cast<double*>(A.coeff(i, 2 * i + 1).data());
            double* bi = m_b.data() + m_nx * i;
            d_ocp_qp_set_A(i, Ai, &m_qp);
            d_ocp_qp_set_B(i, Bi, &m_qp);
            d_ocp_qp_set_b(i, bi, &m_qp);

            // cost
            double* Qi = const_cast<double*>(H.coeff(2 * i, 2 * i).data());
            double* Si = const_cast<double*>(H.coeff(2 * i, 2 * i + 1).data());
            double* Ri = const_cast<double*>(H.coeff(2 * i + 1, 2 * i + 1).data());
            double* qi = const_cast<double*>(f.data()) + offset;
            double* ri = const_cast<double*>(f.data()) + offset + m_nx;
            d_ocp_qp_set_Q(i, Qi, &m_qp);
            d_ocp_qp_set_S(i, Si, &m_qp);
            d_ocp_qp_set_R(i, Ri, &m_qp);
            d_ocp_qp_set_q(i, qi, &m_qp);
            d_ocp_qp_set_r(i, ri, &m_qp);

            // nx box constraints
            for (int j = 0; j < nbx[i]; j++) {
                // inverse of d_ocp_qp_set_idxbx()
                int idxbx = offset + m_qp.idxb[i][nbu[i] + j] - nu[i];
                // d_ocp_qp_set_lbx()
                m_qp.d[i].pa[nbu[i] + j] = xlb[idxbx];
                // d_ocp_qp_set_ubx()
                m_qp.d[i].pa[nb[i] + ng[i] + nbu[i] + j] = -xub[idxbx];
            }
            offset += m_nx;

            // nu box constraints
            for (int j = 0; j < nbu[i]; j++) {
                // inverse of d_ocp_qp_set_idxbu()
                int idxbu = offset + m_qp.idxb[i][j];
                // d_ocp_qp_set_lbu()
                m_qp.d[i].pa[j] = xlb[idxbu];
                // d_ocp_qp_set_ubu()
                m_qp.d[i].pa[nb[i] + ng[i] + j] = -xub[idxbu];
            }
            offset += m_nu;
        }
        // final cost
        i = m_N;
        double* QN = const_cast<double*>(H.coeff(2 * i, 2 * i).data());
        double* qN = const_cast<double*>(f.data()) + offset;
        d_ocp_qp_set_Q(i, QN, &m_qp);
        d_ocp_qp_set_q(i, qN, &m_qp);

        // final nx box constraints
        for (int j = 0; j < nbx[i]; j++) {
            // inverse of d_ocp_qp_set_idxbx()
            int idxbx = offset + m_qp.idxb[i][nbu[i] + j] - nu[i];
            // d_ocp_qp_set_lbx()
            m_qp.d[i].pa[nbu[i] + j] = xlb[idxbx];
            // d_ocp_qp_set_ubx()
            m_qp.d[i].pa[nb[i] + ng[i] + nbu[i] + j] = -xub[idxbx];
        }
        offset += m_nx;

        // ng0 constraints
        offset = m_N * m_nx;
        int block_offset = m_N;
        if (m_ng0 > 0) {
            i = 0;
            double* C0 = const_cast<double*>(A.coeff(block_offset, 2 * i).data());
            double* D0 = const_cast<double*>(A.coeff(block_offset, 2 * i + 1).data());
            d_ocp_qp_set_C(0, C0, &m_qp);
            d_ocp_qp_set_D(0, D0, &m_qp);

            for (int j = 0; j < m_ng0; j++) {
                // d_ocp_qp_set_lg()
                m_qp.d[i].pa[nb[i] + j] = Alb[offset + j];
                // d_ocp_qp_set_ug()
                m_qp.d[i].pa[2 * nb[i] + ng[i] + j] = -Aub[offset + j];
            }
            offset += m_ng0;
            block_offset++;
        }

        if (m_ng > 0) {
            for (i = 1; i < m_N; i++) {
                double* Ci = const_cast<double*>(A.coeff(block_offset, 2 * i).data());
                double* Di = const_cast<double*>(A.coeff(block_offset, 2 * i + 1).data());
                d_ocp_qp_set_C(i, Ci, &m_qp);
                d_ocp_qp_set_D(i, Di, &m_qp);

                for (int j = 0; j < m_ng; j++) {
                    // d_ocp_qp_set_lg()
                    m_qp.d[i].pa[nb[i] + j] = Alb[offset + j];
                    // d_ocp_qp_set_ug()
                    m_qp.d[i].pa[2 * nb[i] + ng[i] + j] = -Aub[offset + j];
                }
                offset += m_ng;
                block_offset++;
            }
        }

        if (m_ngf > 0) {
            i = m_N;
            double* CN = const_cast<double*>(A.coeff(block_offset, 2 * i).data());
            double* DN = const_cast<double*>(A.coeff(block_offset, 2 * i + 1).data());
            d_ocp_qp_set_C(i, CN, &m_qp);
            d_ocp_qp_set_D(i, DN, &m_qp);

            for (int j = 0; j < m_ngf; j++) {
                // d_ocp_qp_set_lg()
                m_qp.d[i].pa[nb[i] + j] = Alb[offset + j];
                // d_ocp_qp_set_ug()
                m_qp.d[i].pa[2 * nb[i] + ng[i] + j] = -Aub[offset + j];
            }
            offset += m_ngf;
            block_offset++;
        }
    }

    EIGEN_STRONG_INLINE void extract_hpipm_sol() noexcept
    {
        // extract primal solution
        int offset = 0;
        for (int i = 0; i < m_N; i++) {
            d_ocp_qp_sol_get_x(i, &m_qp_sol, this->m_x.data() + offset);
            offset += m_nx;
            d_ocp_qp_sol_get_u(i, &m_qp_sol, this->m_x.data() + offset);
            offset += m_nu;
        }
        d_ocp_qp_sol_get_x(m_N, &m_qp_sol, this->m_x.data() + offset);

        // extract dynamics eq dual variables
        offset = 0;
        for (int i = 0; i < m_N; i++) {
            d_ocp_qp_sol_get_pi(i, &m_qp_sol, this->m_lam.data() + offset);
            offset += m_nx;
        }

        // extract box dual variables
        offset = 0;
        int i;
        for (i  = 0; i < m_N; i++) {
            // nx box constraints
            d_ocp_qp_sol_get_lam_lbx(i, &m_qp_sol, m_tmp_lam_lb.data());
            d_ocp_qp_sol_get_lam_ubx(i, &m_qp_sol, m_tmp_lam_ub.data());
            int bx_offset = 0;
            for (int j = 0; j < m_nx; j++) {
                if (this->m_box_constraint_type[offset + j] == constraint_t::EQ_CONSTR ||
                    this->m_box_constraint_type[offset + j] == constraint_t::INEQ_CONSTR)
                {
                    this->m_lam_bounds(offset + j) = m_tmp_lam_ub(bx_offset) - m_tmp_lam_lb(bx_offset);
                    bx_offset++;
                }
                else if (this->m_box_constraint_type[offset + j] == constraint_t::INEQ_LB_ONLY_CONSTR)
                {
                    this->m_lam_bounds(offset + j) = -m_tmp_lam_lb(bx_offset++);
                }
                else if (this->m_box_constraint_type[offset + j] == constraint_t::INEQ_UB_ONLY_CONSTR)
                {
                    this->m_lam_bounds(offset + j) = m_tmp_lam_ub(bx_offset++);
                }
                else
                {
                    this->m_lam_bounds(offset + j) = 0;
                }
            }
            offset += m_nx;

            // nu box constraints
            d_ocp_qp_sol_get_lam_lbu(i, &m_qp_sol, m_tmp_lam_lb.data());
            d_ocp_qp_sol_get_lam_ubu(i, &m_qp_sol, m_tmp_lam_ub.data());
            int bu_offset = 0;
            for (int j = 0; j < m_nu; j++) {
                if (this->m_box_constraint_type[offset + j] == constraint_t::EQ_CONSTR ||
                    this->m_box_constraint_type[offset + j] == constraint_t::INEQ_CONSTR)
                {
                    this->m_lam_bounds(offset + j) = m_tmp_lam_ub(bu_offset) - m_tmp_lam_lb(bu_offset);
                    bu_offset++;
                }
                else if (this->m_box_constraint_type[offset + j] == constraint_t::INEQ_LB_ONLY_CONSTR)
                {
                    this->m_lam_bounds(offset + j) = -m_tmp_lam_lb(bu_offset++);
                }
                else if (this->m_box_constraint_type[offset + j] == constraint_t::INEQ_UB_ONLY_CONSTR)
                {
                    this->m_lam_bounds(offset + j) = m_tmp_lam_ub(bu_offset++);
                }
                else
                {
                    this->m_lam_bounds(offset + j) = 0;
                }
            }
            offset += m_nu;
        }
        // final nx constraints
        i = m_N;
        d_ocp_qp_sol_get_lam_lbx(i, &m_qp_sol, m_tmp_lam_lb.data());
        d_ocp_qp_sol_get_lam_ubx(i, &m_qp_sol, m_tmp_lam_ub.data());
        int bx_offset = 0;
        for (int j = 0; j < m_nx; j++) {
            if (this->m_box_constraint_type[offset + j] == constraint_t::EQ_CONSTR ||
                this->m_box_constraint_type[offset + j] == constraint_t::INEQ_CONSTR)
            {
                this->m_lam_bounds(offset + j) = m_tmp_lam_ub(bx_offset) - m_tmp_lam_lb(bx_offset);
                bx_offset++;
            }
            else if (this->m_box_constraint_type[offset + j] == constraint_t::INEQ_LB_ONLY_CONSTR)
            {
                this->m_lam_bounds(offset + j) = -m_tmp_lam_lb(bx_offset++);
            }
            else if (this->m_box_constraint_type[offset + j] == constraint_t::INEQ_UB_ONLY_CONSTR)
            {
                this->m_lam_bounds(offset + j) = m_tmp_lam_ub(bx_offset++);
            }
            else
            {
                this->m_lam_bounds(offset + j) = 0;
            }
        }
        offset += m_nx;

        // extract generic ineq dual variables
        // ng0 constraints
        offset = m_N * m_nx;
        i = 0;
        d_ocp_qp_sol_get_lam_lg(i, &m_qp_sol, m_tmp_lam_lb.data());
        d_ocp_qp_sol_get_lam_ug(i, &m_qp_sol, m_tmp_lam_ub.data());
        for (int j = 0; j < m_ng0; j++) {
            if (this->m_constraint_type[offset + j] == constraint_t::EQ_CONSTR ||
                this->m_constraint_type[offset + j] == constraint_t::INEQ_CONSTR)
            {
                this->m_lam(offset + j) = m_tmp_lam_ub(j) - m_tmp_lam_lb(j);
            }
            else if (this->m_constraint_type[offset + j] == constraint_t::INEQ_LB_ONLY_CONSTR)
            {
                this->m_lam(offset + j) = -m_tmp_lam_lb(j);
            }
            else if (this->m_constraint_type[offset + j] == constraint_t::INEQ_UB_ONLY_CONSTR)
            {
                this->m_lam(offset + j) = m_tmp_lam_ub(j);
            }
            else
            {
                this->m_lam(offset + j) = 0;
            }
        }
        offset += m_ng0;

        // ng constraints
        for (i = 1; i < m_N; i++) {
            d_ocp_qp_sol_get_lam_lg(i, &m_qp_sol, m_tmp_lam_lb.data());
            d_ocp_qp_sol_get_lam_ug(i, &m_qp_sol, m_tmp_lam_ub.data());
            for (int j = 0; j < m_ng; j++) {
                if (this->m_constraint_type[offset + j] == constraint_t::EQ_CONSTR ||
                    this->m_constraint_type[offset + j] == constraint_t::INEQ_CONSTR)
                {
                    this->m_lam(offset + j) = m_tmp_lam_ub(j) - m_tmp_lam_lb(j);
                }
                else if (this->m_constraint_type[offset + j] == constraint_t::INEQ_LB_ONLY_CONSTR)
                {
                    this->m_lam(offset + j) = -m_tmp_lam_lb(j);
                }
                else if (this->m_constraint_type[offset + j] == constraint_t::INEQ_UB_ONLY_CONSTR)
                {
                    this->m_lam(offset + j) = m_tmp_lam_ub(j);
                }
                else
                {
                    this->m_lam(offset + j) = 0;
                }
            }
            offset += m_ng;
        }

        // ngf constraints
        i = m_N;
        d_ocp_qp_sol_get_lam_lg(i, &m_qp_sol, m_tmp_lam_lb.data());
        d_ocp_qp_sol_get_lam_ug(i, &m_qp_sol, m_tmp_lam_ub.data());
        for (int j = 0; j < m_ngf; j++) {
            if (this->m_constraint_type[offset + j] == constraint_t::EQ_CONSTR ||
                this->m_constraint_type[offset + j] == constraint_t::INEQ_CONSTR)
            {
                this->m_lam(offset + j) = m_tmp_lam_ub(j) - m_tmp_lam_lb(j);
            }
            else if (this->m_constraint_type[offset + j] == constraint_t::INEQ_LB_ONLY_CONSTR)
            {
                this->m_lam(offset + j) = -m_tmp_lam_lb(j);
            }
            else if (this->m_constraint_type[offset + j] == constraint_t::INEQ_UB_ONLY_CONSTR)
            {
                this->m_lam(offset + j) = m_tmp_lam_ub(j);
            }
            else
            {
                this->m_lam(offset + j) = 0;
            }
        }
        offset += m_ngf;
    }

    void set_hpipm_settings() noexcept
    {
        d_ocp_qp_ipm_arg_set_tol_stat(&this->m_settings.eps_abs, &m_qp_ipm_arg);
        d_ocp_qp_ipm_arg_set_tol_eq(&this->m_settings.eps_abs, &m_qp_ipm_arg);
        d_ocp_qp_ipm_arg_set_tol_ineq(&this->m_settings.eps_abs, &m_qp_ipm_arg);
        d_ocp_qp_ipm_arg_set_tol_comp(&this->m_settings.eps_abs, &m_qp_ipm_arg);
        d_ocp_qp_ipm_arg_set_iter_max(&this->m_settings.max_iter, &m_qp_ipm_arg);
    }
};

template<typename T>
struct is_fixed_end_time_multiple_shooting : std::false_type {};

template<typename ControlProblem, unsigned N_segs, template<typename, typename, typename, int> class Integrator, int DiffOptions>
struct is_fixed_end_time_multiple_shooting<laopt_tools::MultipleShooting<ControlProblem, N_segs, Integrator, DiffOptions>> : std::true_type
{
    static_assert((ControlProblem::Options & laopt_tools::FreeEndTime) == 0, "Multiple shooting transcription must be fixed end time");
};

template<typename T>
struct is_bs_problem : std::false_type {};

template<typename UserCode, typename scalar_t>
struct is_bs_problem<BSProblem<UserCode, scalar_t>> : std::true_type {};

template<typename Problem>
class SQPSolver<Problem, HPIPMSolver> : public SQPBase<SQPSolver<Problem, HPIPMSolver>, Problem, HPIPMSolver>
{
    using Base = SQPBase<SQPSolver<Problem, HPIPMSolver>, Problem, HPIPMSolver>;
    using UserCode = typename Problem::UserCode;
    static_assert(is_bs_problem<Problem>::value, "HPIPM requires a block sparse problem (BSProblem)");
    static_assert(is_fixed_end_time_multiple_shooting<UserCode>::value, "HPIPM only works with laopt_tools::MultipleShooting");

    using ControlProblem = typename UserCode::ControlProblem;
    static_assert(ControlProblem::NP == 0, "HPIPM doesn't support parameters");

public:
    explicit SQPSolver(const std::shared_ptr<Problem>& prob) : Base(prob)
    {
        this->m_qp_solver.set_problem_dims(UserCode::N, ControlProblem::NX, ControlProblem::NU,
                                           ControlProblem::NG, ControlProblem::NG0, ControlProblem::NGF);
    }
};

} // namespace laopt

#endif // LAOPT_HPIPM_INTERFACE_HPP
